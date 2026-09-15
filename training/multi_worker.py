"""Multi-game worker process for crforge_gym.

One OS process hosts K CRForgeEnv instances (one bridge socket each) and drives
them with a pipelined step loop: all K step requests are sent first, then all K
replies are collected, so the JVM round trips overlap instead of serializing
across processes. This keeps the per-game logic byte-identical to the
single-game path (same env, same wrappers, same opponent code) -- it only
changes the process/transport layout.

Pairs with MultiBridgeVecEnv (multi_vec_env.py), which implements the SB3
VecEnv contract on top of one pipe per worker process.

Worker command protocol (parent -> worker), all messages tuples:
    ("reset", seeds | None)          -> ("reset_ok", obs(K,1094) f32, masks(K,21) bool)
    ("step", actions(K,3) int64)     -> ("step_ok", obs, rew(K,) f32, done(K,) bool,
                                                  infos(list[dict]), masks)
    ("has_attr", name)               -> ("has_attr", bool)
    ("env_method", (name, args, kw)) -> ("method", list_of_results)
    ("get_attr", name)               -> ("attr", list_of_values)
    ("set_attr", (name, value))      -> ("ok",)
    ("close", None)                  -> (process exits)
"""

import time

import numpy as np


def _make_selfplay_opponent(snapshot_path: str, label: str):
    """Self-play opponent that reloads the trainer snapshot when its mtime changes."""
    import os

    from crforge_gym.opponents import SelfPlayOpponent

    class ReloadingSelfPlayOpponent(SelfPlayOpponent):
        def __init__(self):
            super().__init__(model=None)
            self._loaded_mtime = None
            self._last_check = 0.0
            self._announced = False

        def _maybe_reload(self):
            now = time.time()
            if now - self._last_check < 2.0:
                return
            self._last_check = now
            try:
                mtime = os.path.getmtime(snapshot_path)
            except OSError:
                return
            if self._loaded_mtime is not None and mtime <= self._loaded_mtime:
                return
            from sb3_contrib import MaskablePPO

            try:
                self._install_model(MaskablePPO.load(snapshot_path, device="cpu"))
                self._loaded_mtime = mtime
                if not self._announced:
                    self._announced = True
                    print(f"[multi-worker {label}] self-play opponent loaded snapshot",
                          flush=True)
            except Exception:
                pass  # file mid-write; retry on next check

        def act(self, obs_raw, player="red", obs_flat=None):
            self._maybe_reload()
            if self.model is None:
                return None
            return super().act(obs_raw, player=player, obs_flat=obs_flat)

    return ReloadingSelfPlayOpponent()


def build_env(port: int, blue_deck, red_deck, opponent_mode: str,
              snapshot_path: str | None, label: str):
    """Build one fully wrapped CRForgeEnv (identical stack to the single-game path)."""
    from crforge_gym import CRForgeEnv
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    if opponent_mode == "selfplay":
        opponent = _make_selfplay_opponent(snapshot_path or "", label)
    else:
        opponent = opponent_mode  # built-in name: "random" / "rule_based" / "noop"

    env = CRForgeEnv(
        endpoint=f"tcp://localhost:{port}",
        ticks_per_step=15,
        opponent=opponent,
        binary_obs=True,
        blue_deck=blue_deck,
        red_deck=red_deck,
    )
    env = EpisodeStatsWrapper(env)
    env = ActionMaskedWrapper(env)
    return env


def multi_worker_main(conn, worker_idx, ports, blue_decks, red_decks,
                      opponent_mode, snapshot_path):
    """Entry point for one multi-game worker process."""
    from stable_baselines3.common.vec_env.patch_gym import _patch_env

    k = len(ports)
    label = f"w{worker_idx}p{ports[0]}"
    envs = []
    for i in range(k):
        env = build_env(ports[i], blue_decks[i], red_decks[i], opponent_mode,
                        snapshot_path, f"{label}e{i}")
        envs.append(_patch_env(env))

    conn.send(("ready", envs[0].observation_space, envs[0].action_space))

    while True:
        try:
            cmd, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "reset":
            seeds = payload
            obs_list = []
            mask_list = []
            for i, env in enumerate(envs):
                seed = None
                if seeds is not None:
                    seed = seeds[i]
                o, _info = env.reset(seed=seed)
                obs_list.append(np.asarray(o, dtype=np.float32))
                mask_list.append(np.asarray(env.action_masks(), dtype=bool))
            conn.send(("reset_ok", np.stack(obs_list), np.stack(mask_list)))

        elif cmd == "step":
            actions = payload
            # Pipelining: issue all K requests first (K sockets), then collect.
            for i, env in enumerate(envs):
                env.step_async(np.asarray(actions[i]))
            obs_list = []
            rew_list = []
            done_list = []
            info_list = []
            for env in envs:
                o, r, terminated, truncated, info = env.step_wait()
                done = bool(terminated or truncated)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    # SB3 SubprocVecEnv worker semantics: stash the terminal
                    # observation and auto-reset, return the post-reset obs.
                    info["terminal_observation"] = o
                    o, _reset_info = env.reset()
                obs_list.append(np.asarray(o, dtype=np.float32))
                rew_list.append(float(r))
                done_list.append(done)
                info_list.append(info)
            mask_list = [np.asarray(env.action_masks(), dtype=bool) for env in envs]
            conn.send((
                "step_ok",
                np.stack(obs_list),
                np.asarray(rew_list, dtype=np.float32),
                np.asarray(done_list, dtype=bool),
                info_list,
                np.stack(mask_list),
            ))

        elif cmd == "has_attr":
            try:
                envs[0].get_wrapper_attr(payload)
                conn.send(("has_attr", True))
            except AttributeError:
                conn.send(("has_attr", False))

        elif cmd == "env_method":
            name, args, kwargs = payload
            results = []
            for env in envs:
                method = env.get_wrapper_attr(name)
                results.append(method(*args, **kwargs))
            conn.send(("method", results))

        elif cmd == "is_wrapped":
            from stable_baselines3.common.env_util import is_wrapped

            conn.send(("is_wrapped", [is_wrapped(env, payload) for env in envs]))

        elif cmd == "get_attr":
            conn.send(("attr", [env.get_wrapper_attr(payload) for env in envs]))

        elif cmd == "set_attr":
            name, value = payload
            for env in envs:
                env.set_wrapper_attr(name, value)
            conn.send(("ok",))

        elif cmd == "close":
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass
            return

        else:
            conn.send(("error", f"unknown command {cmd!r}"))

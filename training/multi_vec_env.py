"""SB3 VecEnv on top of multi-game worker processes (see multi_worker.py).

Layout: W worker processes x K games each = N envs. Compared with SB3's
SubprocVecEnv (1 process + 1 bridge JVM per env) this collapses the process
count and, crucially, returns the action masks together with every step/reset
payload -- SB3's MaskablePPO fetches masks via env_method("action_masks")
EVERY step, which on SubprocVecEnv is an extra pickle round trip per env per
step. Here that becomes a cache lookup.

Only the process/transport layout differs; each game still runs the exact
same env + wrappers + opponent code as the single-game path.
"""

from typing import Any, Optional, Sequence

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class MultiBridgeVecEnv(VecEnv):
    """VecEnv where each worker process hosts K pipelined game sessions."""

    def __init__(self, worker_specs: list[dict], start_method: str = "fork",
                 handshake_timeout: float = 300.0):
        """worker_specs: one dict per worker process:
        {"ports": [...], "blue_decks": [...], "red_decks": [...],
         "opponent": "selfplay"|"rule_based"|"random", "snapshot_path": str|None}
        """
        import multiprocessing as mp

        self._specs = worker_specs
        self._sizes = [len(s["ports"]) for s in worker_specs]
        num_envs = sum(self._sizes)

        ctx = mp.get_context(start_method)
        self.conns = []
        self.processes = []
        for idx, spec in enumerate(worker_specs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_import_worker_main(),
                args=(child_conn, idx, spec["ports"], spec["blue_decks"],
                      spec["red_decks"], spec.get("opponent", "selfplay"),
                      spec.get("snapshot_path")),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.conns.append(parent_conn)
            self.processes.append(proc)

        # Handshake: spaces from the first worker (all workers share the layout).
        obs_space = act_space = None
        for c in self.conns:
            kind, obs_space, act_space = _recv_with_timeout(c, handshake_timeout)
            if kind != "ready":
                raise RuntimeError(f"multi-worker handshake failed: {kind!r}")

        super().__init__(num_envs, obs_space, act_space)

        self._masks_cache = np.ones(
            (num_envs, 2 + 4 + 15), dtype=bool)  # replaced on first reset
        self._reset_infos: list = [{} for _ in range(num_envs)]
        self.waiting = False
        self.closed = False

    # -- helpers ----------------------------------------------------------
    def _chunks(self):
        out = []
        start = 0
        for size in self._sizes:
            out.append(slice(start, start + size))
            start += size
        return out

    # -- VecEnv API -------------------------------------------------------
    def reset(self) -> VecEnvObs:
        for c in self.conns:
            c.send(("reset", None))
        obs_list = []
        mask_list = []
        for c in self.conns:
            kind, obs, masks = _recv(c)
            if kind != "reset_ok":
                raise RuntimeError(f"multi-worker reset failed: {kind!r} {obs!r}")
            obs_list.append(obs)
            mask_list.append(masks)
        self._masks_cache = np.concatenate(mask_list, axis=0)
        self._reset_infos = [{} for _ in range(self.num_envs)]
        return np.concatenate(obs_list, axis=0)

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        chunks = self._chunks()
        for c, sl in zip(self.conns, chunks):
            c.send(("step", np.ascontiguousarray(actions[sl])))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        mask_list = []
        for c in self.conns:
            kind, obs, rew, done, infos, masks = _recv(c)
            if kind != "step_ok":
                raise RuntimeError(f"multi-worker step failed: {kind!r}")
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.extend(infos)
            mask_list.append(masks)
        self._masks_cache = np.concatenate(mask_list, axis=0)
        self.waiting = False
        return (
            np.concatenate(obs_list, axis=0),
            np.concatenate(rew_list, axis=0),
            np.concatenate(done_list, axis=0),
            info_list,
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for c in self.conns:
                try:
                    c.recv()
                except Exception:
                    pass
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=15)
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
        self.closed = True

    # -- attribute/method routing ----------------------------------------
    def has_attr(self, attr_name: str) -> bool:
        # Fast local answer for the mask method (must be True for MaskablePPO);
        # anything else: ask the workers once.
        if attr_name in ("action_masks",):
            return True
        for c in self.conns:
            c.send(("has_attr", attr_name))
        return all(_recv(c)[1] for c in self.conns)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None,
                   **method_kwargs) -> list[Any]:
        if method_name == "action_masks" and indices is None:
            # Served from cache: masks travel with every reset/step payload.
            return [self._masks_cache[i] for i in range(self.num_envs)]
        for c in self.conns:
            c.send(("env_method", (method_name, method_args, method_kwargs)))
        results: list[Any] = []
        for c in self.conns:
            kind, res = _recv(c)
            if kind != "method":
                raise RuntimeError(f"multi-worker env_method failed: {kind!r}")
            results.extend(res)
        if indices is not None:
            return [results[i] for i in indices]
        return results

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        for c in self.conns:
            c.send(("get_attr", attr_name))
        results: list[Any] = []
        for c in self.conns:
            kind, res = _recv(c)
            if kind != "attr":
                raise RuntimeError(f"multi-worker get_attr failed: {kind!r}")
            results.extend(res)
        if indices is not None:
            return [results[i] for i in indices]
        return results

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for c in self.conns:
            c.send(("set_attr", (attr_name, value)))
        for c in self.conns:
            _recv(c)

    def env_is_wrapped(self, wrapper_class: type[gymnasium.Wrapper],
                       indices: VecEnvIndices = None) -> list[bool]:
        for c in self.conns:
            c.send(("is_wrapped", wrapper_class))
        results: list[bool] = []
        for c in self.conns:
            kind, res = _recv(c)
            if kind != "is_wrapped":
                raise RuntimeError(f"multi-worker is_wrapped failed: {kind!r}")
            results.extend(res)
        if indices is not None:
            return [results[i] for i in indices]
        return results

    def seed(self, seed: Optional[int] = None) -> Sequence[None | int]:
        return [None for _ in range(self.num_envs)]

    def render(self, mode: str | None = None):
        return None

    def get_images(self):
        return [None for _ in range(self.num_envs)]


def _import_worker_main():
    from crforge_gym.multi_worker import multi_worker_main

    return multi_worker_main


def _recv(conn):
    msg = conn.recv()
    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
        raise RuntimeError(f"multi-worker error: {msg[1]}")
    return msg


def _recv_with_timeout(conn, timeout: float):
    if not conn.poll(timeout):
        raise TimeoutError(
            f"multi-worker handshake timed out after {timeout:.0f}s")
    return _recv(conn)

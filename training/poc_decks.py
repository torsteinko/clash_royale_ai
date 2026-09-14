#!/usr/bin/env python3
"""POC: self-play PPO with two different decks (hog cycle vs giant beatdown) on CRForge.

Env knobs: POC_STEPS, POC_SAVE, POC_LOG. Requires a bridge server on :9876.
"""
import os
import time

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from crforge_gym import CRForgeEnv
from crforge_gym.opponents import SelfPlayOpponent
from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

BLUE = ["hogrider", "musketeer", "cannon", "skeletons", "icespirits", "fireball", "log", "zap"]
RED = ["giant", "witch", "minions", "musketeer", "fireball", "arrows", "valkyrie", "knight"]
STEPS = int(os.environ.get("POC_STEPS", "400000"))
SAVE = os.environ.get("POC_SAVE", "/tmp/crforge_poc/models/ppo_decks")
LOG = os.environ.get("POC_LOG", "/tmp/crforge_poc/logs_decks")
INTERVAL = 20000


class SelfPlayCb(BaseCallback):
    def __init__(self, opponent, interval, save_dir):
        super().__init__(0)
        self.opponent = opponent
        self.interval = interval
        self.save_dir = save_dir
        self.last = 0

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._snap("init")

    def _on_step(self):
        if self.num_timesteps - self.last >= self.interval:
            self._snap(f"step_{self.num_timesteps}")
            self.last = self.num_timesteps
        return True

    def _snap(self, tag):
        p = os.path.join(self.save_dir, f"self_play_{tag}")
        self.model.save(p)
        self.opponent.model = MaskablePPO.load(p)
        print(f"[selfplay] opponent updated @ {self.num_timesteps} ({tag})", flush=True)


class LogCb(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.ep = self.w = self.l = self.d = 0

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:
            self.ep += 1
            o = info.get("game_outcome", "?")
            self.w += o == "win"
            self.l += o == "loss"
            self.d += o == "draw"
            if self.ep % 25 == 0:
                print(f"[{self.num_timesteps} steps, ep {self.ep}] w/l/d = {self.w}/{self.l}/{self.d}",
                      flush=True)
        return True


def main():
    opp = SelfPlayOpponent(model=None)
    env = CRForgeEnv(endpoint="tcp://localhost:9876", ticks_per_step=15, opponent=opp,
                     binary_obs=True, blue_deck=BLUE, red_deck=RED)
    env = EpisodeStatsWrapper(env)
    env = ActionMaskedWrapper(env)
    model = MaskablePPO("MlpPolicy", env, policy_kwargs={"net_arch": [512, 256]},
                        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
                        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                        vf_coef=0.5, max_grad_norm=0.5, seed=7, verbose=1,
                        tensorboard_log=LOG)
    opp.model = model
    t0 = time.time()
    model.learn(total_timesteps=STEPS,
                callback=CallbackList([LogCb(),
                                       SelfPlayCb(opp, INTERVAL, os.path.dirname(SAVE) or "models")]))
    dt = time.time() - t0
    model.save(SAVE)
    print(f"trained {STEPS} steps in {dt:.0f}s ({STEPS / dt:.0f} fps); saved to {SAVE}", flush=True)

    for name in ["rule_based", "noop"]:
        ev = CRForgeEnv(endpoint="tcp://localhost:9876", ticks_per_step=15, opponent=name,
                        binary_obs=True, blue_deck=BLUE, red_deck=RED)
        ev = EpisodeStatsWrapper(ev)
        ev = ActionMaskedWrapper(ev)
        mr, sr = evaluate_policy(model, ev, n_eval_episodes=20)
        print(f"eval vs {name}: mean_reward={mr:.2f} +/- {sr:.2f}", flush=True)
        ev.close()


if __name__ == "__main__":
    main()

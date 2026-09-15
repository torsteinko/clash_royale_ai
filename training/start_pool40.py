#!/usr/bin/env python3
"""Detached launcher: pool-40 run on the multi-game stack (4 workers x 10 games)."""
import os
import subprocess

ARGS = [
    "/home/admeraas/venvs/clash/bin/python",
    "/home/admeraas/crforge/multi_selfplay_train.py",
    "--num-envs", "40",
    "--multi-k", "10",
    "--steps", "50000000",
    "--base-port", "9890",
    "--red-pool", "all",
    "--blue-pool", "hog,hog,hog,hog,hog,hog,hog,hog,giant,logb,xbow,lava,yard,rg,golem,miner",
    "--eval-every", "2000000",
    "--eval-episodes", "3",
    "--snapshot-interval", "250000",
    "--ent-coef", "0.02",
    "--resume",
    "--save", "/home/admeraas/runs/pool12/models/ppo_multi",
    "--logdir", "/home/admeraas/runs/pool12/logs",
]
ENV = dict(os.environ)
ENV.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    # The parent's PPO update + per-step policy forward are the serial bottleneck;
    # workers are pinned back to 1 thread inside multi_worker_main.
    "CRFORGE_PARENT_THREADS": "4",
})
log = open("/home/admeraas/runs/pool12.log", "ab")
proc = subprocess.Popen(ARGS, stdout=log, stderr=log, env=ENV,
                        start_new_session=True, cwd="/home/admeraas/crforge")
print("pool40 launched (multi-k 10)")

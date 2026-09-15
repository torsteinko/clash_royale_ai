"""GPU/CPU throughput probe for the batched sim (milestone M5 preview).

Run from the repo root:
    python -m gpusim.tests.bench_gpu            # GPU if available, else CPU
    python -m gpusim.tests.bench_gpu 8192 300   # batch size, ticks
"""
from __future__ import annotations

import sys
import time

import torch

from gpusim.env import make_sim


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"device: {name} (compute capability {cap[0]}.{cap[1]})")
        print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    else:
        print(f"device: CPU (torch {torch.__version__}) — no CUDA available")
    default_b = 4096 if dev == "cuda" else 256
    b = int(sys.argv[1]) if len(sys.argv) > 1 else default_b
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    sim = make_sim("fidelity/patched", batch_size=b, device=dev)
    knight = sim.t.card_index["knight"]
    mus = sim.t.card_index["musketeer"]
    # blue knight advances; red musketeer shoots -> exercises movement, targeting,
    # combat and projectile flight every tick
    sim.deploy(0, torch.full((b,), knight), torch.full((b,), 9.0), torch.full((b,), 20.0))
    sim.deploy(1, torch.full((b,), mus), torch.full((b,), 9.0), torch.full((b,), 12.0))

    sim.tick(20)  # warmup
    t0 = time.perf_counter()
    sim.tick(ticks)
    dt = time.perf_counter() - t0
    rate = b * ticks / dt
    print(f"B={b}: {ticks} ticks in {dt:.2f}s = {rate:,.0f} env-steps/s ({dev})")


if __name__ == "__main__":
    main()

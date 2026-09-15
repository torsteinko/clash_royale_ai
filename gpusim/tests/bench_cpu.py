"""Quick CPU throughput probe for the batched sim (not a test).

Run: python3 -m gpusim.tests.bench_cpu [BATCH]
"""
import sys
import time

import torch

from gpusim.env import make_sim


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    sim = make_sim("fidelity/patched", batch_size=b, device="cpu")
    knight = sim.t.card_index["knight"]
    sim.deploy(
        side=0,
        card_idx=torch.full((b,), knight),
        x=torch.full((b,), 9.0, dtype=torch.float32),
        y=torch.full((b,), 20.0, dtype=torch.float32),
    )
    n = 200
    # warmup
    sim.tick(10)
    t0 = time.perf_counter()
    sim.tick(n)
    dt = time.perf_counter() - t0
    print(f"B={b}: {n} ticks in {dt:.3f}s = {b * n / dt:,.0f} env-steps/s (CPU, {sim.device})")


if __name__ == "__main__":
    main()

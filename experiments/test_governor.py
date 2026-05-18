#!/usr/bin/env python3
"""Governor integration test — thrash + collapse + dynamic λ."""
import sys, numpy as np
from pathlib import Path
PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO)); sys.path.insert(0, str(PROJECT))
from os_runtime.governor import Governor

rng = np.random.RandomState(42)
gov = Governor()
gov.dyn_lambda.set_base(3.0)

print("Governor Simulation — 200 tokens")
print(f"  {'tok':>4s} {'λ':>5s} {'k':>3s} {'thrash':>8s} {'fault':>6s} {'coll':>6s} {'cons':>5s} {'div':>5s}")

lam_history, k_history = [], []
for i in range(200):
    if i < 50:  # Normal
        fault = rng.random() < 0.1; unique_exp = rng.randint(8, 15)
        ram_mb = 2000; entropy = 0.15; token_class = 'default'
    elif i < 100:  # Thrash
        fault = rng.random() < 0.6; unique_exp = rng.randint(30, 60)
        ram_mb = 3800; entropy = 0.10; token_class = 'default'
    elif i < 120:  # Collapse
        fault = False; unique_exp = rng.randint(1, 3)
        ram_mb = 1800; entropy = 0.003; token_class = 'repetitive'
    elif i < 150:  # Both
        fault = rng.random() < 0.7; unique_exp = rng.randint(2, 5)
        ram_mb = 3900; entropy = 0.004; token_class = 'repetitive'
    else:  # Recovery
        fault = False; unique_exp = rng.randint(12, 20)
        ram_mb = 2200; entropy = 0.18; token_class = 'default'

    lam, k = gov.update(fault, unique_exp, ram_mb, entropy,
                        rng.randint(0, 32000), (i > 0 and i % 7 == 0), token_class)
    lam_history.append(lam); k_history.append(k)
    if i % 25 == 0 or i in [49, 99, 119, 149, 199]:
        print(f"  {i:4d} {lam:5.1f} {k:3d} {gov.thrash.current_level:>8s} "
              f"{gov.thrash.stats()['fault_rate']:.2f} {gov.collapse.collapse_risk:5.2f} "
              f"{'Y' if gov.conservative_mode else 'N':>5s} "
              f"{'Y' if gov.force_diversity else 'N':>5s}")

print()
for start, end, name in [(0,50,'Normal'),(50,100,'Thrash'),(100,120,'Collapse'),
                          (120,150,'Both'),(150,200,'Recovery')]:
    print(f"  {name:<12s}: avg_λ={np.mean(lam_history[start:end]):.1f}, avg_k={np.mean(k_history[start:end]):.1f}")

print("\nGovernor behavior:")
print("  Normal:     λ≈3.0, k=8 — standard")
print("  Thrash:     λ↑ (>4.5), k↓ — reduce I/O")
print("  Collapse:   λ↓ (≈0), k=8 — force diversity")
print("  Both:       collapse wins → diversity priority")
print("  Recovery:   λ→3.0, k→8")

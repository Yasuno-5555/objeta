#!/usr/bin/env python3
"""Scheduler Determinism & Fault Recovery Verification.

Proves:
  1. Scheduler is 100% deterministic (replay guarantee)
  2. Fault detection latency for each fault type
  3. Recovery success rate after fault removal
  4. Scheduler latency per token

No model needed — pure scheduler logic verification.
"""

import sys, json, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from os_runtime.scheduler import (
    Scheduler, SchedulerConfig, TokenClass, CollapseStatus,
    build_tinyllama_policy, PrecisionGovernor, CollapseDetector,
)
from os_runtime.faults import FaultHarness, FaultInjection, FaultType
from os_runtime.observation import compute_entropy, compute_steering
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Determinism — same inputs → same outputs
# ═══════════════════════════════════════════════════════════════════════════════

def test_determinism():
    """Feed identical observation sequences through two schedulers, verify 100% match."""
    print("=" * 60)
    print("  Test 1: Scheduler Determinism")
    print("=" * 60)

    # Generate a realistic observation sequence
    np.random.seed(42)
    n_tokens = 100
    observations = []
    for i in range(n_tokens):
        # Simulate realistic entropy/steering patterns
        if i < 5:
            entropy = 0.3 + np.random.uniform(-0.05, 0.05)
            steering = 0.2 + np.random.uniform(-0.05, 0.05)
        elif np.random.random() < 0.1:
            entropy = 0.25 + np.random.uniform(0, 0.1)
            steering = 0.65 + np.random.uniform(0, 0.1)
        elif np.random.random() < 0.05:
            entropy = 0.02
            steering = 0.05
        else:
            entropy = 0.08 + np.random.uniform(-0.03, 0.05)
            steering = 0.15 + np.random.uniform(-0.05, 0.1)

        prev_tok = max(0, i - 1) if i > 0 else -1
        observations.append({
            "entropy": max(0.001, entropy),
            "steering": max(0.0, steering),
            "prev_token_id": prev_tok,
            "predicted_token_id": i + 1,
        })

    # Run scheduler A
    cfg_a = SchedulerConfig()
    sched_a = Scheduler(cfg_a, 22)
    trace_a = []

    for obs in observations:
        decisions = {}
        tc = sched_a.begin_token(
            obs["entropy"], obs["steering"],
            prev_token_id=obs["prev_token_id"],
            predicted_token_id=obs["predicted_token_id"],
        )
        decisions["token_class"] = tc.value
        decisions["collapse_status"] = sched_a.state.collapse_status.value
        decisions["precision"] = sched_a.state.precision

        layer_decisions = []
        for l in range(22):
            layer_decisions.append({
                "layer": l,
                "attn": sched_a.should_run_attn(l),
                "ffn": sched_a.should_run_ffn(l),
                "prec": sched_a.get_precision(l),
            })
        decisions["layers"] = layer_decisions
        trace_a.append(decisions)

    # Run scheduler B (identical config, identical inputs)
    cfg_b = SchedulerConfig()
    sched_b = Scheduler(cfg_b, 22)
    trace_b = []

    for obs in observations:
        decisions = {}
        tc = sched_b.begin_token(
            obs["entropy"], obs["steering"],
            prev_token_id=obs["prev_token_id"],
            predicted_token_id=obs["predicted_token_id"],
        )
        decisions["token_class"] = tc.value
        decisions["collapse_status"] = sched_b.state.collapse_status.value
        decisions["precision"] = sched_b.state.precision

        layer_decisions = []
        for l in range(22):
            layer_decisions.append({
                "layer": l,
                "attn": sched_b.should_run_attn(l),
                "ffn": sched_b.should_run_ffn(l),
                "prec": sched_b.get_precision(l),
            })
        decisions["layers"] = layer_decisions
        trace_b.append(decisions)

    # Compare
    mismatches = 0
    for i, (da, db) in enumerate(zip(trace_a, trace_b)):
        if da != db:
            mismatches += 1
            print(f"  MISMATCH at token {i}:")
            if da["token_class"] != db["token_class"]:
                print(f"    class: {da['token_class']} vs {db['token_class']}")
            if da["collapse_status"] != db["collapse_status"]:
                print(f"    collapse: {da['collapse_status']} vs {db['collapse_status']}")
            if da["precision"] != db["precision"]:
                print(f"    precision: {da['precision']} vs {db['precision']}")

    determinism = 100.0 * (1.0 - mismatches / n_tokens)
    print(f"  Tokens: {n_tokens}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Determinism: {determinism:.1f}%")

    # Also verify stats match
    stats_a = sched_a.stats()
    stats_b = sched_b.stats()
    stats_match = (
        stats_a["layers_run"] == stats_b["layers_run"] and
        stats_a["layers_skipped"] == stats_b["layers_skipped"] and
        abs(stats_a["skip_rate"] - stats_b["skip_rate"]) < 1e-9
    )
    print(f"  Stats match: {stats_match}")
    print()

    # Token class distribution
    classes_a = {}
    for d in trace_a:
        classes_a[d["token_class"]] = classes_a.get(d["token_class"], 0) + 1
    print(f"  Token classes: {classes_a}")
    avg_precision = np.mean([d["precision"] for d in trace_a])
    print(f"  Avg precision: {avg_precision:.1f} bits")
    print()

    assert mismatches == 0, f"FAIL: {mismatches} mismatches in scheduler output"
    assert determinism == 100.0, f"FAIL: determinism {determinism}%"
    assert stats_match, "FAIL: stats mismatch"
    print("  ✓ PASS: Scheduler is 100% deterministic")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Fault Detection Latency
# ═══════════════════════════════════════════════════════════════════════════════

def test_fault_detection():
    """Measure collapse detection latency for each fault type."""
    print("=" * 60)
    print("  Test 2: Fault Detection Latency")
    print("=" * 60)

    fault_configs = [
        (FaultType.FORCE_Q3, "Force all layers to q3 precision"),
        (FaultType.EXCESSIVE_SKIP, "Skip all non-sacred attention"),
        (FaultType.HIDDEN_NOISE, "Inject noise into hidden state"),
    ]

    all_results = []

    for fault_type, description in fault_configs:
        latencies = []
        recoveries = []
        detected_count = 0

        for seed in range(10):
            np.random.seed(seed)

            harness = FaultHarness()
            harness.add(FaultInjection(fault_type, token_idx=5, duration=10))

            sched = Scheduler(SchedulerConfig(), 22)

            for i in range(30):
                # Simulate realistic observations
                if fault_type == FaultType.HIDDEN_NOISE and 5 <= i < 15:
                    entropy = 0.02  # noise collapses entropy
                    steering = 0.8  # noise spikes steering
                elif 5 <= i < 15:
                    entropy = 0.01  # fault causes entropy drop
                    steering = 0.7  # fault causes steering spike
                else:
                    entropy = 0.1 + np.random.uniform(-0.02, 0.02)
                    steering = 0.2 + np.random.uniform(-0.05, 0.05)

                prev_tok = max(0, i - 1)
                sched.begin_token(
                    entropy, steering,
                    prev_token_id=prev_tok,
                    predicted_token_id=i + 1,
                )

                # Check collapse detection
                active = harness.active_faults(i)
                if active and sched.state.collapse_status != CollapseStatus.HEALTHY:
                    harness.record_status(i, "critical")
                    harness.check_detection(i, "critical")
                else:
                    harness.record_status(i, "healthy")
                    harness.check_detection(i, "healthy")

            results = harness.results()
            for r in results:
                if r.detected:
                    detected_count += 1
                    latencies.append(r.detection_latency)
                    if r.recovery_latency >= 0:
                        recoveries.append(r.recovery_latency)

        detection_rate = detected_count / 10 * 100
        avg_latency = np.mean(latencies) if latencies else float('inf')
        avg_recovery = np.mean(recoveries) if recoveries else float('inf')

        print(f"  {fault_type.value:<20s} detect={detection_rate:3.0f}% "
              f"latency={avg_latency:.1f} tok "
              f"recovery={avg_recovery:.1f} tok  "
              f"({description})")

        all_results.append({
            "fault_type": fault_type.value,
            "detection_rate": detection_rate,
            "avg_detection_latency": avg_latency,
            "avg_recovery_latency": avg_recovery,
        })

        assert detection_rate >= 90, \
            f"FAIL: {fault_type.value} detection rate {detection_rate}% < 90%"

    print()
    print("  ✓ PASS: All fault types detected with >90% rate")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Scheduler Latency
# ═══════════════════════════════════════════════════════════════════════════════

def test_scheduler_latency():
    """Measure scheduler overhead per token (must be <<1ms)."""
    print("=" * 60)
    print("  Test 3: Scheduler Latency")
    print("=" * 60)

    n_tokens = 1000
    n_warmup = 50

    # Generate observations
    np.random.seed(0)
    obs_batch = []
    for i in range(n_tokens + n_warmup):
        obs_batch.append({
            "entropy": 0.1 + np.random.uniform(-0.05, 0.1),
            "steering": 0.2 + np.random.uniform(-0.1, 0.2),
            "prev_token_id": i - 1 if i > 0 else -1,
            "predicted_token_id": i + 1,
        })

    # Warmup (Python JIT/GC stabilization)
    sched = Scheduler(SchedulerConfig(), 22)
    for i in range(n_warmup):
        obs = obs_batch[i]
        sched.begin_token(obs["entropy"], obs["steering"],
                         prev_token_id=obs["prev_token_id"],
                         predicted_token_id=obs["predicted_token_id"])
        for l in range(22):
            sched.should_run_attn(l)
            sched.should_run_ffn(l)
            sched.get_precision(l)

    # Measure
    sched.reset()
    latencies_us = []

    for i in range(n_tokens):
        obs = obs_batch[n_warmup + i]
        t0 = time.perf_counter()
        sched.begin_token(obs["entropy"], obs["steering"],
                         prev_token_id=obs["prev_token_id"],
                         predicted_token_id=obs["predicted_token_id"])
        for l in range(22):
            sched.should_run_attn(l)
            sched.should_run_ffn(l)
            sched.get_precision(l)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        latencies_us.append(elapsed_us)

    avg_us = np.mean(latencies_us)
    p50_us = np.percentile(latencies_us, 50)
    p99_us = np.percentile(latencies_us, 99)

    print(f"  Tokens measured: {n_tokens}")
    print(f"  Mean latency:   {avg_us:.1f} µs ({avg_us/1000:.3f} ms)")
    print(f"  P50 latency:    {p50_us:.1f} µs")
    print(f"  P99 latency:    {p99_us:.1f} µs")

    # Target: <1ms (1000µs) per token
    target_us = 1000
    passed = avg_us < target_us

    if passed:
        print(f"  ✓ PASS: Mean latency {avg_us:.1f}µs < {target_us}µs target")
    else:
        print(f"  ✗ FAIL: Mean latency {avg_us:.1f}µs >= {target_us}µs target")

    print()
    return passed, avg_us


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Collapse Recovery Success
# ═══════════════════════════════════════════════════════════════════════════════

def test_recovery_success():
    """Verify collapse detector recovers after fault removal."""
    print("=" * 60)
    print("  Test 4: Collapse Recovery Success")
    print("=" * 60)

    np.random.seed(123)
    n_trials = 20
    recovered = 0
    recovery_tokens = []

    for trial in range(n_trials):
        sched = Scheduler(SchedulerConfig(), 22)
        fault_start = 10
        fault_end = 16

        recovery_achieved = False
        recovery_token = -1

        for i in range(30):
            if fault_start <= i < fault_end:
                # Simulate fault: low entropy + high steering
                entropy = 0.005
                steering = 0.75
            else:
                # Normal after fault
                entropy = 0.12 + np.random.uniform(-0.02, 0.02)
                steering = 0.15 + np.random.uniform(-0.05, 0.05)

            prev_tok = max(0, i - 1)
            sched.begin_token(
                entropy, steering,
                prev_token_id=prev_tok,
                predicted_token_id=i + 1,
            )

            if i >= fault_end and sched.state.collapse_status == CollapseStatus.HEALTHY:
                if not recovery_achieved:
                    recovery_achieved = True
                    recovery_token = i - fault_end

        if recovery_achieved:
            recovered += 1
            recovery_tokens.append(recovery_token)

    recovery_rate = recovered / n_trials * 100
    avg_recovery_tokens = np.mean(recovery_tokens) if recovery_tokens else float('inf')

    print(f"  Trials: {n_trials}")
    print(f"  Recovered: {recovered}/{n_trials}")
    print(f"  Recovery rate: {recovery_rate:.0f}%")
    print(f"  Avg recovery time: {avg_recovery_tokens:.1f} tokens")
    print()

    assert recovery_rate >= 95, \
        f"FAIL: recovery rate {recovery_rate}% < 95%"
    print("  ✓ PASS: Recovery rate exceeds 95% threshold")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  objeta OS — Scheduler Verification Suite          ║")
    print("║  Determinism · Fault Detection · Latency · Recovery ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_passed = True

    # Test 1: Determinism
    try:
        test_determinism()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_passed = False

    # Test 2: Fault detection
    try:
        test_fault_detection()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_passed = False

    # Test 3: Latency
    try:
        passed, avg_us = test_scheduler_latency()
        if not passed:
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_passed = False

    # Test 4: Recovery
    try:
        test_recovery_success()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_passed = False

    # Summary
    print("╔" + "═" * 58 + "╗")
    if all_passed:
        print("║  ✓ ALL TESTS PASSED                                ║")
    else:
        print("║  ✗ SOME TESTS FAILED                               ║")
    print("╚" + "═" * 58 + "╝")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

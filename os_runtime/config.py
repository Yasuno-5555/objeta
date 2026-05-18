"""Configuration loader — bridges static analysis (objeta analyze) to runtime.

Reads phase_profile.json, strategy.json, and quantization_plan.json
and produces a SchedulerConfig ready for the runtime.
"""

import json
from pathlib import Path

from .scheduler import SchedulerConfig


def load_phase_profile(path: str | Path) -> dict:
    """Load a phase_profile.json from objeta analyze."""
    with open(path) as f:
        return json.load(f)


def load_strategy(path: str | Path) -> dict:
    """Load a strategy.json from objeta strategy."""
    with open(path) as f:
        return json.load(f)


def config_from_strategy(strategy: dict) -> SchedulerConfig:
    """Convert objeta strategy.json to SchedulerConfig.

    Maps:
      family → SchedulerConfig.family
      dominance → SchedulerConfig.backbone
      fusion_ratio → SchedulerConfig.fusion_ratio
      steering_layers → temporal stride policy
    """
    family = strategy.get("family", "residual_transport")
    dominance = strategy.get("dominance", "AttentionBandwidth")
    executor = strategy.get("executor_config", {})

    # Map SensitivityDominance to backbone string
    backbone = {
        "AttentionBandwidth": "attention",
        "FfnCoherence": "ffn",
        "SteeringBackbone": "steering",
    }.get(dominance, "attention")

    fusion_ratio = executor.get("fusion_ratio", 0.5)

    # Family B with MoE gets temporal stride
    temporal_stride = 0
    if "spherical" in family.lower() and fusion_ratio < 0.5:
        temporal_stride = 2

    return SchedulerConfig(
        family=family,
        backbone=backbone,
        fusion_ratio=fusion_ratio,
        temporal_stride=temporal_stride,
        safe_skip_ceiling=0.30,
    )


def config_from_phase_profile(profile: dict) -> SchedulerConfig:
    """Convert phase_profile.json to SchedulerConfig.

    Uses phase structure and inversion layers to set thresholds.
    """
    family = profile.get("family", "ResidualTransport")
    phase = profile.get("phase", "Split2D")

    # Map Rust enums to config strings
    family_str = {
        "ResidualTransport": "residual_transport",
        "SphericalSteering": "spherical_steering",
    }.get(family, "residual_transport")

    # Use inversion zone to set steering thresholds
    inversion_onset = profile.get("inversion_onset")
    steering_active_min = 0.5
    if inversion_onset is not None:
        # Earlier inversion = more steering activity
        steering_active_min = max(0.35, 0.5 - inversion_onset * 0.01)

    fusion_ratio = 1.0  # Default: no fusion on dense models
    if family_str == "spherical_steering":
        phase_str = profile.get("phase", "")
        if "Mixed" in phase_str:
            fusion_ratio = 0.33
        elif "Collapse" in phase_str:
            fusion_ratio = 0.5
        else:
            fusion_ratio = 0.67

    return SchedulerConfig(
        family=family_str,
        backbone="attention" if family_str == "residual_transport" else "steering",
        fusion_ratio=fusion_ratio,
        steering_active_min=steering_active_min,
    )


def load_runtime_config(
    strategy_path: str | Path | None = None,
    profile_path: str | Path | None = None,
) -> SchedulerConfig:
    """Load runtime configuration from static analysis output.

    Precedence: strategy.json > phase_profile.json > defaults
    """
    if strategy_path is not None and Path(strategy_path).exists():
        return config_from_strategy(load_strategy(strategy_path))

    if profile_path is not None and Path(profile_path).exists():
        return config_from_phase_profile(load_phase_profile(profile_path))

    return SchedulerConfig()

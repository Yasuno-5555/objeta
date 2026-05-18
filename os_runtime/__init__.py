"""objeta OS Runtime — v1.0 API (frozen).

LLM inference as adaptive dynamical resource allocation.
observe → classify → allocate → execute

Public API (frozen):
  OSRuntime    — main entry point: wraps LLM + Scheduler
  SchedulerConfig — OS configuration
  TraceReplay  — record → replay
  FaultHarness — collapse detector testing

Usage:
  from os_runtime import OSRuntime, SchedulerConfig

  os = OSRuntime(llm, config)
  tokens = os.generate("prompt", tokenizer)
  trace = os.trace  # TokenLog list
  os.save_trace("trace.jsonl")
"""

__version__ = "1.0.0"

from .scheduler import (
    Scheduler,
    SchedulerConfig,
    TokenClass,
    CollapseStatus,
    LayerPolicy,
    PrecisionGovernor,
    CollapseDetector,
    build_tinyllama_policy,
)
from .observation import (
    ObservationPipeline,
    compute_entropy,
    compute_steering,
    compute_attention_divergence,
)
from .logging import (
    RuntimeLogger,
    TokenLog,
    LayerAction,
    LogLevel,
)
from .replay import TraceReplay
from .faults import (
    FaultHarness,
    FaultInjection,
    FaultType,
    FaultTestResult,
)
from .os_llm import OSLLM
from .config import load_runtime_config


class OSRuntime:
    """Frozen v1.0 API — OS-augmented LLM runtime.

    This is THE public entry point. All internal modules are accessible
    but this class is the supported interface.

    Lifecycle:
        os = OSRuntime(llm, config)
        tokens = os.generate(prompt, tokenizer)
        trace = os.trace
        os.save_trace("trace.jsonl")
    """

    def __init__(self, llm, config: SchedulerConfig | None = None,
                 logger: RuntimeLogger | None = None,
                 fault_harness: FaultHarness | None = None):
        self.config = config or SchedulerConfig()
        self._os_llm = OSLLM(llm, self.config, logger, fault_harness)

    def generate(self, prompt, tokenizer=None, max_tokens: int = 512,
                 temperature: float = 0.7, top_k: int = 40) -> list[int]:
        """Generate tokens with full OS scheduling.

        Returns list of token IDs. Trace is available via .trace property.
        """
        return self._os_llm.generate(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, tokenizer=tokenizer,
        )

    @property
    def trace(self) -> list:
        """List of TokenLog objects from the last generation run."""
        return self._os_llm.logger.token_logs

    @property
    def scheduler(self) -> Scheduler:
        return self._os_llm.scheduler

    @property
    def stats(self) -> dict:
        return self._os_llm.logger.run_summary()

    def save_trace(self, path: str):
        """Save the trace as a replayable JSON-lines file."""
        from .replay import TraceReplay
        replay = TraceReplay(tokens=self.trace)
        replay.save(path)

    def report(self) -> str:
        """One-line status report."""
        return self._os_llm.report()

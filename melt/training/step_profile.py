"""Opt-in step profiler: where does one optimizer step go? (``run.step_profile``)

Two instruments, both off unless ``run.step_profile.enabled`` is true:

1. **Phase marks.** A timestamp (a CUDA event plus the host clock) at every
   boundary of the step: dataloader, host-to-device copy, the three forward
   phases, the loss, the four backward phases, gradient clipping, the
   optimizer. Events are recorded in stream order and resolved a step late, so
   nothing here synchronises the device inside the step. The marks go to
   ``<output_dir>/step_phases_rank<r>.jsonl`` raw; the arithmetic lives in
   ``projects/ablation-campaign/scaling-bench/profile/analyze_profile.py`` so a
   different split can be made without re-running a job.

   The backward boundaries are gradient hooks, not model edits: the gradient of
   the loss, of the decoder logits, of the adapter output, of the encoder
   output, and the accumulation of the encoder's first parameter, which is the
   last thing the backward pass touches.

2. **torch.profiler windows.** One non-final micro-batch and the final one
   (which runs through the optimizer step), on the steps named in
   ``trace_steps``. Phases are wrapped in ``record_function`` ranges so a
   trace can be cut per phase and the NCCL kernels attributed to the
   all-reduce they belong to.

Neither changes what the model computes. With the option off, none of this is
imported by the training path beyond one ``None`` check per hook site.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext

import torch

from melt.logging_utils import get_logger
from transformers import TrainerCallback


logger = get_logger(__name__)


class _CudaClock:
    """CUDA events: timestamps taken on the device, in stream order."""

    def record(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def elapsed_ms(self, start, end) -> float:
        return float(start.elapsed_time(end))

    def done(self, event) -> bool:
        return bool(event.query())

    def wait(self, event) -> None:
        event.synchronize()


class _HostClock:
    """Host timestamps standing in for events; for CPU-only runs and tests."""

    def record(self) -> float:
        return time.perf_counter()

    def elapsed_ms(self, start: float, end: float) -> float:
        return (end - start) * 1000.0

    def done(self, event) -> bool:
        return True

    def wait(self, event) -> None:
        return None


def _first_grad_tensor(value) -> torch.Tensor | None:
    """The first tensor that takes a gradient inside a module output."""
    if isinstance(value, torch.Tensor):
        return value if value.requires_grad else None
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_grad_tensor(item)
            if found is not None:
                return found
    return None


class StepProfiler:
    """Phase marks and profiler windows for one rank. See the module docstring."""

    def __init__(self, cfg: dict, output_dir: str, grad_accum: int, rank: int, clock=None):
        self.output_dir = output_dir
        self.grad_accum = max(int(grad_accum), 1)
        self.rank = int(rank)
        self.active = self.rank in [int(r) for r in (cfg.get("ranks") or [0])]
        self.trace_steps = {int(s) for s in (cfg.get("trace_steps") or [])}
        self.trace_micro = {int(m) for m in (cfg.get("trace_micro") or [])}
        self.trace_dir = cfg.get("trace_dir") or os.path.join(output_dir, "profile")
        self.clock = clock or (_CudaClock() if torch.cuda.is_available() else _HostClock())

        self._step = -1
        self._micro = -1
        self._in_micro = False
        self._marks: list[tuple[str, int, object, float]] = []
        self._meta: list[dict] = []
        self._traced = False
        self._closed: list[dict] = []
        self._attached = False
        self._model: torch.nn.Module | None = None
        self._unused: list[dict] = []

        self._prof = None
        self._tracing = False
        self._open_ranges: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    def mark(self, name: str) -> None:
        if not self.active:
            return
        self._marks.append((name, self._micro, self.clock.record(), time.perf_counter()))

    def _enter(self, name: str) -> None:
        if self._tracing and name not in self._open_ranges:
            ctx = torch.autograd.profiler.record_function(name)
            ctx.__enter__()
            self._open_ranges[name] = ctx

    def _exit(self, name: str) -> None:
        ctx = self._open_ranges.pop(name, None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    def _close_all_ranges(self) -> None:
        for name in list(self._open_ranges):
            self._exit(name)

    # ------------------------------------------------------------------
    # Step and micro-batch boundaries (called by MELTTrainer)
    # ------------------------------------------------------------------

    def begin_step(self, global_step: int) -> None:
        """First call of an optimizer step: the dataloader is about to be read."""
        if not self.active:
            return
        event, cpu = self.clock.record(), time.perf_counter()
        if self._marks:
            # The new step's first event is also the previous step's last: the
            # interval between them is the step the framework times.
            self._marks.append(("close", -1, event, cpu))
            self._closed.append(self._take_record())
            self._resolve(keep_last=1)
        self._step = int(global_step)
        self._micro = -1
        self._meta = []
        self._traced = False
        self._marks = [("dl0", -1, event, cpu)]

    def end_dataloader(self) -> None:
        self.mark("dl1")

    def begin_micro(self, ddp_model) -> None:
        if not self.active:
            return
        self._micro += 1
        self._in_micro = True
        sync = bool(getattr(ddp_model, "require_backward_grad_sync", True))
        self._meta.append({"micro": self._micro, "sync": sync})
        final = self._micro == self.grad_accum - 1
        want_window = self._step in self.trace_steps and (final or self._micro in self.trace_micro)
        if want_window and not self._tracing:
            self._start_trace("final" if final else f"micro{self._micro}")
        self.mark("mb0")

    def note_micro(self, cuts: int | None, audio_s: float | None) -> None:
        if self.active and self._meta:
            self._meta[-1].update(cuts=cuts, audio_s=audio_s)

    def end_micro(self) -> None:
        if not self.active:
            return
        self.mark("mb1")
        self._in_micro = False
        if self._meta and torch.cuda.is_available():
            self._meta[-1]["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if self._micro == 0 and len(self._unused) < 8:
            self._note_unused_gradients()
        if self._tracing and self._micro != self.grad_accum - 1:
            self._stop_trace()

    def end_step(self) -> None:
        """After the optimizer step: close the window that began at the final micro-batch."""
        if not self.active:
            return
        self.mark("step_end")
        if self._tracing:
            self._stop_trace()

    def _note_unused_gradients(self) -> None:
        """Which trainable parameters got no gradient from the first micro-batch of a step.

        ``find_unused_parameters`` exists for these. Listing them says whether
        the cost can be avoided by touching them, or whether the graph itself
        leaves parameters out (a different fix).
        """
        if self._model is None:
            return
        missing = [n for n, p in self._model.named_parameters() if p.requires_grad and p.grad is None]
        self._unused.append({"step": self._step, "count": len(missing), "names": missing})

    # ------------------------------------------------------------------
    # Profiler windows
    # ------------------------------------------------------------------

    def _start_trace(self, tag: str) -> None:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self._prof = profile(activities=activities)
        self._trace_tag = tag
        self._prof.start()
        self._tracing = True
        self._traced = True

    def _stop_trace(self) -> None:
        self._close_all_ranges()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._prof.stop()
        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(self.trace_dir, f"trace_rank{self.rank}_step{self._step}_{self._trace_tag}.json.gz")
        self._prof.export_chrome_trace(path)
        logger.info("[step_profile] wrote %s", path)
        self._prof = None
        self._tracing = False

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def attach(self, model: torch.nn.Module, ddp_model: torch.nn.Module) -> None:
        """Install the hooks once. *model* is the unwrapped MELT model, *ddp_model* what the Trainer calls."""
        if self._attached or not self.active:
            return
        self._attached = True
        self._model = model
        stack = getattr(model, "audio_stack", None)
        encoder = getattr(stack, "encoder", None)
        adapter = getattr(stack, "adapter", None)
        decoder = getattr(model, "text_decoder", None)

        model.register_forward_pre_hook(self._fwd_pre)
        model.register_forward_hook(self._fwd_post)
        if ddp_model is not model:
            ddp_model.register_forward_hook(self._ddp_post)
        if encoder is not None:
            encoder.register_forward_pre_hook(self._enc_pre)
            encoder.register_forward_hook(self._enc_post)
            first = next((p for p in encoder.parameters() if p.requires_grad), None)
            if first is not None:
                first.register_post_accumulate_grad_hook(self._g_encend)
        if adapter is not None:
            adapter.register_forward_pre_hook(self._ad_pre)
            adapter.register_forward_hook(self._ad_post)
        if decoder is not None:
            decoder.register_forward_pre_hook(self._dec_pre)
            decoder.register_forward_hook(self._dec_post)

    def _live(self) -> bool:
        return self._in_micro and torch.is_grad_enabled()

    def _fwd_pre(self, module, args, kwargs=None):
        if self._live():
            self.mark("fwd0")
            self._enter("melt/fwd_pre")

    def _ddp_post(self, module, args, output):
        if self._live():
            self.mark("ddp1")

    def _enc_pre(self, module, args, kwargs=None):
        if self._live():
            self._exit("melt/fwd_pre")
            self.mark("enc0")
            self._enter("melt/fwd_enc")

    def _enc_post(self, module, args, output):
        if self._live():
            self.mark("enc1")
            self._exit("melt/fwd_enc")
            self._hook(output, "g_enc", exit_range="melt/bwd_adapter", enter_range="melt/bwd_enc")

    def _ad_pre(self, module, args, kwargs=None):
        if self._live():
            self.mark("ad0")
            self._enter("melt/fwd_adapter")

    def _ad_post(self, module, args, output):
        if self._live():
            self.mark("ad1")
            self._exit("melt/fwd_adapter")
            self._enter("melt/fwd_merge")
            self._hook(output, "g_adapter", exit_range="melt/bwd_dec", enter_range="melt/bwd_adapter")

    def _dec_pre(self, module, args, kwargs=None):
        if self._live():
            self._exit("melt/fwd_merge")
            self.mark("dec0")
            self._enter("melt/fwd_dec")

    def _dec_post(self, module, args, output):
        if self._live():
            self.mark("dec1")
            self._exit("melt/fwd_dec")
            self._enter("melt/fwd_loss")
            self._hook(getattr(output, "logits", output), "g_logits", exit_range="melt/bwd_loss", enter_range="melt/bwd_dec")

    def _fwd_post(self, module, args, output):
        if self._live():
            self.mark("fwd1")
            self._exit("melt/fwd_loss")
            loss = getattr(output, "loss", None)
            if loss is None and isinstance(output, (tuple, list)) and output:
                loss = output[0]
            self._hook(loss, "bwd0", enter_range="melt/bwd_loss")

    def _hook(self, value, name: str, exit_range: str | None = None, enter_range: str | None = None) -> None:
        """Mark when the gradient of *value* is ready, and move the trace range along."""
        tensor = _first_grad_tensor(value)
        if tensor is None:
            return

        def fire(grad):
            self.mark(name)
            if exit_range:
                self._exit(exit_range)
            if enter_range:
                self._enter(enter_range)

        tensor.register_hook(fire)

    def _g_encend(self, param) -> None:
        if self._in_micro:
            self.mark("g_encend")
            self._exit("melt/bwd_enc")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _take_record(self) -> dict:
        record = {"step": self._step, "traced": self._traced, "micro": self._meta, "marks": self._marks}
        return record

    def _resolve(self, keep_last: int = 0) -> None:
        """Write every closed step that is old enough that its events are done."""
        while len(self._closed) > keep_last:
            self._write(self._closed.pop(0))

    def _write(self, record: dict) -> None:
        marks = record["marks"]
        if not marks:
            return
        self.clock.wait(marks[-1][2])
        t0_event, t0_cpu = marks[0][2], marks[0][3]
        out = {
            "step": record["step"],
            "traced": record["traced"],
            "micro": record["micro"],
            "marks": [
                [name, micro, round(self.clock.elapsed_ms(t0_event, event), 3), round((cpu - t0_cpu) * 1000.0, 3)]
                for name, micro, event, cpu in marks
            ],
        }
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"step_phases_rank{self.rank}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")

    def close(self) -> None:
        if not self.active:
            return
        if self._tracing:
            self._stop_trace()
        if self._unused:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"unused_grads_rank{self.rank}.json"), "w", encoding="utf-8") as fh:
                json.dump(self._unused, fh)
        if self._marks:
            # The run ended mid-interval: the last step is closed where it stopped.
            self._marks.append(("close", -1, self.clock.record(), time.perf_counter()))
            self._closed.append(self._take_record())
            self._marks = []
        self._resolve()


class StepProfileCallback(TrainerCallback):
    """Trainer events the profiler cannot see from inside ``training_step``."""

    def __init__(self, profiler: StepProfiler):
        self.profiler = profiler

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.profiler.mark("opt0")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.profiler.mark("opt1")

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.end_step()

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.close()


def profiling_span(profiler: StepProfiler | None, start: str, end: str):
    """Context manager marking *start* and *end* around a block; a no-op without a profiler."""
    if profiler is None:
        return nullcontext()

    class _Span:
        def __enter__(self_inner):
            profiler.mark(start)

        def __exit__(self_inner, *exc):
            profiler.mark(end)
            return False

    return _Span()

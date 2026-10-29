"""Trainer callbacks of the MELT training entrypoint."""

from omegaconf import DictConfig

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..logging_utils import get_logger


logger = get_logger(__name__)


class StopAfterStep(TrainerCallback):
    """Save a checkpoint at an optimizer step and end the run there, with the LR schedule untouched.

    The warmup, the stable phase and the decay of a learning-rate schedule are all read off
    ``trainer.max_steps``, so ending a run early by lowering it changes the learning rate at every step.
    This callback keeps the schedule of the full run and stops at *step*: the steps it trains are the
    first *step* steps of that run, and the checkpoint it leaves is one the full run would have
    written (optimizer, scheduler and sampler state included), so resuming it carries the run on.

    ``state.global_step`` counts optimizer steps. The save is forced because *step* need not be a
    multiple of ``save_steps``; when it is, the run still saves once, since both requests set the same flag.

    Args:
        step: Optimizer step after which the run ends (``>= 1``).
    """

    def __init__(self, step: int):
        if step < 1:
            raise ValueError(f"run.stop_after_step must be >= 1, got {step}")
        self.step = int(step)

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Ask for the save and the stop once *step* optimizer steps are done."""
        if state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
            logger.info(
                f"run.stop_after_step={self.step}: saving a checkpoint at step {state.global_step} "
                f"and ending the run (max_steps={state.max_steps} still sets the LR schedule)"
            )
        return control


def stop_after_step_from_config(run_cfg: DictConfig | dict, max_steps: int) -> int | None:
    """Read ``run.stop_after_step`` and check it against the run's ``max_steps``.

    Args:
        run_cfg: The ``run`` section of the training config.
        max_steps: ``TrainingArguments.max_steps`` (``-1`` when the run is epoch-based).

    Returns:
        The step to stop after, or ``None`` when the option is off (the default).

    Raises:
        ValueError: If the value is not a positive integer, or lies beyond ``max_steps`` (the run
            would end at ``max_steps`` first and the option would silently do nothing).
    """
    value = run_cfg.get("stop_after_step", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"run.stop_after_step must be a positive integer, got {value!r}")
    if 0 < max_steps < value:
        raise ValueError(
            f"run.stop_after_step={value} is beyond trainer.max_steps={max_steps}: "
            "the run would end at max_steps first"
        )
    return value

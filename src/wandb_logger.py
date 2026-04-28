"""Optional Weights & Biases logger.

Behaviour:

* If the ``wandb`` package is not installed, every call is a no-op.
* If the env var ``WANDB_MODE=disabled`` is set, every call is a no-op
  (this is also wandb's standard "off switch").
* Otherwise the first call to ``init`` calls ``wandb.init`` with the
  project from ``WANDB_PROJECT`` (default: ``"graphite"``) and any
  config dict passed in.

The module deliberately exposes a tiny surface (``init``, ``log``,
``finish``) so the call sites in ``pretrain.py`` stay short.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

try:
    import wandb as _wandb  # type: ignore
except Exception:
    _wandb = None  # type: ignore


_run = None  # set after init()
_disabled = False


def _is_off() -> bool:
    if _wandb is None:
        return True
    if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
        return True
    if os.environ.get('GRAPHITE_DISABLE_WANDB', '').lower() in ('1', 'true', 'yes'):
        return True
    return False


def init(config: Optional[Mapping[str, Any]] = None,
         name: Optional[str] = None,
         project: Optional[str] = None) -> None:
    """Start a run.  Safe to call when wandb is unavailable or disabled."""
    global _run, _disabled
    if _is_off():
        _disabled = True
        return
    if _run is not None:
        return
    project = project or os.environ.get('WANDB_PROJECT', 'graphite-registration')
    try:
        _run = _wandb.init(project=project, name=name,
                           config=dict(config) if config else None,
                           reinit=False)
    except Exception as exc:
        print('[wandb] init failed, continuing without it:', exc)
        _disabled = True
        _run = None
        return

    # Tell wandb that epoch-keyed metrics should be charted against
    # ``epoch`` rather than the implicit global step.  Iteration metrics
    # keep the default global-step axis.  Glob patterns require a
    # recent-enough wandb but the call is wrapped in try/except below.
    try:
        _wandb.define_metric('epoch')
        _wandb.define_metric('train/*', step_metric='epoch')
        _wandb.define_metric('val/*', step_metric='epoch')
        _wandb.define_metric('eval_modelnet/*', step_metric='epoch')
        _wandb.define_metric('best/*', step_metric='epoch')
        _wandb.define_metric('lr', step_metric='epoch')
    except Exception:
        # Older wandb versions or non-default backends — don't crash.
        pass


def log(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Log a dict of scalars; no-op when wandb is unavailable."""
    if _disabled or _run is None or _wandb is None:
        return
    try:
        _wandb.log(dict(metrics), step=step)
    except Exception as exc:
        print('[wandb] log failed:', exc)


def finish() -> None:
    global _run
    if _disabled or _run is None or _wandb is None:
        return
    try:
        _wandb.finish()
    except Exception:
        pass
    _run = None


def is_active() -> bool:
    return (not _disabled) and (_run is not None)

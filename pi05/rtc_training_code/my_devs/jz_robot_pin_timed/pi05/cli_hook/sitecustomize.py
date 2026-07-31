"""Opt-in Python startup hook used only by this module's training script."""

from __future__ import annotations

import os


if os.environ.get("JZ_PI05_ENABLE_TRAIN_HOOK") == "1":
    from jz_pi05_train_hook import install

    install()


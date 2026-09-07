#!/usr/bin/env bash
# Zero-shot Walrus evaluation: no training, just load the pretrained
# checkpoint and roll it forward autoregressively on the MEGNO time-
# evolution dataset. This is the baseline every fine-tune is compared to.
set -euo pipefail
cd "$(dirname "$0")"

MEGNO_DATA_SUFFIX=_1024x512 \
MEGNO_N_STEPS_INPUT=5 \
python run_walrus_time_evolution.py

#!/usr/bin/env bash
# Resume a full fine-tune from a previously-saved per-epoch checkpoint for
# more epochs (e.g. continuing epoch4 -> epochs 5-9). Epoch numbering and
# the training-step-count log continue automatically from the checkpoint's
# saved "epoch" field; only FT_RESUME_CKPT and FT_EPOCHS (= additional
# epochs to run, not a new total) need to change vs. the original launch.
set -euo pipefail
cd "$(dirname "$0")"

OMP_NUM_THREADS=16 \
MEGNO_DATA_SUFFIX=_finegrained_1024x512 \
MEGNO_N_STEPS_INPUT=25 \
FT_HELDOUT_MA=60,240,330 \
FT_WINDOW_STRIDE=25 \
FT_TRAIN_MA_STRIDE=3 \
FT_RESUME_CKPT=walrus_time_evolution_finetuned_full_finegrained_in25_epoch4.pt \
FT_EPOCHS=5 \
FT_OUT_NAME=walrus_time_evolution_finetuned_full_finegrained_in25.pt \
python finetune_walrus_time_evolution_full.py

#!/usr/bin/env bash
# FULL (unfrozen-backbone) fine-tune: the entire 1.29B-parameter model,
# including all 40 processor blocks, is trainable -- matches Walrus's own
# recommended recipe (walrus/run_scripts/finetuning_example_distributed_walrus.sh):
# no freezing, gradient_checkpointing_freq=2, plain Adam, lr=1e-4.
# Run from inside this directory (code/).
#
# This example targets the "fair-comparison" fine-grained-dataset config
# (N_STEPS_INPUT=25, predicting years 52-100 from years 2-50, every-3rd MA
# for tractable epoch time). For the original small-dataset config instead,
# drop FT_TRAIN_MA_STRIDE and FT_WINDOW_STRIDE, and set
# MEGNO_DATA_SUFFIX=_1024x512 MEGNO_N_STEPS_INPUT=5.
set -euo pipefail
cd "$(dirname "$0")"

OMP_NUM_THREADS=16 \
MEGNO_DATA_SUFFIX=_finegrained_1024x512 \
MEGNO_N_STEPS_INPUT=25 \
FT_HELDOUT_MA=60,240,330 \
FT_WINDOW_STRIDE=25 \
FT_TRAIN_MA_STRIDE=3 \
FT_EPOCHS=5 \
FT_OUT_NAME=walrus_time_evolution_finetuned_full_finegrained_in25.pt \
python finetune_walrus_time_evolution_full.py

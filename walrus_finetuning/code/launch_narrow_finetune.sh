#!/usr/bin/env bash
# NARROW (frozen-backbone) fine-tune: only the new "megno" field's
# embedding/decoder-head layers are trainable; all 40 Walrus processor
# blocks stay frozen. Run from inside this directory (code/).
#
# Result on the reference run (1024x512, 5-frame input, 45 steps): came back
# numerically identical to zero-shot to 3 decimal places on every held-out
# MA value -- see README "Expected outputs" for the reference numbers.
set -euo pipefail
cd "$(dirname "$0")"

OMP_NUM_THREADS=16 \
MEGNO_DATA_SUFFIX=_1024x512 \
MEGNO_N_STEPS_INPUT=5 \
FT_HELDOUT_MA=60,240,330 \
FT_EPOCHS=1 \
FT_OUT_NAME=walrus_time_evolution_finetuned_1024x512_in5.pt \
python finetune_walrus_time_evolution.py

"""
Fine-tune Walrus on the corrected MEGNO time-evolution dataset, this time
following Walrus's own recommended recipe (walrus/run_scripts/
finetuning_example_distributed_walrus.sh) rather than our narrow
embedding-only approach (finetune_walrus_time_evolution.py):

  - Full model unfrozen (no gradient masking) -- their config system has no
    "freeze everything except the new field" option; they fine-tune
    (near-)everything.
  - model.gradient_checkpointing_freq=2 (their setting; we'd disabled this
    reasoning "128GB RAM, no memory pressure" -- but the narrow run's 220s/
    step at 1024x512 (vs ~12s forward-only) suggests the bottleneck was
    memory *bandwidth* moving all-block activations around, not capacity,
    which checkpointing directly addresses by recomputing instead of storing.
  - optimizer=adam, lr=1e-4 (their config; we'd used AdamW at 2e-4).

Rationale for trying this now: three separate held-out evaluations of the
narrow (embedding+decoder-head-only) approach came back numerically IDENTICAL
to zero-shot (to 3 decimal places) -- strong evidence that scope, not
training amount, was the limiting factor. The "copy the last frame forward"
failure mode lives in the 40 processor blocks, which the narrow approach
never touches.

Same train/held-out MA split as finetune_walrus_time_evolution.py, for a
direct, apples-to-apples comparison.
"""

import copy
import itertools
import os
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map

from run_walrus_megno import CHECKPOINT_PATH, CONFIG_PATH, build_trajectory_example
from run_walrus_time_evolution import N_STEPS_INPUT, load_data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "Data", "Walrus_results")

HELDOUT_MA = [int(x) for x in os.environ.get("FT_HELDOUT_MA", "60,240,330").split(",")]
LR = float(os.environ.get("FT_LR", "1e-4"))
N_EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
WINDOW_STRIDE = int(os.environ.get("FT_WINDOW_STRIDE", "1"))
GRADIENT_CHECKPOINTING_FREQ = int(os.environ.get("FT_GRAD_CKPT_FREQ", "2"))
MAX_STEPS = int(os.environ.get("FT_MAX_STEPS", "0")) or None
# Resume: name of a previously saved fine-tune checkpoint (in OUT_DIR) to
# continue training from, e.g. "walrus_time_evolution_finetuned_full_epoch4.pt".
# Epoch numbering continues from that checkpoint's saved "epoch" + 1 unless
# FT_START_EPOCH overrides it.
RESUME_CKPT = os.environ.get("FT_RESUME_CKPT", "") or None
START_EPOCH_OVERRIDE = os.environ.get("FT_START_EPOCH")


def build_model_for_finetune():
    config = OmegaConf.load(CONFIG_PATH)

    if RESUME_CKPT:
        resume_path = os.path.join(OUT_DIR, RESUME_CKPT)
        print(f"Resuming from fine-tuned checkpoint {resume_path} ...")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        new_field_to_index_map = resumed["field_to_index_map"]
        megno_idx = resumed["megno_idx"]

        model = instantiate(config.model, n_states=max(new_field_to_index_map.values()) + 1,
                             gradient_checkpointing_freq=GRADIENT_CHECKPOINTING_FREQ)
        model.load_state_dict(resumed["model_state_dict"])

        if START_EPOCH_OVERRIDE is not None:
            start_epoch = int(START_EPOCH_OVERRIDE)
        else:
            start_epoch = int(resumed.get("epoch", -1)) + 1
        prior_losses = resumed.get("losses", [])
    else:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)["app"]["model"]
        field_to_index_map = dict(config.data.field_index_map_override)
        new_field_to_index_map = copy.deepcopy(field_to_index_map)
        new_field_to_index_map["megno"] = max(field_to_index_map.values()) + 1
        megno_idx = new_field_to_index_map["megno"]

        model = instantiate(config.model, n_states=max(new_field_to_index_map.values()) + 1,
                             gradient_checkpointing_freq=GRADIENT_CHECKPOINTING_FREQ)
        revised_checkpoint = align_checkpoint_with_field_to_index_map(
            checkpoint_state_dict=checkpoint,
            model_state_dict=model.state_dict(),
            checkpoint_field_to_index_map=field_to_index_map,
            model_field_to_index_map=new_field_to_index_map,
        )
        model.load_state_dict(revised_checkpoint)
        start_epoch = int(START_EPOCH_OVERRIDE) if START_EPOCH_OVERRIDE is not None else 0
        prior_losses = []

    # Full fine-tune: everything stays trainable (default requires_grad=True),
    # matching Walrus's own recipe -- no freezing, no gradient masking.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable scalars (full model): {n_trainable:,}")
    print(f"Starting at epoch {start_epoch} ({len(prior_losses)} prior training steps carried over)")

    device = torch.device("cpu")
    model.to(device)
    return model, config, new_field_to_index_map, megno_idx, device, start_epoch, prior_losses


def make_window_batch(seq, start, device):
    window = seq[start:start + N_STEPS_INPUT + 1]
    return build_trajectory_example(window, device, n_steps_input=N_STEPS_INPUT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stack, ma_values, years = load_data()
    ma_values = list(ma_values)
    train_ma = [ma for ma in ma_values if ma not in HELDOUT_MA and ma < 361]
    ma_stride = int(os.environ.get("FT_TRAIN_MA_STRIDE", "1"))
    if ma_stride != 1:
        train_ma = train_ma[::ma_stride]
    print(f"train MA: {train_ma}")
    print(f"held-out MA: {HELDOUT_MA}")

    print("Loading Walrus checkpoint + config for full fine-tuning...")
    model, config, field_to_index_map, megno_idx, device, start_epoch, losses = build_model_for_finetune()
    print(f"'megno' field index={megno_idx}, gradient_checkpointing_freq={GRADIENT_CHECKPOINTING_FREQ}, "
          f"lr={LR}")

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_years = stack.shape[1]
    max_start = n_years - N_STEPS_INPUT - 1
    all_pairs = list(itertools.product(train_ma, range(0, max_start + 1)))
    pairs = all_pairs[::WINDOW_STRIDE]
    print(f"{len(pairs)} training windows (stride={WINDOW_STRIDE}) out of {len(all_pairs)} possible")

    model.train()
    t0 = time.time()
    step_count = 0
    stop = False
    for epoch in range(start_epoch, start_epoch + N_EPOCHS):
        if stop:
            break
        epoch_losses = []
        for ma, start in pairs:
            if MAX_STEPS is not None and step_count >= MAX_STEPS:
                print(f"Reached FT_MAX_STEPS={MAX_STEPS}, stopping early.")
                stop = True
                break
            step_count += 1
            t_step = time.time()
            mi = ma_values.index(ma)
            seq = stack[mi]
            example = make_window_batch(seq, start, device)
            example["field_indices"] = torch.tensor([megno_idx], device=device)

            inputs, y_target = formatter.process_input(
                example, causal_in_time=model.causal_in_time, predict_delta=True, train=True,
            )
            with torch.no_grad():
                normalization_stats = revin.compute_stats(inputs[0], example["metadata"], epsilon=1e-5)
            normalized_x = revin.normalize_stdmean(inputs[0], normalization_stats)

            y_pred_raw = model(normalized_x, inputs[1], inputs[2].tolist(), metadata=example["metadata"])
            y_pred_raw = revin.denormalize_stdmean(y_pred_raw, normalization_stats)
            y_pred = formatter.process_output(y_pred_raw, example["metadata"])

            loss = F.l1_loss(y_pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            losses.append(loss.item())
            print(f"epoch {epoch} MA={ma:>3d} start={start}  loss={loss.item():.4f}  "
                  f"({time.time() - t_step:.1f}s)", flush=True)

        print(f"epoch {epoch} mean loss: {np.mean(epoch_losses):.4f}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)

        # Per-epoch checkpoint so we can trace held-out performance vs. epoch
        # afterward (and catch overfitting early), not just at the very end.
        out_name = os.environ.get("FT_OUT_NAME", "walrus_time_evolution_finetuned_full.pt")
        base, ext = os.path.splitext(out_name)
        epoch_ckpt_path = os.path.join(OUT_DIR, f"{base}_epoch{epoch}{ext}")
        torch.save({"model_state_dict": model.state_dict(), "field_to_index_map": field_to_index_map,
                    "megno_idx": megno_idx, "losses": losses, "train_ma": train_ma,
                    "heldout_ma": HELDOUT_MA, "epoch": epoch}, epoch_ckpt_path)
        print("Saved epoch checkpoint to", epoch_ckpt_path, flush=True)

    print(f"Total fine-tuning time: {time.time() - t0:.1f}s")

    out_name = os.environ.get("FT_OUT_NAME", "walrus_time_evolution_finetuned_full.pt")
    ckpt_path = os.path.join(OUT_DIR, out_name)
    torch.save({"model_state_dict": model.state_dict(), "field_to_index_map": field_to_index_map,
                "megno_idx": megno_idx, "losses": losses, "train_ma": train_ma,
                "heldout_ma": HELDOUT_MA}, ckpt_path)
    print("Saved final fine-tuned checkpoint to", ckpt_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(losses, "o-")
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 loss")
    ax.set_title("Walrus time-evolution FULL fine-tuning loss")
    fig.tight_layout()
    loss_plot_name = os.path.splitext(out_name)[0] + "_loss.png"
    fig.savefig(os.path.join(OUT_DIR, loss_plot_name), dpi=150)
    print("Saved loss curve")


if __name__ == "__main__":
    main()

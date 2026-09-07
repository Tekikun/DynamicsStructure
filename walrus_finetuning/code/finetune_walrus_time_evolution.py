"""
Fine-tune Walrus on the corrected MEGNO time-evolution dataset (one
continuous 100-yr integration per (a,e) point, checkpointed every 10 yr, for
16 MA values). Same scope as finetune_walrus_megno.py: freeze the pretrained
backbone, only train the 'megno' field's slice of the encoder input-embed
conv and decoder output-head conv (gradient-masked so the other 67 fields'
pretrained weights are untouched).

Train/held-out split: MA=390,420,450,480 are exact duplicates of
MA=30,60,90,120 (periodicity) so they're excluded entirely (would leak).
Of the 12 distinct MA values, 9 are used for training windows and 3
(90, 240, 360) are held out completely for evaluation -- testing whether
fine-tuning improves the *general* megno-field representation, not just
memorizes these specific MA trajectories.
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
from run_walrus_time_evolution import DATA_SUFFIX, N_STEPS_INPUT, load_data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "Data", "Walrus_results")

HELDOUT_MA = [int(x) for x in os.environ.get("FT_HELDOUT_MA", "60,240,330").split(",")]
LR = 2e-4
N_EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
WINDOW_STRIDE = int(os.environ.get("FT_WINDOW_STRIDE", "1"))
DIM_KEY = "2"
GRADIENT_CHECKPOINTING_FREQ = 0
MAX_STEPS = int(os.environ.get("FT_MAX_STEPS", "0")) or None


def build_model_for_finetune():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)["app"]["model"]
    config = OmegaConf.load(CONFIG_PATH)

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

    for p in model.parameters():
        p.requires_grad = False

    proj1 = model.embed[DIM_KEY].proj1.weight
    proj2_w = model.debed[DIM_KEY].proj2.weight
    proj2_b = model.debed[DIM_KEY].proj2.bias
    for p in (proj1, proj2_w, proj2_b):
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in (proj1, proj2_w, proj2_b))
    print(f"Trainable tensors: 3, total trainable scalars: {n_trainable}")

    device = torch.device("cpu")
    model.to(device)
    return model, config, new_field_to_index_map, megno_idx, device, (proj1, proj2_w, proj2_b)


def mask_gradients(proj1, proj2_w, proj2_b, megno_idx):
    if proj1.grad is not None:
        mask = torch.zeros_like(proj1.grad)
        mask[:, megno_idx] = 1.0
        proj1.grad.mul_(mask)
    if proj2_w.grad is not None:
        mask = torch.zeros_like(proj2_w.grad)
        mask[megno_idx] = 1.0
        proj2_w.grad.mul_(mask)
    if proj2_b.grad is not None:
        mask = torch.zeros_like(proj2_b.grad)
        mask[megno_idx] = 1.0
        proj2_b.grad.mul_(mask)


def make_window_batch(seq, start, device):
    window = seq[start:start + N_STEPS_INPUT + 1]
    return build_trajectory_example(window, device, n_steps_input=N_STEPS_INPUT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stack, ma_values, years = load_data()
    ma_values = list(ma_values)
    train_ma = [ma for ma in ma_values if ma not in HELDOUT_MA and ma < 361]
    print(f"train MA: {train_ma}")
    print(f"held-out MA: {HELDOUT_MA}")

    print("Loading Walrus checkpoint + config for fine-tuning...")
    model, config, field_to_index_map, megno_idx, device, trainable_tensors = build_model_for_finetune()
    proj1, proj2_w, proj2_b = trainable_tensors
    print(f"'megno' field index={megno_idx}")

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()
    optimizer = torch.optim.AdamW(trainable_tensors, lr=LR, weight_decay=1e-4, eps=1e-10)

    n_years = stack.shape[1]
    max_start = n_years - N_STEPS_INPUT - 1
    all_pairs = list(itertools.product(train_ma, range(0, max_start + 1)))
    pairs = all_pairs[::WINDOW_STRIDE]
    print(f"{len(pairs)} training windows (stride={WINDOW_STRIDE}) out of {len(all_pairs)} possible")

    model.train()
    t0 = time.time()
    losses = []
    step_count = 0
    stop = False
    for epoch in range(N_EPOCHS):
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
            mask_gradients(proj1, proj2_w, proj2_b, megno_idx)
            optimizer.step()

            epoch_losses.append(loss.item())
            losses.append(loss.item())
            print(f"epoch {epoch} MA={ma:>3d} start={start}  loss={loss.item():.4f}  "
                  f"({time.time() - t_step:.1f}s)")

        print(f"epoch {epoch} mean loss: {np.mean(epoch_losses):.4f}")

    print(f"Total fine-tuning time: {time.time() - t0:.1f}s")

    out_name = os.environ.get("FT_OUT_NAME", "walrus_time_evolution_finetuned.pt")
    ckpt_path = os.path.join(OUT_DIR, out_name)
    torch.save({"model_state_dict": model.state_dict(), "field_to_index_map": field_to_index_map,
                "megno_idx": megno_idx, "losses": losses, "train_ma": train_ma,
                "heldout_ma": HELDOUT_MA}, ckpt_path)
    print("Saved fine-tuned checkpoint to", ckpt_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(losses, "o-")
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 loss")
    ax.set_title("Walrus time-evolution fine-tuning loss")
    fig.tight_layout()
    loss_plot_name = os.path.splitext(out_name)[0] + "_loss.png"
    fig.savefig(os.path.join(OUT_DIR, loss_plot_name), dpi=150)
    print("Saved loss curve")


if __name__ == "__main__":
    main()

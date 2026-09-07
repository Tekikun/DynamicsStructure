"""
Feed our Sun-Jupiter Greek-case MEGNO(a,e) sequence (megno_dense_ma_sequence.npz,
44 frames, MA offset 0..430 deg in 10 deg steps) to the pretrained Walrus
checkpoint in zero-shot mode, following the "Part 2: Non-Well data" pattern
from walrus/demo_notebooks/walrus_example_1_RunningWalrus.ipynb.

Our field ("megno") isn't in Walrus's pretrained field_to_index_map, so we
add a new index for it and use align_checkpoint_with_field_to_index_map to
extend the pretrained checkpoint (new field gets a fresh, untrained
embedding row; everything else keeps pretrained weights).
"""

import copy
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from the_well.data.datasets import WellMetadata

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "walrus.pt")
CONFIG_PATH = os.path.join(HERE, "configs", "extended_config.yaml")
DATA_PATH = os.path.join(HERE, "..", "Data", "MEGNO_maps", "megno_dense_ma_sequence.npz")
OUT_DIR = os.path.join(HERE, "..", "Data", "Walrus_results")

N_STEPS_INPUT = 6  # matches config.data.module_parameters.n_steps_input
MAX_ROLLOUT_STEPS = 200


def load_megno_sequence():
    d = np.load(DATA_PATH)
    stack = d["MEGNO_stack"]  # (T, n_ecc, n_sma)
    ma_offsets = d["MA_offsets_deg"]
    # Clip pathological whfast blow-ups (see megno_map.py docstring) so a few
    # numerical outliers don't dominate normalization; NaNs (ejections) -> a
    # large-but-finite chaotic value.
    stack = np.nan_to_num(stack, nan=50.0)
    stack = np.clip(stack, -10, 50)
    return stack.astype(np.float32), ma_offsets


def build_model_and_checkpoint():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)["app"]["model"]
    config = OmegaConf.load(CONFIG_PATH)

    field_to_index_map = dict(config.data.field_index_map_override)
    new_field_to_index_map = copy.deepcopy(field_to_index_map)
    new_field_to_index_map["megno"] = max(field_to_index_map.values()) + 1

    model = instantiate(config.model, n_states=max(new_field_to_index_map.values()) + 1)

    revised_checkpoint = align_checkpoint_with_field_to_index_map(
        checkpoint_state_dict=checkpoint,
        model_state_dict=model.state_dict(),
        checkpoint_field_to_index_map=field_to_index_map,
        model_field_to_index_map=new_field_to_index_map,
    )
    model.load_state_dict(revised_checkpoint)

    # MPS lacks ConvTranspose3d (used by the decoder), and PYTORCH_ENABLE_MPS_FALLBACK
    # doesn't catch it (hard RuntimeError, not the usual not-implemented path), so
    # fall back to CPU rather than CUDA-only MPS.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, config, new_field_to_index_map, device


def build_trajectory_example(stack, device, n_steps_input=None):
    T, H, W = stack.shape
    D = 1
    C_var = 1
    n_in = N_STEPS_INPUT if n_steps_input is None else n_steps_input

    fields = torch.from_numpy(stack).view(T, H, W, D, C_var).unsqueeze(0)  # (1,T,H,W,D,C)
    input_fields = fields[:, :n_in]
    output_fields = fields[:, n_in:]

    example = {
        "input_fields": input_fields.to(device),
        "output_fields": output_fields.to(device),
        "constant_fields": torch.zeros(1, H, W, D, 0, device=device),
        "boundary_conditions": torch.tensor([[[1, 1], [1, 1], [0, 0]]], device=device),  # e,a: open; dummy D: wall
        "padded_field_mask": torch.tensor([True], device=device),
        "field_indices": None,  # filled in by caller once we know the index
        "metadata": WellMetadata(
            dataset_name="sunjup_greek_megno",
            n_spatial_dims=2,
            field_names={0: ["megno"], 1: [], 2: []},
            spatial_resolution=(H, W, D),
            scalar_names=[],
            constant_field_names={0: [], 1: [], 2: []},
            constant_scalar_names=[],
            boundary_condition_types=[],
            n_files=[],
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[],
        ),
    }
    return example


def rollout_model(model, revin, batch, formatter, max_rollout_steps, device):
    """Adapted from walrus_example_1_RunningWalrus.ipynb (simplified: no mask)."""
    metadata = batch["metadata"]
    batch = {k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
              for k, v in batch.items()}

    inputs, y_ref = formatter.process_input(
        batch, causal_in_time=model.causal_in_time, predict_delta=True, train=False,
    )

    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(y_ref.shape[1], max_rollout_steps)
    train_rollout_limit = 1

    y_ref = y_ref[:, :rollout_steps]
    moving_batch = copy.deepcopy(batch)
    y_preds = []
    for i in range(train_rollout_limit - 1, rollout_steps):
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)
        with torch.no_grad():
            normalization_stats = revin.compute_stats(inputs[0], metadata, epsilon=1e-5)
        normalized_inputs = inputs[:]
        normalized_inputs[0] = revin.normalize_stdmean(normalized_inputs[0], normalization_stats)
        y_pred = model(
            normalized_inputs[0], normalized_inputs[1], normalized_inputs[2].tolist(),
            metadata=metadata,
        )
        y_pred = revin.denormalize_stdmean(y_pred, normalization_stats)
        # Model I/O is (T, B, C, ...) ("channels first with time"); convert back
        # to the Well convention (B, T, ..., C) used by moving_batch/input_fields.
        y_pred = formatter.process_output(y_pred, metadata)
        # causal_in_time=True: model returns one prediction per input timestep
        # (next-step-ahead at each position); only the last one is the actual
        # forecast beyond the current input window.
        y_pred = y_pred[:, -1:]
        # config.trainer.prediction_type == "delta": model output is a residual
        # added to the last input frame, not the absolute next state.
        y_pred = y_pred + moving_batch["input_fields"][:, -1:]
        y_preds.append(y_pred)

        moving_batch["input_fields"] = torch.cat(
            [moving_batch["input_fields"][:, 1:], y_pred], dim=1
        )

    y_pred = torch.cat(y_preds, dim=1)
    return y_pred, y_ref


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stack, ma_offsets = load_megno_sequence()
    print(f"Loaded MEGNO sequence: {stack.shape}, MA offsets {ma_offsets[0]}..{ma_offsets[-1]}")

    print("Loading Walrus checkpoint + config...")
    model, config, field_to_index_map, device = build_model_and_checkpoint()
    megno_index = field_to_index_map["megno"]
    print(f"Using device={device}, 'megno' field index={megno_index} "
          f"(new, previously untrained embedding row)")

    example = build_trajectory_example(stack, device)
    example["field_indices"] = torch.tensor([megno_index], device=device)

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()

    with torch.no_grad():
        y_pred, y_ref = rollout_model(
            model, revin, example, formatter, max_rollout_steps=MAX_ROLLOUT_STEPS, device=device,
        )

    y_pred_np = y_pred[0, ..., 0].squeeze(-1).cpu().numpy()  # (T_out, H, W)
    y_ref_np = y_ref[0, ..., 0].squeeze(-1).cpu().numpy()

    rmse_per_step = np.sqrt(np.mean((y_pred_np - y_ref_np) ** 2, axis=(1, 2)))
    corr_per_step = [
        np.corrcoef(y_pred_np[t].flatten(), y_ref_np[t].flatten())[0, 1]
        for t in range(y_pred_np.shape[0])
    ]

    out_ma = ma_offsets[N_STEPS_INPUT:N_STEPS_INPUT + len(rmse_per_step)]
    for ma, rmse, corr in zip(out_ma, rmse_per_step, corr_per_step):
        print(f"MA+{ma:>3d}deg  rmse={rmse:.3f}  corr={corr:.3f}")

    np.savez(os.path.join(OUT_DIR, "zero_shot_rollout.npz"),
             y_pred=y_pred_np, y_ref=y_ref_np, ma_offsets=out_ma,
             rmse_per_step=rmse_per_step, corr_per_step=corr_per_step)
    print("Saved rollout results to", os.path.join(OUT_DIR, "zero_shot_rollout.npz"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out_ma, corr_per_step, "o-", label="correlation")
    ax.set_xlabel("MA offset (deg)")
    ax.set_ylabel("Pearson correlation (pred vs true)")
    ax.set_title("Zero-shot Walrus rollout accuracy vs. rollout horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "zero_shot_accuracy.png"), dpi=150)
    print("Saved accuracy plot")

    n_show = min(4, y_pred_np.shape[0])
    idxs = np.linspace(0, y_pred_np.shape[0] - 1, n_show).astype(int)
    fig2, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    for col, i in enumerate(idxs):
        vmax = 6
        axes[0, col].imshow(y_ref_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
        axes[0, col].set_title(f"true, MA+{out_ma[i]}deg")
        axes[1, col].imshow(y_pred_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
        axes[1, col].set_title(f"pred, MA+{out_ma[i]}deg")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "zero_shot_maps.png"), dpi=150)
    print("Saved map comparison")


if __name__ == "__main__":
    main()

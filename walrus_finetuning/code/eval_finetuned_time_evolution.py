"""
Evaluate a fine-tuned time-evolution Walrus checkpoint on ALL 16 MA
trajectories (using the same rollout as run_walrus_time_evolution.py, for a
direct zero-shot-vs-finetuned comparison), explicitly separating results for
the 3 fully held-out MA values (90, 240, 360 -- never seen in training) from
the 9 trained-on ones.

Usage: python eval_finetuned_time_evolution.py [checkpoint_name]
"""

import os
import sys

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter

from run_walrus_megno import CONFIG_PATH, MAX_ROLLOUT_STEPS, build_trajectory_example, rollout_model
from run_walrus_time_evolution import N_STEPS_INPUT, OUT_TAG, load_data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "Data", "Walrus_results")


def build_model_from_finetuned(ckpt_name):
    ft_ckpt = torch.load(os.path.join(OUT_DIR, ckpt_name), map_location="cpu", weights_only=False)
    config = OmegaConf.load(CONFIG_PATH)
    field_to_index_map = ft_ckpt["field_to_index_map"]
    megno_idx = ft_ckpt["megno_idx"]

    model = instantiate(config.model, n_states=max(field_to_index_map.values()) + 1,
                         gradient_checkpointing_freq=0)
    model.load_state_dict(ft_ckpt["model_state_dict"])
    model.eval()
    return model, config, megno_idx, ft_ckpt.get("train_ma", []), ft_ckpt.get("heldout_ma", [])


def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "walrus_time_evolution_finetuned.pt"
    tag = os.path.splitext(ckpt_name)[0]

    stack, ma_values, years = load_data()
    print(f"Loaded {stack.shape}, MA values {list(ma_values)}, years {list(years)}")

    eval_ma_env = os.environ.get("MEGNO_EVAL_MA")
    if eval_ma_env:
        eval_ma = {int(x) for x in eval_ma_env.split(",")}
        keep = [i for i, ma in enumerate(ma_values) if int(ma) in eval_ma]
        stack, ma_values = stack[keep], ma_values[keep]
        print(f"Restricted to {len(ma_values)} eval MA values: {list(ma_values)}")

    print(f"Loading fine-tuned checkpoint {ckpt_name}...")
    model, config, megno_idx, train_ma, heldout_ma = build_model_from_finetuned(ckpt_name)
    device = torch.device("cpu")
    model.to(device)
    print(f"train_ma={train_ma}, heldout_ma={heldout_ma}")

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()

    all_corr = {}
    all_pred = {}
    all_ref = {}
    out_years = None
    for mi, ma in enumerate(ma_values):
        seq = stack[mi]
        example = build_trajectory_example(seq, device, n_steps_input=N_STEPS_INPUT)
        example["field_indices"] = torch.tensor([megno_idx], device=device)
        with torch.no_grad():
            y_pred, y_ref = rollout_model(
                model, revin, example, formatter, max_rollout_steps=MAX_ROLLOUT_STEPS, device=device,
            )
        y_pred_np = y_pred[0, ..., 0].squeeze(-1).cpu().numpy()
        y_ref_np = y_ref[0, ..., 0].squeeze(-1).cpu().numpy()
        corr = [np.corrcoef(y_pred_np[t].flatten(), y_ref_np[t].flatten())[0, 1]
                for t in range(y_pred_np.shape[0])]
        out_years = years[N_STEPS_INPUT:N_STEPS_INPUT + len(corr)]
        all_corr[int(ma)] = corr
        all_pred[int(ma)] = y_pred_np
        all_ref[int(ma)] = y_ref_np
        if int(ma) in heldout_ma:
            kind = "HELDOUT"
        elif int(ma) in train_ma:
            kind = "train"
        else:
            kind = "excluded"  # MA=390-480: exact periodic duplicates of 30-120
        print(f"MA={ma:>3d}deg [{kind:8s}] " + " ".join(f"t{y}:{c:.3f}" for y, c in zip(out_years, corr)))

    mean_train = np.mean([np.mean(all_corr[ma]) for ma in train_ma if ma in all_corr])
    mean_heldout = np.mean([np.mean(all_corr[ma]) for ma in heldout_ma if ma in all_corr])
    print(f"\nMean corr, TRAIN MA: {mean_train:.3f}")
    print(f"Mean corr, HELD-OUT MA: {mean_heldout:.3f}")
    print(f"Overall mean corr: {np.mean([np.mean(v) for v in all_corr.values()]):.3f}")

    np.savez(os.path.join(OUT_DIR, f"{tag}_eval_{OUT_TAG}.npz"), ma_values=ma_values, out_years=out_years,
             **{f"corr_MA{ma}": np.array(c) for ma, c in all_corr.items()})

    # Comparison plot against zero-shot (same resolution/input-window config)
    zs_path = os.path.join(OUT_DIR, f"time_evolution_zeroshot_all_ma_{OUT_TAG}.npz")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    if os.path.exists(zs_path):
        zs = np.load(zs_path)
        zs_years = zs["out_years"]
        for ma in heldout_ma:
            key = f"corr_MA{ma}"
            if key in zs:
                ax.plot(zs_years, zs[key], "--", color="gray", alpha=0.5,
                        label="zero-shot (held-out)" if ma == heldout_ma[0] else None)
    for ma in heldout_ma:
        if ma in all_corr:
            ax.plot(out_years, all_corr[ma], "o-", color="tab:red",
                     label="finetuned (held-out)" if ma == heldout_ma[0] else None)
    ax.set_xlabel("year")
    ax.set_ylabel("Pearson correlation (pred vs true)")
    ax.set_title(f"{tag} ({OUT_TAG}): held-out MA values, finetuned vs zero-shot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_heldout_comparison_{OUT_TAG}.png"), dpi=150)
    print("Saved held-out comparison plot")

    if 60 in all_pred:
        y_pred_np, y_ref_np = all_pred[60], all_ref[60]
        n_show = min(len(out_years), 6)
        idxs = np.linspace(0, len(out_years) - 1, n_show).astype(int)
        fig2, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for col, i in enumerate(idxs):
            vmax = 6
            axes[0, col].imshow(y_ref_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
            axes[0, col].set_title(f"true, t={out_years[i]}yr")
            axes[1, col].imshow(y_pred_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
            axes[1, col].set_title(f"{tag}, t={out_years[i]}yr")
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUT_DIR, f"{tag}_MA60_maps_{OUT_TAG}.png"), dpi=150)
        print("Saved MA=60 map comparison")


if __name__ == "__main__":
    main()

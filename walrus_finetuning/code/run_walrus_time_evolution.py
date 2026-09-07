"""
Zero-shot Walrus test on the *corrected* MEGNO time-evolution dataset
(megno_time_evolution_all_ma.py): one continuous 100-year integration per
(a,e) point, checkpointed at 10,20,...,100 yr, for each of 16 MA values.
Loads the Walrus checkpoint once and loops over all 16 MA "videos" (10
frames each) for efficiency.
"""

import os

import numpy as np
import torch
from hydra.utils import instantiate

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter

from run_walrus_megno import (
    MAX_ROLLOUT_STEPS, build_model_and_checkpoint, build_trajectory_example, rollout_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "Data", "MEGNO_maps")
OUT_DIR = os.path.join(HERE, "..", "Data", "Walrus_results")

N_STEPS_INPUT = int(os.environ.get("MEGNO_N_STEPS_INPUT", "6"))
DATA_SUFFIX = os.environ.get("MEGNO_DATA_SUFFIX", "_32x32")
OUT_TAG = os.environ.get("MEGNO_OUT_TAG", DATA_SUFFIX.lstrip("_"))
if N_STEPS_INPUT != 6:
    OUT_TAG += f"_in{N_STEPS_INPUT}"


def load_data():
    d = np.load(os.path.join(DATA_DIR, f"megno_time_evolution_all_ma{DATA_SUFFIX}.npz"))
    stack = d["MEGNO_stack"]  # (n_ma, n_years, H, W)
    ma_values = d["ma_values"]
    years = d["checkpoint_years"]
    stack = np.nan_to_num(stack, nan=50.0)
    stack = np.clip(stack, -10, 50)
    return stack.astype(np.float32), ma_values, years


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stack, ma_values, years = load_data()
    print(f"Loaded {stack.shape}, MA values {list(ma_values)}, years {list(years)}")

    eval_ma_env = os.environ.get("MEGNO_EVAL_MA")
    if eval_ma_env:
        eval_ma = {int(x) for x in eval_ma_env.split(",")}
        keep = [i for i, ma in enumerate(ma_values) if int(ma) in eval_ma]
        stack, ma_values = stack[keep], ma_values[keep]
        print(f"Restricted to {len(ma_values)} eval MA values: {list(ma_values)}")

    print("Loading Walrus checkpoint + config...")
    model, config, field_to_index_map, device = build_model_and_checkpoint()
    megno_index = field_to_index_map["megno"]
    print(f"device={device}, 'megno' field index={megno_index}")

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()

    all_corr = {}
    all_rmse = {}
    all_pred = {}
    all_ref = {}
    out_years = None

    for mi, ma in enumerate(ma_values):
        seq = stack[mi]
        example = build_trajectory_example(seq, device, n_steps_input=N_STEPS_INPUT)
        example["field_indices"] = torch.tensor([megno_index], device=device)
        with torch.no_grad():
            y_pred, y_ref = rollout_model(
                model, revin, example, formatter, max_rollout_steps=MAX_ROLLOUT_STEPS, device=device,
            )
        y_pred_np = y_pred[0, ..., 0].squeeze(-1).cpu().numpy()
        y_ref_np = y_ref[0, ..., 0].squeeze(-1).cpu().numpy()
        rmse = np.sqrt(np.mean((y_pred_np - y_ref_np) ** 2, axis=(1, 2)))
        corr = [np.corrcoef(y_pred_np[t].flatten(), y_ref_np[t].flatten())[0, 1]
                for t in range(y_pred_np.shape[0])]
        out_years = years[N_STEPS_INPUT:N_STEPS_INPUT + len(rmse)]

        all_corr[int(ma)] = corr
        all_rmse[int(ma)] = rmse
        all_pred[int(ma)] = y_pred_np
        all_ref[int(ma)] = y_ref_np

        corr_str = " ".join(f"t{y}:{c:.2f}" for y, c in zip(out_years, corr))
        print(f"MA={ma:>3d}deg  {corr_str}")

    mean_corr_per_ma = {ma: float(np.mean(c)) for ma, c in all_corr.items()}
    print("\nMean correlation per MA value:")
    for ma, mc in mean_corr_per_ma.items():
        print(f"  MA={ma:>3d}deg  mean_corr={mc:.3f}")
    print(f"\nOverall mean corr: {np.mean(list(mean_corr_per_ma.values())):.3f}")

    np.savez(os.path.join(OUT_DIR, f"time_evolution_zeroshot_all_ma_{OUT_TAG}.npz"),
             ma_values=ma_values, out_years=out_years,
             **{f"corr_MA{ma}": np.array(c) for ma, c in all_corr.items()},
             **{f"rmse_MA{ma}": np.array(r) for ma, r in all_rmse.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for ma in ma_values:
        ax.plot(out_years, all_corr[int(ma)], "o-", label=f"MA={ma}", alpha=0.7)
    ax.set_xlabel("year")
    ax.set_ylabel("Pearson correlation (pred vs true)")
    ax.set_title(f"Zero-shot Walrus on corrected MEGNO time evolution, all MA values ({OUT_TAG})")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"time_evolution_zeroshot_all_ma_accuracy_{OUT_TAG}.png"), dpi=150)
    print("Saved accuracy plot")

    # Map comparison for MA=60 (our verified reference case)
    ma60 = 60
    y_pred_np, y_ref_np = all_pred[ma60], all_ref[ma60]
    n_show = min(len(out_years), 6)
    idxs = np.linspace(0, len(out_years) - 1, n_show).astype(int)
    fig2, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    for col, i in enumerate(idxs):
        vmax = 6
        axes[0, col].imshow(y_ref_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
        axes[0, col].set_title(f"true, t={out_years[i]}yr")
        axes[1, col].imshow(y_pred_np[i], origin="lower", vmin=2, vmax=vmax, cmap="viridis")
        axes[1, col].set_title(f"pred, t={out_years[i]}yr")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, f"time_evolution_zeroshot_MA60_maps_{OUT_TAG}.png"), dpi=150)
    print("Saved MA=60 map comparison")


if __name__ == "__main__":
    main()

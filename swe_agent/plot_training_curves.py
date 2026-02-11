#!/usr/bin/env python3
"""
SWE-Agent VERL Training Curve Plotter

Generates **two** sets of training curves from a VERL metrics JSONL file:

1. **Per-Step** — raw metric at each training step (fine-grained view).
2. **Per-Epoch** — metrics aggregated (mean + min/max band) within each epoch.

Both images are saved side by side with ``_step.png`` and ``_epoch.png``
suffixes.  A trajectory summary text file is also produced.

Usage:
    python3 recipe/swe_agent/plot_training_curves.py \\
        --metrics /data1/lmy/workspace/logs/qwen3-4b-swe-train-v3_metrics.jsonl \\
        --trajectories /data1/lmy/workspace/trajectories/qwen3-4b-swe-train-v3 \\
        --output /data1/lmy/workspace/training_curves_v3.png
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# =========================================================================
# Data loading
# =========================================================================

def load_metrics(path: str) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def load_trajectories(traj_dir: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    if not os.path.exists(traj_dir):
        return out
    for fname in sorted(os.listdir(traj_dir)):
        if not fname.endswith(".jsonl"):
            continue
        step = int(fname.replace(".jsonl", ""))
        entries = []
        with open(os.path.join(traj_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        out[step] = entries
    return out


def _pv(v) -> float | None:
    """Parse a value that may be a numpy string repr."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and "np." in v:
        try:
            return float(v.split("(")[1].rstrip(")"))
        except (IndexError, ValueError):
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# =========================================================================
# Per-step extraction
# =========================================================================

def extract_by_step(metrics: list[dict], key: str):
    """Return (steps, values) for a metric key — one point per step."""
    steps, vals = [], []
    for m in metrics:
        data = m.get("data", m)
        if key not in data:
            continue
        v = _pv(data[key])
        if v is None:
            continue
        steps.append(m.get("step", 0))
        vals.append(v)
    return steps, vals


# =========================================================================
# Per-epoch aggregation
# =========================================================================

def _step2epoch(metrics: list[dict]) -> dict[int, int]:
    s2e: dict[int, int] = {}
    for m in metrics:
        data = m.get("data", m)
        ep = data.get("training/epoch")
        if ep is not None:
            s2e[m.get("step", 0)] = int(ep)
    return s2e


def _nearest_epoch(step: int, s2e: dict[int, int]) -> int | None:
    if step in s2e:
        return s2e[step]
    if not s2e:
        return None
    return s2e[min(s2e, key=lambda s: abs(s - step))]


def extract_by_epoch(metrics: list[dict], key: str):
    """Return (epochs, means, mins, maxs) aggregated per epoch."""
    s2e = _step2epoch(metrics)
    buckets: dict[int, list[float]] = defaultdict(list)
    for m in metrics:
        data = m.get("data", m)
        if key not in data:
            continue
        v = _pv(data[key])
        if v is None:
            continue
        ep = data.get("training/epoch")
        if ep is not None:
            ep = int(ep)
        else:
            ep = _nearest_epoch(m.get("step", 0), s2e)
        if ep is None:
            ep = -1
        buckets[ep].append(v)
    if not buckets:
        return [], [], [], []
    epochs, means, mins, maxs = [], [], [], []
    for ep in sorted(buckets):
        a = buckets[ep]
        epochs.append(ep)
        means.append(float(np.mean(a)))
        mins.append(float(np.min(a)))
        maxs.append(float(np.max(a)))
    return epochs, means, mins, maxs


# =========================================================================
# Shared colour palette
# =========================================================================

C_TRAIN   = "#2196F3"
C_VAL     = "#FF5722"
C_REWARD  = "#4CAF50"
C_LOSS    = "#F44336"
C_ENTROPY = "#9C27B0"
C_KL      = "#FF9800"
C_TURNS   = "#00BCD4"
C_GRAD    = "#795548"
C_CLIP    = "#607D8B"


# =========================================================================
# Per-STEP figure
# =========================================================================

def _draw_step_figure(metrics: list[dict], out_path: str):
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("SWE-Agent VERL Training Curves  (per Step)",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           top=0.93, bottom=0.05)
    xl = "Step"

    def _simple(ax, key, color, title, ylabel, ylim=None):
        xs, ys = extract_by_step(metrics, key)
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.4, marker="o",
                    markersize=3, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xl); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    def _band(ax, key_mean, key_max, key_min, color, title, ylabel, ylim=None):
        xs, ys = extract_by_step(metrics, key_mean)
        xsM, ysM = extract_by_step(metrics, key_max)
        xsm, ysm = extract_by_step(metrics, key_min)
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.6, marker="o",
                    markersize=3, label="Mean")
        if xsM and xsm and len(xsM) == len(xsm):
            ax.fill_between(xsM, ysm, ysM, alpha=0.13, color=color,
                            label="Min–Max")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xl); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    # (0,0) Training Reward
    ax = fig.add_subplot(gs[0, 0])
    _band(ax, "critic/score/mean", "critic/score/max", "critic/score/min",
          C_REWARD, "Training Reward (Score)", "Score", (-0.05, 1.05))

    # (0,1) Validation Reward & Accuracy
    ax = fig.add_subplot(gs[0, 1])
    xs1, ys1 = extract_by_step(metrics, "val-aux/swe_agent_simple/reward/mean@1")
    xs2, ys2 = extract_by_step(metrics, "val-core/swe_agent_simple/acc/mean@1")
    if xs1:
        ax.plot(xs1, ys1, color=C_VAL, linewidth=1.6, marker="s",
                markersize=4, label="Val Reward")
    if xs2:
        ax.plot(xs2, ys2, color=C_TRAIN, linewidth=1.4, marker="^",
                markersize=4, label="Val Accuracy", linestyle="--")
    ax.set_title("Validation Reward & Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("Score")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (0,2) PG Loss
    _simple(fig.add_subplot(gs[0, 2]),
            "actor/pg_loss", C_LOSS, "Policy Gradient Loss", "PG Loss")

    # (1,0) Entropy
    _simple(fig.add_subplot(gs[1, 0]),
            "actor/entropy", C_ENTROPY, "Policy Entropy", "Entropy")

    # (1,1) KL
    ax = fig.add_subplot(gs[1, 1])
    xs_kl, ys_kl = extract_by_step(metrics, "rollout_corr/kl")
    xs_pk, ys_pk = extract_by_step(metrics, "actor/ppo_kl")
    if xs_kl:
        ax.plot(xs_kl, ys_kl, color=C_KL, linewidth=1.6, marker="o",
                markersize=3, label="Rollout KL")
    if xs_pk:
        ax.plot(xs_pk, ys_pk, color="#E91E63", linewidth=1.4, marker="^",
                markersize=3, label="PPO KL")
    ax.set_title("KL Divergence", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("KL")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (1,2) Grad Norm
    _simple(fig.add_subplot(gs[1, 2]),
            "actor/grad_norm", C_GRAD, "Gradient Norm", "Grad Norm")

    # (2,0) Response Length
    ax = fig.add_subplot(gs[2, 0])
    _band(ax, "response_length/mean", "response_length/max",
          "response_length/min", C_TRAIN,
          "Response Length (tokens)", "Tokens")

    # (2,1) Turns
    ax = fig.add_subplot(gs[2, 1])
    xs_t, ys_t = extract_by_step(metrics, "num_turns/mean")
    xs_vt, ys_vt = extract_by_step(metrics, "val-aux/num_turns/mean")
    if xs_t:
        ax.plot(xs_t, ys_t, color=C_TURNS, linewidth=1.6, marker="o",
                markersize=3, label="Train Mean")
    if xs_vt:
        ax.plot(xs_vt, ys_vt, color=C_VAL, linewidth=1.4, marker="s",
                markersize=4, label="Val Mean")
    ax.set_title("Number of Turns", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("Turns")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (2,2) Clip Fraction
    _simple(fig.add_subplot(gs[2, 2]),
            "actor/pg_clipfrac", C_CLIP, "PPO Clip Fraction", "Clip Fraction")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step curves]  saved to: {out_path}")


# =========================================================================
# Per-EPOCH figure
# =========================================================================

def _draw_epoch_figure(metrics: list[dict], out_path: str):
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("SWE-Agent VERL Training Curves  (per Epoch)",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           top=0.93, bottom=0.05)
    xl = "Epoch"

    def _one(ax, key, color, title, ylabel, ylim=None):
        epochs, means, mins, maxs = extract_by_epoch(metrics, key)
        if epochs:
            ax.plot(epochs, means, color=color, linewidth=2.2, marker="o",
                    markersize=5, label="Epoch Mean")
            ax.fill_between(epochs, mins, maxs, alpha=0.15, color=color,
                            label="Min–Max")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xl); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # (0,0) Training Reward
    _one(fig.add_subplot(gs[0, 0]),
         "critic/score/mean", C_REWARD,
         "Training Reward (Score)", "Score", (-0.05, 1.05))

    # (0,1) Validation Reward & Accuracy
    ax = fig.add_subplot(gs[0, 1])
    ep_vr, mn_vr, mi_vr, mx_vr = extract_by_epoch(
        metrics, "val-aux/swe_agent_simple/reward/mean@1")
    ep_va, mn_va, _, _ = extract_by_epoch(
        metrics, "val-core/swe_agent_simple/acc/mean@1")
    if ep_vr:
        ax.plot(ep_vr, mn_vr, color=C_VAL, linewidth=2.2, marker="s",
                markersize=5, label="Val Reward")
        ax.fill_between(ep_vr, mi_vr, mx_vr, alpha=0.12, color=C_VAL)
    if ep_va:
        ax.plot(ep_va, mn_va, color=C_TRAIN, linewidth=2, marker="^",
                markersize=5, label="Val Accuracy", linestyle="--")
    ax.set_title("Validation Reward & Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("Score")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # (0,2) PG Loss
    _one(fig.add_subplot(gs[0, 2]),
         "actor/pg_loss", C_LOSS, "Policy Gradient Loss", "PG Loss")

    # (1,0) Entropy
    _one(fig.add_subplot(gs[1, 0]),
         "actor/entropy", C_ENTROPY, "Policy Entropy", "Entropy")

    # (1,1) KL
    ax = fig.add_subplot(gs[1, 1])
    ek, mk, ik, xk = extract_by_epoch(metrics, "rollout_corr/kl")
    ep, mp, _, _ = extract_by_epoch(metrics, "actor/ppo_kl")
    if ek:
        ax.plot(ek, mk, color=C_KL, linewidth=2.2, marker="o",
                markersize=5, label="Rollout KL")
        ax.fill_between(ek, ik, xk, alpha=0.15, color=C_KL)
    if ep:
        ax.plot(ep, mp, color="#E91E63", linewidth=2, marker="^",
                markersize=4, label="PPO KL")
    ax.set_title("KL Divergence", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("KL")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # (1,2) Grad Norm
    _one(fig.add_subplot(gs[1, 2]),
         "actor/grad_norm", C_GRAD, "Gradient Norm", "Grad Norm")

    # (2,0) Response Length
    _one(fig.add_subplot(gs[2, 0]),
         "response_length/mean", C_TRAIN,
         "Response Length (tokens)", "Tokens")

    # (2,1) Turns
    ax = fig.add_subplot(gs[2, 1])
    et, mt, it, xt = extract_by_epoch(metrics, "num_turns/mean")
    evt, mvt, _, _ = extract_by_epoch(metrics, "val-aux/num_turns/mean")
    if et:
        ax.plot(et, mt, color=C_TURNS, linewidth=2.2, marker="o",
                markersize=5, label="Train Mean")
        ax.fill_between(et, it, xt, alpha=0.15, color=C_TURNS)
    if evt:
        ax.plot(evt, mvt, color=C_VAL, linewidth=2, marker="s",
                markersize=5, label="Val Mean")
    ax.set_title("Number of Turns", fontsize=13, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel("Turns")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # (2,2) Clip Fraction
    _one(fig.add_subplot(gs[2, 2]),
         "actor/pg_clipfrac", C_CLIP, "PPO Clip Fraction", "Clip Fraction")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Epoch curves] saved to: {out_path}")


# =========================================================================
# Trajectory summary (unchanged)
# =========================================================================

def _print_trajectory_summary(rollout_trajs: dict, val_trajs: dict,
                              output_path: str):
    summary_path = output_path.replace(".png", "_trajectories.txt")
    lines = ["=" * 80, "TRAJECTORY SUMMARY", "=" * 80]
    for tag, trajs in [("ROLLOUT", rollout_trajs), ("VALIDATION", val_trajs)]:
        lines.append(f"\n--- {tag} TRAJECTORIES ---")
        for step in sorted(trajs):
            ent = trajs[step]
            sc = [e.get("score", 0) for e in ent]
            avg = sum(sc) / len(sc) if sc else 0
            ok  = sum(1 for s in sc if s >= 0.5)
            top = sum(1 for s in sc if s >= 1.0)
            lines.append(
                f"Step {step:3d}: {len(ent)} samples, "
                f"avg_score={avg:.3f}, "
                f"success(>=0.5)={ok}/{len(ent)}, "
                f"perfect(=1.0)={top}/{len(ent)}")
    if rollout_trajs:
        latest = max(rollout_trajs)
        entries = rollout_trajs[latest]
        scored = sorted(((e.get("score", 0), e) for e in entries),
                        key=lambda x: x[0], reverse=True)
        lines.append(f"\n--- SAMPLE TRAJECTORIES (Step {latest}) ---")
        for label, sub in [("TOP (highest)", scored[:3]),
                           ("BOTTOM (lowest)", scored[-3:])]:
            lines.append(f"\n  [{label}]")
            for score, entry in sub:
                inp = entry.get("input", "")
                prob = (inp[inp.find("problem_statement"):
                            inp.find("problem_statement") + 200]
                        if "problem_statement" in inp else inp[-200:])
                lines.append(f"    Score: {score:.2f}")
                lines.append(f"    Problem: {prob[:120]}...")
                lines.append(f"    Output (first 300 chars): "
                             f"{entry.get('output', '')[:300]}...")
                lines.append(f"    Ground truth (first 150 chars): "
                             f"{entry.get('gts', '')[:150]}...")
                lines.append("")
    text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(text)
    print(f"Trajectory summary saved to: {summary_path}")
    print(text)


# =========================================================================
# Main entry
# =========================================================================

def plot_training_curves(metrics_path: str, traj_dir: str, output_base: str):
    """Generate both step-level and epoch-level curve images."""
    metrics = load_metrics(metrics_path)
    if not metrics:
        print("No metrics found!")
        return

    base, ext = os.path.splitext(output_base)
    if not ext:
        ext = ".png"
    step_path  = f"{base}_step{ext}"
    epoch_path = f"{base}_epoch{ext}"

    # --- Draw both figures ---
    _draw_step_figure(metrics, step_path)
    _draw_epoch_figure(metrics, epoch_path)

    # --- Trajectory summary ---
    rollout_trajs = (load_trajectories(os.path.join(traj_dir, "rollout"))
                     if traj_dir else {})
    val_trajs = (load_trajectories(os.path.join(traj_dir, "validation"))
                 if traj_dir else {})
    if rollout_trajs or val_trajs:
        _print_trajectory_summary(rollout_trajs, val_trajs, output_base)


def main():
    p = argparse.ArgumentParser(
        description="Plot SWE-Agent training curves (step & epoch)")
    p.add_argument("--metrics",
        default="/data1/lmy/workspace/logs/qwen3-4b-swe-train-v6_metrics.jsonl")
    p.add_argument("--trajectories",
        default="/data1/lmy/workspace/trajectories/qwen3-4b-swe-train-v6")
    p.add_argument("--output",
        default="/data1/lmy/workspace/training_curves_v6.png",
        help="Base output path; produces <base>_step.png and <base>_epoch.png")
    args = p.parse_args()
    plot_training_curves(args.metrics, args.trajectories, args.output)


if __name__ == "__main__":
    main()

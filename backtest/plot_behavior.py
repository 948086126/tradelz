import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def safe_plot_equity(df, out_path: str, title: str):
    if "equity" not in df.columns:
        print(f"[skip] {title}: no equity column")
        return
    plt.figure()
    plt.plot(df["equity"].values)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)


def safe_plot_position(df, out_path: str, title: str):
    if "position" not in df.columns:
        print(f"[skip] {title}: no position column")
        return
    plt.figure()
    plt.plot(df["position"].values)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("position")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)


def safe_plot_drawdown(df, out_path: str, title: str):
    if "drawdown" not in df.columns:
        print(f"[skip] {title}: no drawdown column")
        return
    plt.figure()
    plt.plot(df["drawdown"].values)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)


def safe_plot_signal_desired_gap(df, out_path: str, title: str):
    needed = ["signal", "desired_pos", "part_gap"]
    for c in needed:
        if c not in df.columns:
            print(f"[skip] {title}: missing {c}")
            return

    plt.figure()
    plt.plot(df["signal"].values, label="signal")
    plt.plot(df["desired_pos"].values, label="desired_pos")
    plt.plot(df["part_gap"].values, label="part_gap")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="logs")
    args = ap.parse_args()

    log_dir = args.log_dir
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    files = {
        "ppo": os.path.join(log_dir, "behavior_ppo.csv"),
        "always0": os.path.join(log_dir, "behavior_baseline_always0.csv"),
        "buyhold": os.path.join(log_dir, "behavior_baseline_buyhold.csv"),
        "ddcontrol": os.path.join(log_dir, "behavior_baseline_ddcontrol.csv"),
    }

    for name, path in files.items():
        df = load_csv(path)
        if df is None:
            print(f"[missing] {name}: {path}")
            continue

        safe_plot_equity(df, os.path.join(plot_dir, f"{name}_equity.png"), f"{name} equity")
        safe_plot_position(df, os.path.join(plot_dir, f"{name}_position.png"), f"{name} position")
        safe_plot_drawdown(df, os.path.join(plot_dir, f"{name}_drawdown.png"), f"{name} drawdown")

        # 只对 PPO 画 signal/desired/gap（baseline 没这些列）
        if name == "ppo":
            safe_plot_signal_desired_gap(
                df,
                os.path.join(plot_dir, f"{name}_signal_desired_gap.png"),
                f"{name} signal/desired/gap"
            )

    print("\nAll plots saved to:", plot_dir)


if __name__ == "__main__":
    main()

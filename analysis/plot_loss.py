import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(jsonl_path):
    metrics = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics

def load_best_dev(best_path):
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def plot_metrics(metrics, title=None, save_path=None, best_dev=None):
    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    uas = [m["dev_uas"] for m in metrics]
    las = [m["dev_las"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # 左轴：Loss
    ax1.plot(epochs, train_loss, color="tab:blue", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # 右轴：UAS / LAS
    ax2 = ax1.twinx()
    ax2.plot(epochs, uas, color="tab:green", marker="o", label="Dev UAS")
    ax2.plot(epochs, las, color="tab:orange", marker="s", label="Dev LAS")
    ax2.set_ylabel("UAS / LAS", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # 如果有 best_dev.json，就在图上标出最优点
    if best_dev:
        best_epoch = best_dev["epoch"]
        best_uas = best_dev["dev_uas"]
        best_las = best_dev["dev_las"]

        ax2.scatter(best_epoch, best_uas, color="green", s=100, marker="*", label="Best UAS")
        ax2.scatter(best_epoch, best_las, color="orange", s=100, marker="*", label="Best LAS")

        # 在点旁边标数值
        ax2.text(best_epoch + 0.3, best_uas, f"{best_uas:.2f}", color="green", fontsize=9)
        ax2.text(best_epoch + 0.3, best_las, f"{best_las:.2f}", color="orange", fontsize=9)

    # 图例与标题
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.title(title or "Training & Dev Metrics")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    print("Current working directory:", Path.cwd())
    results_dir = Path("./outputs")
    results_dir.mkdir(exist_ok=True)

    for model_dir in Path("../models").glob("*"):
        metrics_path = model_dir / "metrics.jsonl"
        best_path = model_dir / "best_dev.json"
        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            best_dev = load_best_dev(best_path)
            output = results_dir / f"{model_dir.name}_metrics.png"
            plot_metrics(metrics, title=model_dir.name, save_path=output, best_dev=best_dev)

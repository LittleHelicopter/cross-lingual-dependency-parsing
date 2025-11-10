import json
from pathlib import Path
import pandas as pd

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

def summarize_model(model_dir):
    """返回该模型的最后一轮训练结果 + best_dev.json"""
    metrics_path = model_dir / "metrics.jsonl"
    best_path = model_dir / "best_dev.json"

    if not metrics_path.exists():
        return None

    metrics = load_metrics(metrics_path)
    last = metrics[-1] if metrics else {}
    best = load_best_dev(best_path)

    summary = {
        "model": model_dir.name,
        # "last_epoch": last.get("epoch"),
        # "last_train_loss": last.get("train_loss"),
        # "last_uas": last.get("dev_uas"),
        # "last_las": last.get("dev_las"),
        "best_epoch": best.get("epoch") if best else None,
        "best_uas": best.get("dev_uas") if best else None,
        "best_las": best.get("dev_las") if best else None,
    }
    return summary

if __name__ == "__main__":
    base_dir = Path("../models")
    results_dir = Path("./outputs")
    results_dir.mkdir(exist_ok=True)

    summaries = []
    for model_dir in base_dir.glob("*"):
        if model_dir.is_dir():
            s = summarize_model(model_dir)
            if s:
                summaries.append(s)

    if summaries:
        df = pd.DataFrame(summaries)
        df = df.sort_values(by="best_uas", ascending=False)

        print("\n📊 Model Summary Table:")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

        csv_path = results_dir / "model_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved table to {csv_path}")
    else:
        print("⚠️ No metrics found in models directory.")

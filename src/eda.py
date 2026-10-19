import json

import matplotlib.pyplot as plt
import seaborn as sns

from .config import METRICS_DIR, NUMERIC_FEATURES, PLOTS_DIR, TARGET
from .data import clean_data, load_raw_data


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_raw_data())

    # 1. Histograms for numeric features
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_FEATURES):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution: {col}")
    axes[-1].axis("off")
    fig.suptitle("Heart Disease — Numeric Feature Distributions")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_histograms.png", dpi=200)
    plt.close(fig)

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. Class balance
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title("Target Class Balance")
    ax.set_xlabel("Heart disease (0 = absence, 1 = presence)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance.png", dpi=200)
    plt.close(fig)

    # 4. Numeric boxplots
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df[NUMERIC_FEATURES], orient="h", ax=ax)
    ax.set_title("Numeric Feature Boxplots")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "boxplots_numeric.png", dpi=200)
    plt.close(fig)

    summary = {
        "rows": len(df),
        "columns_including_target": int(df.shape[1]),
        "missing_values_by_column": df.isna().sum().to_dict(),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "descriptive_statistics": df.describe().round(3).to_dict(),
    }
    (METRICS_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()

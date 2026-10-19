"""Generate EDA figures into report/figures.

Usage:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES, CLEAN_FILE, FIGURE_DIR, NUMERIC_FEATURES, TARGET,
)


def run(df: pd.DataFrame, out_dir=FIGURE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths = []

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].map({0: "No disease", 1: "Disease"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2",
                hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set(title="Class balance", ylabel="Patients", xlabel="")
    paths.append(_save(fig, out_dir / "class_balance.png"))

    # Numeric histograms split by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax,
                     palette="Set1", element="step")
        ax.set_title(col)
    fig.suptitle("Numeric feature distributions by target")
    paths.append(_save(fig, out_dir / "numeric_histograms.png"))

    # Categorical feature disease rates
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax, palette="Set1")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Categorical features by target")
    paths.append(_save(fig, out_dir / "categorical_counts.png"))

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Feature correlation")
    paths.append(_save(fig, out_dir / "correlation_heatmap.png"))
    return paths


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in run(pd.read_csv(CLEAN_FILE)):
        print("wrote", p)

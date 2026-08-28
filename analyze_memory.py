import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

API_BASE = "http://localhost:8000"


def fetch_memory_data() -> pd.DataFrame:
    """Fetch all stored memory entries from the backend and load them
    into a pandas DataFrame for analysis."""
    response = requests.get(f"{API_BASE}/api/v1/memory")
    response.raise_for_status()
    records = response.json().get("data", [])

    if not records:
        raise ValueError(
            "No memory entries found. Save a few entries via the "
            "dashboard first, then run this script again."
        )

    df = pd.DataFrame(records)

    # Extract the "source" field out of the nested metadata dict
    # (e.g. "manual" for sidebar entries, "chat_log" for chat exchanges)
    df["source"] = df["metadata"].apply(
        lambda m: m.get("source", "unknown") if isinstance(m, dict) else "unknown"
    )

    # Add a column for text length -- this is what we'll analyze with numpy
    df["text_length"] = df["text"].apply(len)

    return df


def analyze(df: pd.DataFrame) -> None:
    """Run basic statistics using numpy/pandas and print a summary."""
    print("=" * 50)
    print("ContextCraft Memory Analysis")
    print("=" * 50)

    print(f"\nTotal entries stored: {len(df)}")

    print("\nEntries by source:")
    print(df["source"].value_counts())

    lengths = df["text_length"].to_numpy()  # convert to a numpy array

    print("\nText length statistics (characters):")
    print(f"  Mean:   {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Min:    {np.min(lengths)}")
    print(f"  Max:    {np.max(lengths)}")
    print(f"  Std dev: {np.std(lengths):.1f}")


def plot(df: pd.DataFrame) -> None:
    """Create a couple of simple charts summarizing the memory store,
    and save them as a PNG file."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: How many entries came from each source
    source_counts = df["source"].value_counts()
    axes[0].bar(source_counts.index, source_counts.values, color="#4C72B0")
    axes[0].set_title("Memory Entries by Source")
    axes[0].set_xlabel("Source")
    axes[0].set_ylabel("Count")

    # Chart 2: Distribution of how long each stored memory is
    axes[1].hist(df["text_length"], bins=10, color="#55A868", edgecolor="black")
    axes[1].set_title("Distribution of Memory Entry Lengths")
    axes[1].set_xlabel("Text length (characters)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    output_path = "memory_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nChart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    df = fetch_memory_data()
    analyze(df)
    plot(df)
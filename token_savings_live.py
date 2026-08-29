import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

API_BASE = "http://localhost:8000"


def fetch_token_log() -> pd.DataFrame:
  response = requests.get(f"{API_BASE}/api/v1/token-stats")
  response.raise_for_status()
  records = response.json().get("data", [])

  if not records:
    raise ValueError(
        "No token log data found yet. Send a few chat messages via the "
        "dashboard first, then run this script again."
    )

  df = pd.DataFrame(records)
  df["cumulative_contextcraft"] = df["contextcraft_tokens"].cumsum()
  df["cumulative_naive"] = df["naive_tokens"].cumsum()
  return df


def summarize(df: pd.DataFrame) -> None:
  print("=" * 50)
  print("ContextCraft — Real Token Usage Comparison")
  print("=" * 50)
  print(f"\nTurns logged: {len(df)}")
  print(f"Total ContextCraft tokens used: {df['contextcraft_tokens'].sum():.0f}")
  print(f"Total naive-equivalent tokens:  {df['naive_tokens'].sum():.0f}")

  total_cc = df["contextcraft_tokens"].sum()
  total_naive = df["naive_tokens"].sum()
  if total_naive > 0:
    savings = (total_naive - total_cc) / total_naive * 100
    print(f"Estimated savings: {savings:.1f}%")


def plot(df: pd.DataFrame) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))

  # Per-turn comparison
  x = np.arange(1, len(df) + 1)
  width = 0.35
  axes[0].bar(x - width / 2, df["contextcraft_tokens"], width,
              label="ContextCraft", color="#55A868")
  axes[0].bar(x + width / 2, df["naive_tokens"], width,
              label="Naive (full history)", color="#C44E52")
  axes[0].set_title("Tokens Used Per Turn")
  axes[0].set_xlabel("Turn")
  axes[0].set_ylabel("Tokens")
  axes[0].legend()

  # Cumulative comparison
  axes[1].plot(x, df["cumulative_contextcraft"], marker="o",
               color="#55A868", label="ContextCraft (cumulative)")
  axes[1].plot(x, df["cumulative_naive"], marker="o",
               color="#C44E52", label="Naive (cumulative)")
  axes[1].set_title("Cumulative Token Usage")
  axes[1].set_xlabel("Turn")
  axes[1].set_ylabel("Cumulative tokens")
  axes[1].legend()

  plt.tight_layout()
  output_path = "token_savings_live.png"
  plt.savefig(output_path, dpi=150)
  print(f"\nChart saved to: {output_path}")
  plt.show()


if __name__ == "__main__":
  df = fetch_token_log()
  summarize(df)
  plot(df)
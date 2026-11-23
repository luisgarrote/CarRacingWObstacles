"""Plot raw episode return and the mean of the last 100 completed episodes."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_rewards(monitor_file, output):
    with Path(monitor_file).open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    if not rows:
        print("No completed episodes yet; train longer to get an episode reward graph.")
        return
    rewards = np.array([float(row["r"]) for row in rows])
    # Before episode 100, use the episodes available so far.
    average = np.array([rewards[max(0, i-99):i+1].mean() for i in range(len(rewards))])
    episodes = np.arange(1, len(rewards)+1)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(episodes, rewards, color="tab:blue", alpha=.45, linewidth=1, label="Raw return")
    ax.plot(episodes, average, color="tab:orange", linewidth=2, label="Mean over last 100 episodes")
    ax.set(xlabel="Completed training episode", ylabel="Total reward (raw return)")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
    with output.with_suffix(".csv").open("w",newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["episode","raw_return","mean_last_100"])
        writer.writerows(zip(episodes,rewards,average))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", "--monitor", dest="monitor", required=True)
    parser.add_argument("--output",default="outputs/rewards.png")
    args = parser.parse_args(); plot_rewards(args.monitor,args.output)

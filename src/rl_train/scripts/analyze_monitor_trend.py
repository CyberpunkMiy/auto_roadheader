import csv
from pathlib import Path


LOG_ROOT = Path(r"D:\vscode_project\chuanlian _No\RL_juejin\src\rl_train\logs")


def read_rows():
    rows = []
    for path in sorted(LOG_ROOT.glob("[0-3].monitor.csv")):
        with path.open("r", encoding="utf-8") as handle:
            data_lines = [line for line in handle if not line.startswith("#")]
        for row in csv.DictReader(data_lines):
            row["_file"] = path.name
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("t") or 0.0))
    return rows


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def summarize(rows, label):
    if not rows:
        return
    removed = [as_float(row, "removed_fraction") for row in rows]
    remaining = [as_float(row, "remaining_voxels") for row in rows]
    rewards = [as_float(row, "r") for row in rows]
    lengths = [as_float(row, "l") for row in rows]
    removed = [v for v in removed if v is not None]
    remaining = [v for v in remaining if v is not None]
    rewards = [v for v in rewards if v is not None]
    lengths = [v for v in lengths if v is not None]

    best_idx = min(range(len(rows)), key=lambda i: as_float(rows[i], "remaining_voxels") or float("inf"))
    best = rows[best_idx]
    print(label)
    print("-" * 60)
    print(f"episodes: {len(rows)}")
    print(f"reward_mean: {sum(rewards) / len(rewards):.3f}" if rewards else "reward_mean: -")
    print(f"len_mean: {sum(lengths) / len(lengths):.1f}" if lengths else "len_mean: -")
    print(f"removed_mean: {sum(removed) / len(removed):.5f}" if removed else "removed_mean: -")
    print(f"removed_best: {max(removed):.5f}" if removed else "removed_best: -")
    print(f"remaining_mean: {sum(remaining) / len(remaining):.2f}" if remaining else "remaining_mean: -")
    print(f"remaining_best: {min(remaining):.0f}" if remaining else "remaining_best: -")
    print(
        "best_episode: "
        f"remaining={best.get('remaining_voxels')} "
        f"removed={best.get('removed_fraction')} "
        f"reward={best.get('r')} "
        f"t={best.get('t')} "
        f"file={best.get('_file')}"
    )
    print()


def main():
    rows = read_rows()
    print(f"total_episodes: {len(rows)}")
    print()
    for n in (20, 50, 100, 200, 500):
        summarize(rows[-n:], f"last_{min(n, len(rows))}_episodes")
    summarize(rows, "all_episodes")


if __name__ == "__main__":
    main()

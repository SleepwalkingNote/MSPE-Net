"""Recompute every configuration in the released adjacent-track benchmark."""
import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from evaluate import ROOT, load_model, predict, scores


def check_scores(result, row, prefix):
    for key, value in result.items():
        expected = float(row[prefix + key])
        if not np.isclose(value, expected, rtol=0, atol=1e-12):
            raise ValueError(f"{row['region']}, target {row['target']}, {row['k']} tracks: "
                             f"{prefix}{key} = {value}, expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_model(args.device)
    regions = defaultdict(lambda: defaultdict(list))
    with (ROOT / "benchmark/track_scores.csv").open() as handle:
        for row in csv.DictReader(handle):
            tracks = tuple(sorted(map(int, row["input_tracks"].split())))
            regions[row["region"]][tracks].append(row)

    verified = 0
    for region, configurations in regions.items():
        with np.load(ROOT / "data" / (region + ".npz")) as archive:
            points, bounds = archive["points"], archive["bounds"]
        for tracks, targets in configurations.items():
            predictions = predict(model, points, bounds, tracks)
            input_scores = scores(points[:, 3], predictions)
            for row in targets:
                target = points[:, 4] == int(row["target"])
                check_scores(input_scores, row, "input_")
                check_scores(scores(points[target, 3], predictions[target]), row, "target_")
                verified += 1
        print(f"{region} verified ({verified} configurations so far)", flush=True)


if __name__ == "__main__":
    main()

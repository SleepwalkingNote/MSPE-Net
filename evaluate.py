"""Run the released MSPE-Net checkpoint on a labeled test region."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from network.models_Photon_ablation_SAG_only.model import PointTransformerSeg


ROOT = Path(__file__).resolve().parent


def load_model(device):
    settings = SimpleNamespace(nblocks=3, nneighbor=16, transformer_dim=128, use_fps=True)
    config = SimpleNamespace(num_point=4096, num_class=2, input_dim=3, model=settings)
    model = PointTransformerSeg(config).to(device)
    weights = torch.load(ROOT / "weights/accepted_fp_reflux.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.eval()


@torch.inference_mode()
def predict(model, points, bounds, tracks=None, reference_gate=True):
    device = next(model.parameters()).device
    selected = np.ones(len(points), dtype=bool) if tracks is None else np.isin(points[:, 4], tracks)
    predictions = np.where(selected, 0, -1).astype(np.int8)
    for chunk, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        indices = np.flatnonzero(selected[start:end]) + start
        if not len(indices):
            continue
        torch.manual_seed(44 + chunk)
        if reference_gate:
            heights = points[indices, 2]
            signal = heights[points[indices, 3] == 1]
            if signal.size:
                indices = indices[(heights >= signal.min() - 10) & (heights <= signal.max() + 10)]
        xyz = points[indices, :3]
        span = np.ptp(xyz, axis=0)
        span[span == 0] = 1
        normalized = (2 * ((xyz - xyz.min(axis=0)) / span) - 1).astype(np.float32)
        batch = torch.from_numpy(normalized).to(device)[None]
        predictions[indices] = model(batch)[0].argmax(-1).cpu().numpy()
    return predictions


def scores(labels, predictions):
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return dict(TP=tp, FP=fp, FN=fn, F1=2 * tp / max(1, 2 * tp + fp + fn),
                Precision=tp / max(1, tp + fp), Recall=tp / max(1, tp + fn))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", choices=sorted(p.stem for p in (ROOT / "data").glob("*.npz")))
    parser.add_argument("--no-reference-gate", action="store_true", help="Predict without reference-assisted height filtering")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with np.load(ROOT / "data" / (args.region + ".npz")) as archive:
        points, bounds = archive["points"], archive["bounds"]
    predictions = predict(load_model(args.device), points, bounds, reference_gate=not args.no_reference_gate)
    print(scores(points[:, 3], predictions))


if __name__ == "__main__":
    main()

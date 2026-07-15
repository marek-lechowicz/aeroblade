"""Evaluate AEROBLADE on the FakeFlickr dataset using the Flickr30k test split.

AEROBLADE is training-free: it reconstructs each image with several latent-diffusion
autoencoders and uses the best (``max``) reconstruction distance as the detection
score. Images that reconstruct well (higher score) are more likely LDM-generated.
There is therefore no classifier and no natural 0.5 threshold -- AP is threshold-free,
but ACC/R_ACC/F_ACC need a threshold, which we calibrate *once* globally.

For every generator under ``<dataset_root>/generated/<gen>/img`` this script:

  1. Filters images to the IDs listed in the Flickr30k test split.
  2. Stages real + fake images resized to 512x512 and re-encoded to JPEG q90 (removing
     the resolution and format confounds). Real images come from
     ``<dataset_root>/real`` (or ``real_rescaled`` for the ``flux_fill_real_rescaled``
     generator, which was conditioned on the rescaled reals). Reals are staged and
     scored once per source and shared across generators.
  3. Computes AEROBLADE reconstruction distances via ``aeroblade.compute_distances``
     (SD1.5, SD2-base, Kandinsky-2.1 autoencoders; ``max`` over AEs; ``lpips_vgg_2``).
  4. Calibrates a single global threshold (max balanced accuracy on the pooled reals +
     all fakes) and reports ACC / AP / R_ACC / F_ACC per generator.

Results are written as one CSV row per generator.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeroblade.high_level_funcs import compute_distances  # noqa: E402

DEFAULT_GENERATORS = [
    "sd_1_5",
    "sd_3_5_large",
    "sdxl_turbo",
    "z_image_turbo",
    "flux_1_dev",
    "flux_fill_flux_1_dev",
    "flux_fill_sd_3_5_large",
    "flux_fill_real_rescaled",
]

# Generators that were conditioned on the rescaled-real source images.
# For these, the matching "real" is the rescaled PNG, not the original JPG.
RESCALED_REAL_GENS = {"flux_fill_real_rescaled"}

# AEROBLADE paper default autoencoders. The paper's SD2 repo
# (stabilityai/stable-diffusion-2-base) was removed from HuggingFace; we use the
# open sd2-community mirror, which is the same pipeline (incl. fp16 variant weights).
DEFAULT_REPO_IDS = [
    "CompVis/stable-diffusion-v1-1",
    "sd2-community/stable-diffusion-2-base",
    "kandinsky-community/kandinsky-2-1",
]


def read_test_ids(split_file: Path) -> list[str]:
    with split_file.open("r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Test split is empty: {split_file}")
    return ids


def find_image(dirpath: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPEG"):
        cand = dirpath / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def preprocess_and_stage(
    src_dir: Path,
    dest_dir: Path,
    test_ids: list[str],
    image_size: int,
    jpeg_quality: int,
) -> int:
    """Resize test-split images to ``image_size`` (short-side resize + center crop) and
    re-encode them to JPEG at ``jpeg_quality`` under ``dest_dir/<id>.jpg``.

    Making every real and fake image identical in size and format removes the
    resolution and JPEG-vs-WebP/PNG confounds before reconstruction. Returns the number
    of images staged.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    resize_crop = transforms.Compose(
        [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    )

    n = 0
    for img_id in tqdm(test_ids, desc=f"  stage {src_dir.name}", leave=False):
        src = find_image(src_dir, img_id)
        if src is None:
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception as exc:  # corrupt / unreadable
            print(f"  skip {src}: {exc}")
            continue
        resize_crop(img).save(dest_dir / f"{img_id}.jpg", quality=jpeg_quality)
        n += 1
    if n == 0:
        raise RuntimeError(f"No test-split images matched under {src_dir}")
    return n


def scores_by_dir(
    distances,
    distance_metric: str,
) -> dict[str, dict[str, float]]:
    """From the compute_distances dataframe, build ``dir -> {file: score}`` using the
    ``max``-over-AEs reconstruction distance (higher score => more likely fake)."""
    sel = distances[
        (distances.repo_id == "max") & (distances.distance_metric == distance_metric)
    ]
    out: dict[str, dict[str, float]] = {}
    for dir_str, group in sel.groupby("dir", sort=False):
        out[dir_str] = dict(zip(group.file, group.distance.astype(float)))
    return out


def calibrate_global_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Threshold (predict fake when score > threshold) maximizing balanced accuracy."""
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    # Candidate thresholds: midpoints between consecutive unique scores, plus outer edges.
    uniq = np.unique(sorted_scores)
    if len(uniq) == 1:
        return float(uniq[0])
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate([[uniq[0] - 1e-6], mids, [uniq[-1] + 1e-6]])

    is_real = y_true == 0
    is_fake = y_true == 1
    n_real = max(int(is_real.sum()), 1)
    n_fake = max(int(is_fake.sum()), 1)

    best_thr, best_bacc = float(candidates[0]), -1.0
    for thr in candidates:
        pred_fake = y_score > thr
        r_acc = (~pred_fake & is_real).sum() / n_real
        f_acc = (pred_fake & is_fake).sum() / n_fake
        bacc = 0.5 * (r_acc + f_acc)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    return best_thr


def evaluate_generator(
    real_scores: dict[str, float],
    fake_scores: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    y_true = np.array([0] * len(real_scores) + [1] * len(fake_scores))
    y_score = np.array(list(real_scores.values()) + list(fake_scores.values()))
    y_pred = y_score > threshold
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "AP": float(average_precision_score(y_true, y_score)),
        "R_ACC": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])),
        "F_ACC": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])),
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
        "threshold": float(threshold),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/fake-flickr"),
                   help="Root of the fake-flickr dataset.")
    p.add_argument("--test-split", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"),
                   help="File with one Flickr30k image ID per line (the test split).")
    p.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS,
                   help=f"Generator subdirs to evaluate (default: {DEFAULT_GENERATORS}).")
    p.add_argument("--work-dir", type=Path,
                   default=REPO_ROOT / "data" / "fake_flickr",
                   help="Where to stage resized images and cache reconstructions.")
    p.add_argument("--results-csv", type=Path,
                   default=REPO_ROOT / "data" / "results" / "fake_flickr_aeroblade.csv",
                   help="Output CSV path.")
    p.add_argument("--repo-ids", nargs="+", default=DEFAULT_REPO_IDS,
                   help="HuggingFace autoencoders used for reconstruction.")
    p.add_argument("--distance-metric", default="lpips_vgg_2",
                   help="Reconstruction distance metric (AEROBLADE paper best).")
    p.add_argument("--image-size", type=int, default=512,
                   help="Size images are resized/cropped to before reconstruction.")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality for re-encoding staged images (format equalization).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: only run on --debug-samples images per set.")
    p.add_argument("--debug-samples", type=int, default=8,
                   help="Number of test IDs to keep in --debug mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.test_split.is_file():
        raise FileNotFoundError(f"--test-split not found: {args.test_split}")

    test_ids = read_test_ids(args.test_split)
    print(f"Loaded {len(test_ids)} test IDs from {args.test_split}")
    if args.debug:
        test_ids = test_ids[: args.debug_samples]
        print(f"[DEBUG] truncated to {len(test_ids)} IDs")

    stage_root = args.work_dir / "stage"
    recon_root = args.work_dir / "reconstructions"
    stage_root.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    # Stage reals once per source (shared across generators).
    real_sources = {
        ("real_rescaled" if g in RESCALED_REAL_GENS else "real")
        for g in args.generators
    }
    real_dirs: dict[str, Path] = {}
    for rn in sorted(real_sources):
        print(f"\n=== staging shared reals: {rn} ===")
        dest = stage_root / f"_shared_{rn}"
        n = preprocess_and_stage(
            args.dataset_root / rn, dest, test_ids, args.image_size, args.jpeg_quality
        )
        print(f"  staged {n} real images ({rn})")
        real_dirs[rn] = dest

    # Stage fakes per generator.
    fake_dirs: dict[str, Path] = {}
    for gen in args.generators:
        print(f"\n=== staging fakes: {gen} ===")
        dest = stage_root / gen
        n = preprocess_and_stage(
            args.dataset_root / "generated" / gen / "img",
            dest, test_ids, args.image_size, args.jpeg_quality,
        )
        print(f"  staged {n} fake images")
        fake_dirs[gen] = dest

    # Compute AEROBLADE reconstruction distances for every staged directory in one call
    # (reconstructions are cached under recon_root).
    all_dirs = list(real_dirs.values()) + list(fake_dirs.values())
    print(f"\nComputing reconstruction distances for {len(all_dirs)} directories "
          f"with AEs {args.repo_ids} ...")
    distances = compute_distances(
        dirs=all_dirs,
        transforms=["clean"],
        repo_ids=args.repo_ids,
        distance_metrics=[args.distance_metric],
        amount=None,
        reconstruction_root=recon_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dir_scores = scores_by_dir(distances, args.distance_metric)

    # Calibrate a single global threshold on pooled reals + all fakes.
    pooled_true: list[int] = []
    pooled_score: list[float] = []
    for rn, dest in real_dirs.items():
        pooled_true += [0] * len(dir_scores[str(dest)])
        pooled_score += list(dir_scores[str(dest)].values())
    for gen, dest in fake_dirs.items():
        pooled_true += [1] * len(dir_scores[str(dest)])
        pooled_score += list(dir_scores[str(dest)].values())
    threshold = calibrate_global_threshold(
        np.array(pooled_true), np.array(pooled_score)
    )
    print(f"\nGlobal threshold (max balanced accuracy): {threshold:.6f}")

    # Per-generator metrics at the global threshold.
    rows: list[dict] = []
    for gen in args.generators:
        rn = "real_rescaled" if gen in RESCALED_REAL_GENS else "real"
        real_scores = dir_scores[str(real_dirs[rn])]
        fake_scores = dir_scores[str(fake_dirs[gen])]
        metrics = evaluate_generator(real_scores, fake_scores, threshold)
        print(f"  {gen}: " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ))
        rows.append({"generator": gen, **metrics})

    fieldnames = ["generator", "ACC", "AP", "R_ACC", "F_ACC",
                  "N_real", "N_fake", "threshold"]
    with args.results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {args.results_csv}")


if __name__ == "__main__":
    main()

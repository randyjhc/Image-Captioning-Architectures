"""Evaluate ViT or CNN-LSTM image captioning models using BLEU, ROUGE-L, and CIDEr.

Usage:
    python evaluate.py vit --checkpoint checkpoints/vit/best.pt
    python evaluate.py cnn --checkpoint checkpoints/cnn/best.pt
    python evaluate.py vit --checkpoint checkpoints/vit/best.pt --data-root data/datasets/flickr8k --batch-size 16

The script uses the same train/val/test split (seed=42) as the trainer, so the
test set is guaranteed not to overlap with training data.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.flickr_dataset import FlickrDataset
from data.image.transforms import get_val_transforms
from data.text.vocabulary import clean_caption, tokenize
from utils import logger_setup


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _get_ngrams(tokens: list[str], max_n: int) -> dict[tuple, int]:
    ngrams: dict[tuple, int] = collections.defaultdict(int)
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i : i + n])] += 1
    return ngrams


def compute_bleu(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> dict[str, float]:
    """Corpus-level BLEU-1 through BLEU-max_n with smoothing.

    Args:
        references: List of reference lists per image; each inner list is a
                    tokenised reference string.
        hypotheses: List of tokenised hypothesis strings, one per image.
        max_n: Maximum n-gram order.

    Returns:
        Dict mapping "BLEU-1" … "BLEU-N" to float scores in [0, 1].
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    sf = SmoothingFunction().method1
    scores: dict[str, float] = {}
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n for _ in range(n))
        scores[f"BLEU-{n}"] = corpus_bleu(
            references, hypotheses, weights=weights, smoothing_function=sf
        )
    return scores


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    # Space-optimised DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
) -> float:
    """Corpus-level ROUGE-L F1 (best-matching reference per image).

    Args:
        references: Same format as in compute_bleu.
        hypotheses: Same format as in compute_bleu.

    Returns:
        Mean ROUGE-L F1 across all images.
    """
    f1_scores: list[float] = []
    for refs, hyp in zip(references, hypotheses):
        best_f1 = 0.0
        for ref in refs:
            if not hyp or not ref:
                continue
            lcs = _lcs_length(ref, hyp)
            precision = lcs / len(hyp)
            recall = lcs / len(ref)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        f1_scores.append(best_f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_cider(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> float:
    """Corpus-level CIDEr-D score.

    Implements the TF-IDF weighted n-gram cosine similarity from:
      Vedantam et al., "CIDEr: Consensus-based Image Description Evaluation", CVPR 2015.

    Args:
        references: Same format as in compute_bleu.
        hypotheses: Same format as in compute_bleu.
        max_n: Maximum n-gram order (default 4, matching original paper).

    Returns:
        CIDEr-D score (unbounded above; human captions ≈ 1.0 on COCO).
    """
    # Build document-frequency table over all reference captions
    df: dict[tuple, int] = collections.defaultdict(int)
    num_refs = sum(len(refs) for refs in references)

    for refs in references:
        for ref in refs:
            seen: set[tuple] = set()
            for ng in _get_ngrams(ref, max_n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)

    def tfidf(ngrams: dict[tuple, int]) -> dict[tuple, float]:
        weights: dict[tuple, float] = {}
        for ng, count in ngrams.items():
            idf = math.log((num_refs + 1.0) / (df.get(ng, 0) + 1.0))
            weights[ng] = count * idf
        return weights

    def vec_norm(v: dict[tuple, float]) -> float:
        return math.sqrt(sum(x * x for x in v.values()))

    cider_scores: list[float] = []
    for refs, hyp in zip(references, hypotheses):
        hyp_ngrams = _get_ngrams(hyp, max_n)
        hyp_w = tfidf(hyp_ngrams)
        hyp_norm = vec_norm(hyp_w)

        score_per_order: list[float] = []
        for n in range(1, max_n + 1):
            # Filter to order-n n-grams only
            hyp_n = {k: v for k, v in hyp_w.items() if len(k) == n}
            hyp_n_norm = math.sqrt(sum(x * x for x in hyp_n.values()))

            ref_sum = 0.0
            for ref in refs:
                ref_n = {k: v for k, v in tfidf(_get_ngrams(ref, n)).items() if len(k) == n}
                ref_n_norm = math.sqrt(sum(x * x for x in ref_n.values()))
                if hyp_n_norm == 0 or ref_n_norm == 0:
                    continue
                dot = sum(hyp_n.get(ng, 0.0) * w for ng, w in ref_n.items())
                ref_sum += dot / (hyp_n_norm * ref_n_norm)

            score_per_order.append((10.0 / len(refs)) * ref_sum)

        cider_scores.append(sum(score_per_order) / max_n if score_per_order else 0.0)

    return sum(cider_scores) / len(cider_scores) if cider_scores else 0.0


# ---------------------------------------------------------------------------
# Image-only dataset for batched inference on unique test images
# ---------------------------------------------------------------------------

class _UniqueImageDataset(Dataset):
    """Yields (image_tensor, image_filename) for a list of unique images."""

    def __init__(self, image_dir: Path, filenames: list[str], transform) -> None:
        self.image_dir = image_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        from data.image.image_utils import load_image

        fname = self.filenames[idx]
        img = load_image(self.image_dir / fname, convert_mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fname


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_vit_generator(checkpoint_path: Path, data_root: Path, device: torch.device):
    from model_vit.generator import GeneratorViT

    gen = GeneratorViT.from_checkpoint(checkpoint_path, data_root)
    gen.model.to(device)
    return gen


def _load_cnn_generator(checkpoint_path: Path, data_root: Path, device: torch.device):
    from model_cnn_lstm.generator import GeneratorCNN

    gen = GeneratorCNN.from_checkpoint(checkpoint_path, data_root)
    gen.model.to(device)
    return gen


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model_type: str,
    checkpoint_path: Path,
    data_root: Path,
    batch_size: int = 32,
    max_len: int = 30,
    num_workers: int = 0,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Run full evaluation on the test split and return metric scores.

    Args:
        model_type: "vit" or "cnn".
        checkpoint_path: Path to the model checkpoint.
        data_root: Root of the Flickr8k dataset (contains captions.txt and Images/).
        batch_size: Images per inference batch.
        max_len: Maximum generation length in tokens.
        num_workers: DataLoader workers.
        device: Torch device. Auto-detected if None.
        seed: Dataset split seed — must match the seed used during training (default 42).

    Returns:
        Dict with keys: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, CIDEr.
    """
    logger = logging.getLogger("image_caption")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load generator (rebuilds vocab from captions.txt)
    logger.info(f"Loading {model_type.upper()} model from {checkpoint_path}")
    if model_type == "vit":
        gen = _load_vit_generator(checkpoint_path, data_root, device)
    elif model_type == "cnn":
        gen = _load_cnn_generator(checkpoint_path, data_root, device)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose 'vit' or 'cnn'.")

    # Build test split (same seed as trainer)
    transform = get_val_transforms(image_size=224)
    _, _, test_ds = FlickrDataset.create_splits(
        root_dir=data_root,
        seed=seed,
        test_transform=transform,
    )
    logger.info(f"Test split: {len(test_ds)} caption samples")

    # Group references by image filename (5 captions per image in Flickr8k)
    image_to_refs: dict[str, list[list[str]]] = collections.defaultdict(list)
    for fname, caption in test_ds.samples:
        tokens = tokenize(clean_caption(caption))
        image_to_refs[fname].append(tokens)

    unique_filenames = list(image_to_refs.keys())
    logger.info(f"Unique test images: {len(unique_filenames)}")

    # Build inference dataloader over unique images
    infer_ds = _UniqueImageDataset(data_root / "Images", unique_filenames, transform)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Generate captions
    hypotheses_map: dict[str, list[str]] = {}
    gen.model.eval()

    for images, fnames in tqdm(infer_loader, desc="Generating captions"):
        images = images.to(device)
        captions = gen.generate_caption(images, max_len=max_len, skip_special=True)
        for fname, cap in zip(fnames, captions):
            hypotheses_map[fname] = tokenize(clean_caption(cap))

    # Align references and hypotheses in the same order
    all_references: list[list[list[str]]] = []
    all_hypotheses: list[list[str]] = []
    for fname in unique_filenames:
        all_references.append(image_to_refs[fname])
        all_hypotheses.append(hypotheses_map[fname])

    # Compute metrics
    logger.info("Computing metrics...")
    bleu_scores = compute_bleu(all_references, all_hypotheses)
    rouge_l = compute_rouge_l(all_references, all_hypotheses)
    cider = compute_cider(all_references, all_hypotheses)

    results = {**bleu_scores, "ROUGE-L": rouge_l, "CIDEr": cider}
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_results(model_type: str, results: dict[str, float]) -> None:
    header = f"  {model_type.upper()} Evaluation Results  "
    bar = "=" * (len(header) + 4)
    print(f"\n{bar}")
    print(f"| {header} |")
    print(bar)
    for metric, score in results.items():
        print(f"  {metric:<10} {score:.4f}")
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate image captioning models with BLEU, ROUGE-L, and CIDEr."
    )
    parser.add_argument(
        "model",
        choices=["vit", "cnn"],
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets/flickr8k"),
        help="Root directory of the Flickr8k dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=30,
        help="Maximum caption generation length.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset split seed (must match the seed used during training).",
    )
    args = parser.parse_args()

    logger_setup()
    logger = logging.getLogger("image_caption")

    device = torch.device(args.device) if args.device else None

    results = evaluate(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    _print_results(args.model, results)

    logger.info("Done.")


if __name__ == "__main__":
    main()

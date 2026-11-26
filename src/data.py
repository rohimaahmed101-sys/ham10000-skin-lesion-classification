"""Data loading, grouping, splitting, leakage checks and the tf.data input pipeline.

Owner: Data Engineer. Checklist items: E1, E2, E3, E4, E5, E7, E8.

Design, from the Phase 1 dataset check (notebooks/00_phase1_dataset_check.ipynb):
- The Kaggle copy stores every image twice; each image_id is mapped to ONE file.
- 1,956 lesions have several photos, so the split is made by lesion, never by image.
- 2 pairs of different image_ids are byte-identical; their lesions are merged into one
  group so both copies always land in the same split.
- The split is stratified by class at group level (every group has a single label) and
  saved to data/splits/split_v1.csv, so every member trains on exactly the same split.
- Class weights come from the training split only (no information from val/test).
- Augmentation is applied to the training split only.
- The same `read_and_resize` and `get_preprocess` functions are used for training and for
  inference (src/predict.py), so preprocessing is never duplicated.

Command line (run from the repository root):
    python -m src.data                 # build the split if missing, run every leakage check
    python -m src.data --force-split   # rebuild the split (changes every later result!)
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import REPO_ROOT, load_config, set_seed

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ["image_id", "lesion_id", "group_id", "dx", "label", "split"]


# --------------------------------------------------------------------------------------
# 1. Locating and loading the raw data
# --------------------------------------------------------------------------------------
def find_data_dir(cfg: dict) -> Path:
    """Return the folder that holds HAM10000.

    Uses the environment variable HAM10000_DIR when it is set (for local copies);
    otherwise downloads with kagglehub, which reuses its cache on later calls.
    """
    env = os.environ.get("HAM10000_DIR")
    if env:
        return Path(env)
    import kagglehub  # imported here so the rest of the module works without it

    return Path(kagglehub.dataset_download(cfg["data"]["kaggle_dataset"]))


def load_metadata(data_dir: str | Path, classes: list[str]) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach exactly one image path and an integer label.

    Images are taken from the HAM10000_images_part_* folders first; the lower-case
    duplicate folders are only a fallback. Raises if any metadata row has no image or
    if a diagnosis code is not in `classes`.
    """
    data_dir = Path(data_dir)
    meta_path = next(p for p in data_dir.rglob("*.csv") if p.name.lower() == "ham10000_metadata.csv")
    meta = pd.read_csv(meta_path)

    paths: dict[str, str] = {}
    preferred = sorted(data_dir.glob("HAM10000_images_part_*/*.jpg"))
    preferred_set = set(preferred)
    fallback = sorted(p for p in data_dir.rglob("*.jpg") if p not in preferred_set)
    for p in preferred + fallback:
        paths.setdefault(p.stem, str(p))

    missing = sorted(set(meta["image_id"]) - set(paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} metadata rows have no image file, e.g. {missing[:3]}")
    unknown = sorted(set(meta["dx"]) - set(classes))
    if unknown:
        raise ValueError(f"Diagnosis codes not in the config class list: {unknown}")

    meta = meta.copy()
    meta["path"] = meta["image_id"].map(paths)
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(classes)}).astype(int)
    return meta


# --------------------------------------------------------------------------------------
# 2. Grouping: lesions + byte-identical images
# --------------------------------------------------------------------------------------
def md5_of(path: str | Path) -> str:
    """Return the MD5 fingerprint of a file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_identical_images(meta: pd.DataFrame) -> list[list[str]]:
    """Return groups of different image_ids whose files are byte-identical."""
    by_hash: dict[str, list[str]] = {}
    for image_id, path in zip(meta["image_id"], meta["path"]):
        by_hash.setdefault(md5_of(path), []).append(image_id)
    return sorted(sorted(ids) for ids in by_hash.values() if len(ids) > 1)


def assign_groups(meta: pd.DataFrame, identical_groups: list[list[str]]) -> pd.Series:
    """Return a group_id per image: its lesion_id, with lesions merged when they share an identical image.

    Uses union-find so chains of merges are handled. The group_id is the alphabetically
    first lesion_id of the merged set.
    """
    parent = {lesion: lesion for lesion in meta["lesion_id"].unique()}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lesion_of = dict(zip(meta["image_id"], meta["lesion_id"]))
    for ids in identical_groups:
        roots = sorted({root(lesion_of[i]) for i in ids})
        for r in roots[1:]:
            parent[r] = roots[0]
    return meta["lesion_id"].map(root).rename("group_id")


# --------------------------------------------------------------------------------------
# 3. Splitting
# --------------------------------------------------------------------------------------
def make_split(meta: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """Split groups into train / val / test, stratified by class.

    Every group must carry one label (checked). Groups, not images, are split, so the
    image-level ratios can differ slightly from the requested ones; they are reported.
    """
    if "group_id" not in meta:
        raise KeyError("meta needs a group_id column; call assign_groups first")
    labels_per_group = meta.groupby("group_id")["label"].nunique()
    mixed = labels_per_group[labels_per_group > 1]
    if len(mixed):
        raise ValueError(f"{len(mixed)} groups carry more than one label, e.g. {list(mixed.index[:3])}")

    groups = meta.groupby("group_id")["label"].first()
    total = ratios["train"] + ratios["val"] + ratios["test"]
    test_frac = ratios["test"] / total
    val_frac_of_rest = ratios["val"] / (ratios["train"] + ratios["val"])

    g_rest, g_test = train_test_split(groups.index.values, test_size=test_frac,
                                      stratify=groups.values, random_state=seed)
    g_train, g_val = train_test_split(g_rest, test_size=val_frac_of_rest,
                                      stratify=groups.loc[g_rest].values, random_state=seed)

    split_of = {**{g: "train" for g in g_train}, **{g: "val" for g in g_val}, **{g: "test" for g in g_test}}
    out = meta.assign(split=meta["group_id"].map(split_of))
    return out[SPLIT_COLUMNS].sort_values(["split", "image_id"]).reset_index(drop=True)


def check_no_leakage(split_df: pd.DataFrame, classes: list[str],
                     identical_groups: list[list[str]] | None = None) -> dict:
    """Run every leakage check and raise AssertionError on the first failure.

    Checks: each image appears once; only train/val/test are used; no lesion and no group
    spans two splits; byte-identical images share a split; every class is in every split.
    Returns a small report of what was checked.
    """
    assert split_df["image_id"].is_unique, "an image_id appears more than once"
    assert set(split_df["split"]) == set(SPLITS), f"unexpected split names: {set(split_df['split'])}"

    lesion_splits = split_df.groupby("lesion_id")["split"].nunique()
    assert (lesion_splits == 1).all(), f"{int((lesion_splits > 1).sum())} lesions appear in two splits"

    group_splits = split_df.groupby("group_id")["split"].nunique()
    assert (group_splits == 1).all(), f"{int((group_splits > 1).sum())} groups appear in two splits"

    split_of = dict(zip(split_df["image_id"], split_df["split"]))
    for ids in identical_groups or []:
        where = {split_of[i] for i in ids}
        assert len(where) == 1, f"identical images {ids} are in different splits: {where}"

    for name in SPLITS:
        present = set(split_df.loc[split_df["split"] == name, "dx"])
        absent = sorted(set(classes) - present)
        assert not absent, f"classes {absent} are missing from the {name} split"

    return {
        "images": int(len(split_df)),
        "lesions": int(split_df["lesion_id"].nunique()),
        "groups": int(split_df["group_id"].nunique()),
        "identical_groups_checked": len(identical_groups or []),
        "passed": True,
    }


def split_summary(split_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Images per class and split (integers), with a TOTAL row."""
    counts = pd.crosstab(split_df["dx"], split_df["split"]).reindex(index=classes, columns=list(SPLITS), fill_value=0)
    counts.loc["TOTAL"] = counts.sum()
    return counts.astype(int)


def split_shares(split_df: pd.DataFrame) -> dict[str, float]:
    """Share of all images in each split, in percent (groups are split, so this is close to, not exactly, 70/15/15)."""
    shares = split_df["split"].value_counts(normalize=True) * 100
    return {name: round(float(shares.get(name, 0.0)), 2) for name in SPLITS}


def class_proportions(split_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Percentage of each class inside each split; similar columns mean the stratification worked."""
    pct = pd.crosstab(split_df["dx"], split_df["split"], normalize="columns") * 100
    return pct.reindex(index=classes, columns=list(SPLITS), fill_value=0).round(2)


def class_weights(split_df: pd.DataFrame, n_classes: int) -> dict[int, float]:
    """Balanced class weights computed from the TRAINING split only: n_samples / (n_classes * count)."""
    y = split_df.loc[split_df["split"] == "train", "label"].to_numpy()
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return {i: float(w) for i, w in enumerate(weights)}


def split_path(cfg: dict) -> Path:
    """Absolute path of the committed split file."""
    return REPO_ROOT / cfg["data"]["split_file"]


def save_split(split_df: pd.DataFrame, cfg: dict) -> Path:
    """Write the split CSV (no file paths inside, so it works on every machine)."""
    path = split_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df[SPLIT_COLUMNS].to_csv(path, index=False)
    return path


def load_split(cfg: dict, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Read the committed split; when `meta` is given, attach each image's local path."""
    split_df = pd.read_csv(split_path(cfg))
    if meta is not None:
        split_df = split_df.merge(meta[["image_id", "path"]], on="image_id", how="left", validate="one_to_one")
        if split_df["path"].isna().any():
            raise FileNotFoundError("some images in the split file were not found in the dataset folder")
    return split_df


# --------------------------------------------------------------------------------------
# 4. tf.data pipeline (TensorFlow is imported only inside these functions)
# --------------------------------------------------------------------------------------
def read_and_resize(path, image_size):
    """Read one JPEG and resize it to `image_size` (direct resize); returns uint8 HxWx3.

    Used by the training pipeline AND by src/predict.py, so inference sees the same input.
    """
    import tensorflow as tf

    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, image_size, method="bilinear")
    return tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)


def get_preprocess(name: str):
    """Pixel scaling for a model family; input is float32 in [0, 255].

    'cnn'      -> x / 255 (custom CNN)
    'resnet50' -> keras.applications.resnet50.preprocess_input
    'vgg16'    -> keras.applications.vgg16.preprocess_input
    Pretrained weights expect the scaling used when they were trained on ImageNet.
    """
    if name == "cnn":
        return lambda x: x / 255.0
    from tensorflow.keras.applications import resnet50, vgg16

    table = {"resnet50": resnet50.preprocess_input, "vgg16": vgg16.preprocess_input}
    if name not in table:
        raise ValueError(f"unknown preprocessing '{name}'; use one of cnn, resnet50, vgg16")
    return table[name]


def build_augmenter(aug_cfg: dict, seed: int):
    """Keras augmentation block for the TRAINING split only (spec 3.2: at least two augmentations).

    Flip and rotation: a lesion has no natural orientation. Zoom: lesions appear at
    different magnifications. Brightness and contrast: lighting differs between images.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.RandomFlip(aug_cfg["flip"], seed=seed),
        layers.RandomRotation(aug_cfg["rotation"], fill_mode="reflect", seed=seed),
        layers.RandomZoom(aug_cfg["zoom"], fill_mode="reflect", seed=seed),
        layers.RandomBrightness(aug_cfg["brightness"], value_range=(0, 255), seed=seed),
        layers.RandomContrast(aug_cfg["contrast"], seed=seed),
    ], name="augmentation")


def make_dataset(df: pd.DataFrame, cfg: dict, preprocess_name: str, training: bool,
                 augment: bool | None = None, batch_size: int | None = None, cache: bool = True):
    """Build a batched tf.data.Dataset of (image, one-hot label) from a split DataFrame with a `path` column.

    Order: read + resize -> cache (uint8) -> [shuffle] -> batch -> [augment] -> scale -> prefetch.
    Augmentation is on by default only when `training` is True; it can never run on val/test
    unless explicitly requested. Shuffling is seeded and reshuffles every epoch.
    """
    import tensorflow as tf

    size = tuple(cfg["data"]["image_size"])
    n_classes = len(cfg["data"]["classes"])
    batch_size = batch_size or cfg["training"]["batch_size"]
    seed = cfg["seed"]
    augment = training if augment is None else augment
    preprocess = get_preprocess(preprocess_name)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].to_numpy(), df["label"].to_numpy()))
    ds = ds.map(lambda p, y: (read_and_resize(p, size), tf.one_hot(y, n_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        augmenter = build_augmenter(cfg["augmentation"], seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------------------
# 5. Command line
# --------------------------------------------------------------------------------------
def prepare_split(cfg: dict, force: bool = False, verify_identical: bool = True):
    """Load the metadata, build (or load) the split, run every leakage check.

    Returns (meta, split_df_with_paths, identical_groups, report).
    """
    set_seed(cfg["seed"])
    classes = cfg["data"]["classes"]
    meta = load_metadata(find_data_dir(cfg), classes)

    identical = find_identical_images(meta) if (verify_identical or force or not split_path(cfg).exists()) else []
    meta["group_id"] = assign_groups(meta, identical)

    if force or not split_path(cfg).exists():
        save_split(make_split(meta, cfg["data"]["split"], cfg["seed"]), cfg)
    split_df = load_split(cfg, meta)
    report = check_no_leakage(split_df, classes, identical)
    return meta, split_df, identical, report


def main() -> None:
    """Build or load the split, run the leakage checks, print the summary and class weights."""
    parser = argparse.ArgumentParser(description="Build / check the lesion-level data split.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force-split", action="store_true", help="rebuild the split even if the file exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, split_df, identical, report = prepare_split(cfg, force=args.force_split)
    classes = cfg["data"]["classes"]

    print("Split file:", split_path(cfg))
    print("Identical image groups (kept in one split):", identical)
    print("Leakage checks:", report)
    print()
    print(split_summary(split_df, classes).to_string())
    print("Share of images per split (%):", split_shares(split_df))
    print()
    print("Class percentage inside each split:")
    print(class_proportions(split_df, classes).to_string())
    print()
    print("Class weights (from train only):")
    for i, w in class_weights(split_df, len(classes)).items():
        print(f"  {i} {classes[i]:<6} {w:.3f}")


if __name__ == "__main__":
    main()

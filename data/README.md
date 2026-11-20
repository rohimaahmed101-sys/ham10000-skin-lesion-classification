# Data

The dataset is **not** stored in git (about 3 GB). Download it with:

```bash
python data/download.py
```

This uses [`kagglehub`](https://github.com/Kaggle/kagglehub) and needs a Kaggle API token in the `KAGGLE_API_TOKEN` environment variable (Colab: add it to **Secrets**).

## What the download contains

| Item | Used? |
|---|---|
| `HAM10000_metadata.csv` — one row per image: `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization` | Yes |
| `HAM10000_images_part_1/`, `HAM10000_images_part_2/` — 600 × 450 JPEG images | Yes |
| `ham10000_images_part_1/`, `ham10000_images_part_2/` — byte-identical copies of the same images | No (each image is loaded once, by `image_id`) |
| `hmnist_*.csv` — downscaled 28×28 and 8×8 versions | No |

## Source and licence

HAM10000, CC BY-NC 4.0. Tschandl, P., Rosendahl, C. & Kittler, H. *Sci. Data* 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

"""Data loading, cleaning, splitting and preprocessing.  (Owner: Data Engineer)

Implemented in Phase 2. Planned contents, from the Phase 1 dataset check:
- load_metadata(): read HAM10000_metadata.csv and map each image_id to ONE file
  (the Kaggle copy stores every image twice).
- split_by_lesion(): 70 / 15 / 15 split by lesion_id, stratified by class, so no lesion
  appears in two splits (spec 3.2). The 2 byte-identical image pairs stay together.
- build_datasets(): tf.data pipelines; resize to 224 x 224; augmentation on train only.
- leakage checks: assert no lesion_id or identical image is shared across splits.

Checklist items: E1, E2, E3, E4, E5, E7, E8.
"""

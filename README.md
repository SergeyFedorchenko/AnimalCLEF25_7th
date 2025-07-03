## Usage
# Step 1: Extract Features

python extract_features.py

This will create:

    output_gf.pickle — global features

    output_lf.pickle — local features for SuperPoint, DISK, ALIKED

# Step 2: Generate Keypoint Matches

python generate_matching.py

This will compute matching scores for all image pairs and save them to:

    output_kp.pickle

# Step 3: Train Boosting Model and Generate Submission

python fit_boosting.py

This trains a LightGBM binary classifier on pairwise features and produces:

    submission.csv

## Features Used for Boosting

    Cosine similarity of global features (score_mega)

    Orientation metadata for both images (orient1, orient2)

    Matched keypoint counts from:

        SuperPoint (score_sp_5, score_sp_8)

        DISK (score_ds_5, score_ds_8)

        A-Liked (ALIKED GLUE) (score_ae_5, score_ae_8)

## Requirements

    torch, torchvision, timm

    lightgbm, pandas, numpy

    wildlife_tools (internal or external dependency)

    Python 3.8+

## Citation

If you use this code for academic work or benchmarks, please cite our CLEF 2025 participation (to be updated after publication).


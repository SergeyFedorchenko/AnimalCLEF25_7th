## Usage
# Step 1: Download Data and Extract Features

kaggle competitions download -c animal-clef-2025

unzip animal-clef-2025.zip -d data/

python extract_features.py \
  --metadata_csv data/metadata.csv \
  --root_dir data/images \
  --ft_model checkpoints/ft_model.pt \
  --output_lf outputs/output_lf.pickle \
  --output_gf outputs/output_gf.pickle

This will create:

    output_gf.pickle — global features

    output_lf.pickle — local features for SuperPoint, DISK, ALIKED

# Step 2: Generate Keypoint Matches

python generate_matching.py \
  --metadata_csv data/metadata.csv \
  --output_gf outputs/output_gf.pickle \
  --output_lf outputs/output_lf.pickle \
  --output_kp outputs/output_kp.pickle \
  --best_k 150 \
  --batch_size_match 128

This will compute matching scores for all image pairs and save them to:

    output_kp.pickle

# Step 3: Train Boosting Model and Generate Submission

python fit_boosting.py \
  --metadata_csv data/metadata.csv \
  --output_gf outputs/output_gf.pickle \
  --output_kp outputs/output_kp.pickle \
  --submission_csv outputs/submission.csv \
  --best_k 150 \
  --thresh 0.75

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


export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python train_ml.py \
  --train_file train_sample73_feature.txt \
  --valid_file valid_sample73_feature.txt \
  --test_file test_all_sample73_feature.txt \
  --mode mlp \
  --scale \
  --seed 42 \
  --output_prefix mlp_out \
  --threshold 0.5 --smote

python train_ml.py \
  --train_file train_sample73_feature.txt \
  --valid_file valid_sample73_feature.txt \
  --test_file test_all_sample73_feature.txt \
  --mode logistic \
  --scale \
  --seed 42 \
  --output_prefix logistic_out \
  --threshold 0.5 --smote

python train_ml.py \
  --train_file train_sample73_feature.txt \
  --valid_file valid_sample73_feature.txt \
  --test_file test_all_sample73_feature.txt \
  --mode random_forest \
  --scale \
  --seed 42 \
  --output_prefix random_forest_out \
  --threshold 0.5 --smote

python train_ml.py \
  --train_file train_sample73_feature.txt \
  --valid_file valid_sample73_feature.txt \
  --test_file test_all_sample73_feature.txt \
  --mode xgboost \
  --scale \
  --seed 42 \
  --output_prefix xgboost_out \
  --threshold 0.5 --smote

python train_ml.py \
  --train_file train_sample73_feature.txt \
  --valid_file valid_sample73_feature.txt \
  --test_file test_all_sample73_feature.txt \
  --mode knn \
  --scale \
  --seed 42 \
  --output_prefix knn_out \
  --threshold 0.5 --smote


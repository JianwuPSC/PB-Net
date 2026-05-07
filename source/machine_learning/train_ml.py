#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Model Trainer for PPI prediction (item level).
After training, test metrics are printed and saved.
Additionally, standardized test features are exported to CSV.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)
    names = data[:, 0]
    features = {
        'nearest_dis': data[:, 1].astype(float),
        'counts_8A': data[:, 2].astype(float),
        'binder_PAE': data[:, 3].astype(float),
        'binder_RMSD': data[:, 4].astype(float),
        'inter_PAE': data[:, 5].astype(float),
        'query_PAE': data[:, 6].astype(float),
        'binder_pLDDT': data[:, 7].astype(float),
        'query_pLDDT': data[:, 8].astype(float),
        'total_pLDDT': data[:, 9].astype(float),
        'query_RMSD': data[:, 10].astype(float),
        '3d_score': data[:, 11].astype(float),
        'ss_score': data[:, 12].astype(float),
        'blast_score': data[:, 13].astype(float),
        'f1': data[:, 14].astype(float),
        'f2': data[:, 15].astype(float),
        'f3': data[:, 16].astype(float),
        'f4': data[:, 17].astype(float),
        'f5': data[:, 18].astype(float),
    }
    df = pd.DataFrame(features)
    feature_names = df.columns.tolist()
    for col in ['f1', 'f2', 'f3', 'f4', 'f5']:
        df[col] = df[col].clip(upper=5) # 10
    target = (data[:, 19].astype(str) == '1').astype(int)
    return df, target, names, feature_names

def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = np.nan
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'auroc': auroc, 'auprc': auprc}

def get_model(model_name, random_state=1):
    if model_name == 'logistic':
        return LogisticRegression(random_state=random_state)
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(objective='reg:squarederror', colsample_bytree=0.3,
                                 learning_rate=0.1, max_depth=5, alpha=10,
                                 n_estimators=10, random_state=random_state)
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                    leaf_size=30, p=2, metric='minkowski')
    elif model_name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, activation='relu',
                             solver='adam', alpha=0.001, learning_rate='constant',
                             learning_rate_init=0.001, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def find_best_f1_threshold(y_true, y_score):
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = -1.0
    best_thr = 0.5
    best_metrics = None
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {'threshold': thr, 'precision': prec, 'recall': rec,
                            'f1': f1, 'accuracy': acc}
    return best_thr, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['logistic', 'random_forest', 'xgboost', 'knn', 'mlp'])
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, default='ml_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', action='store_true', default=True)
    parser.add_argument('--no_scale', dest='scale', action='store_false')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--smote', action='store_true')
    parser.set_defaults(scale=True)
    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"Loading training data from {args.train_file}...")
    X_train, y_train, train_names, feature_names = load_data(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    X_test, y_test, test_names, _ = load_data(args.test_file)

    if args.valid_file:
        print(f"Loading validation data from {args.valid_file}...")
        X_valid, y_valid, valid_names, _ = load_data(args.valid_file)
    else:
        X_valid, y_valid = None, None

    if args.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_valid is not None:
            X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    # ---------- 导出标准化后的测试特征 CSV ----------
    test_feat_df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(X_test.shape[1])])
    test_feat_df.insert(0, 'name', test_names)
    test_feat_df['label'] = y_test
    test_feat_csv = f"{args.output_prefix}_test_standardized.csv"
    test_feat_df.to_csv(test_feat_csv, index=False)
    print(f"Standardized test features saved to {test_feat_csv}")
    # ------------------------------------------------

    if args.smote:
        print(f"Applying SMOTE: train shape before {X_train.shape}, class distribution {np.bincount(y_train)}")
        smote = SMOTE(random_state=args.seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: train shape {X_train.shape}, class distribution {np.bincount(y_train)}")

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    model = get_model(args.mode, args.seed)
    print(f"Training {args.mode} model...")
    model.fit(X_train, y_train)

    if args.mode == 'logistic':
        coef = model.coef_.flatten()
        coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
        coef_df.to_csv(f"{args.output_prefix}_coefficients.csv", index=False)

    if X_valid is not None:
        y_valid_prob = model.predict_proba(X_valid)[:, 1]
        y_valid_pred = (y_valid_prob >= args.threshold).astype(int)
        valid_metrics = compute_metrics(y_valid, y_valid_pred, y_valid_prob)
        print(f"\n========== Validation Set Metrics (threshold={args.threshold:.2f}) ==========")
        for k, v in valid_metrics.items():
            print(f"{k}: {v:.4f}")

    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred_manual = (y_test_prob >= args.threshold).astype(int)
    test_metrics_manual = compute_metrics(y_test, y_test_pred_manual, y_test_prob)

    print(f"\n========== Test Set Metrics (manual threshold = {args.threshold:.2f}) ==========")
    for k, v in test_metrics_manual.items():
        print(f"{k}: {v:.4f}")

    best_thr, best_metrics = find_best_f1_threshold(y_test, y_test_prob)
    y_test_pred_best = (y_test_prob >= best_thr).astype(int)
    best_test_metrics = compute_metrics(y_test, y_test_pred_best, y_test_prob)

    print(f"\n=== Best F1 threshold for test set = {best_thr:.2f} ===")
    for k, v in best_test_metrics.items():
        print(f"{k}: {v:.4f}")

    results = pd.DataFrame({
        'name': test_names,
        'true_label': y_test,
        'prob_positive': y_test_prob,
        'pred_label_manual': y_test_pred_manual,
        'pred_label_best_f1': y_test_pred_best
    })
    results.to_csv(f"{args.output_prefix}_predictions.csv", index=False)

    pd.DataFrame([test_metrics_manual]).to_csv(f"{args.output_prefix}_metrics_manual_thr.csv", index=False)
    pd.DataFrame([best_test_metrics]).to_csv(f"{args.output_prefix}_metrics_best_f1.csv", index=False)
    pd.DataFrame([best_metrics]).to_csv(f"{args.output_prefix}_best_f1_threshold.csv", index=False)

    print(f"Results saved with prefix: {args.output_prefix}")
    print("Done.")

if __name__ == '__main__':
    main()

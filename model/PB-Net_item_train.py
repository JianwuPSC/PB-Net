#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SimpleMLP model for PPI prediction (item-level, with binder).
After training, evaluates on the test set using a standalone test logic
that is 100% consistent with PB-Net_item_test.py.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# ---------------------------- Set Random Seed ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------- Model Definition ----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# ---------------------------- Feature Loading ----------------------------
def load_features(source_file, binder_file, target_file, saprot_source_file, saprot_binder_file, saprot_target_file):
    esm_source = pd.read_csv(source_file, header=None)
    esm_binder = pd.read_csv(binder_file, header=None)
    esm_target = pd.read_csv(target_file, header=None)

    saprot_source = pd.read_csv(saprot_source_file, header=None)
    saprot_binder = pd.read_csv(saprot_binder_file, header=None)
    saprot_target = pd.read_csv(saprot_target_file, header=None)

    name_df = saprot_source[[0]].copy()
    name_df.columns = ['name']
    class_series = saprot_source[1281].copy()
    class_df = pd.DataFrame(class_series)
    class_df.columns = ['class']

    esm_source = esm_source.iloc[:, 1:-1]
    esm_binder = esm_binder.iloc[:, 1:-1]
    esm_target = esm_target.iloc[:, 1:-1]

    saprot_source = saprot_source.iloc[:, 1:-1]
    saprot_binder = saprot_binder.iloc[:, 1:-1]
    saprot_target = saprot_target.iloc[:, 1:-1]

    return {
        'esm_s': esm_source, 'esm_b': esm_binder, 'esm_t': esm_target,
        'sap_s': saprot_source, 'sap_b': saprot_binder, 'sap_t': saprot_target,
        'name': name_df, 'class': class_df
    }

def build_features(feature_dict, mode):
    esm_s = feature_dict['esm_s']
    esm_b = feature_dict['esm_b']
    esm_t = feature_dict['esm_t']
    sap_s = feature_dict['sap_s']
    sap_b = feature_dict['sap_b']
    sap_t = feature_dict['sap_t']

    if mode == 'esm2_source_binder_target':
        X = pd.concat([esm_s, esm_b, esm_t], axis=1)
    elif mode == 'saprot_source_binder_target':
        X = pd.concat([sap_s, sap_b, sap_t], axis=1)
    elif mode == 'esm2_match_source_binder_target':
        X = esm_s - esm_b - esm_t
    elif mode == 'saprot_match_source_binder_target':
        X = sap_s - sap_b - sap_t
    elif mode == 'cat_esm2_match_source_binder_target_saprot_match_source_binder_target':
        X1 = esm_s - esm_b - esm_t
        X2 = sap_s - sap_b - sap_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_esm2_source_binder_target_saprot_match_source_binder_target':
        X1 = pd.concat([esm_s, esm_b, esm_t], axis=1)
        X2 = sap_s - sap_b - sap_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_saprot_source_binder_target_esm2_match_source_binder_target':
        X1 = pd.concat([sap_s, sap_b, sap_t], axis=1)
        X2 = esm_s - esm_b - esm_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_esm_source_target':
        X = pd.concat([esm_s, esm_t], axis=1)
    elif mode == 'cat_saprot_source_target':
        X = pd.concat([sap_s, sap_t], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X = X.fillna(0)
    return X.values.astype(np.float32)

# ---------------------------- Evaluation Metrics (pos_label=1) ----------------------------
def compute_metrics(y_true, y_pred, y_score, pos_label=1):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score, pos_label=pos_label)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc
    }

# ---------------------------- Training Loop (NO test evaluation) ----------------------------
def train_model(train_loader, val_loader, input_size, args, class_weight=None):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(input_size, hidden_size=args.hidden_size, dropout_rate=args.dropout).to(device)

    if class_weight is not None:
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        val_scores = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                prob_class1 = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                pred = (prob_class1 >= 0.5).astype(int)
                val_scores.extend(prob_class1)
                val_preds.extend(pred)
                val_true.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(val_true, val_preds, val_scores, pos_label=1)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val Prec: {val_metrics['precision']:.4f} | "
              f"Val Rec: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        monitor_value = val_metrics[args.monitor]
        if args.monitor_mode == 'max':
            improved = monitor_value > best_val_f1
        else:
            improved = val_loss < best_val_loss

        if improved:
            if args.monitor_mode == 'max':
                best_val_f1 = monitor_value
            else:
                best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, args.save_model)
            print(f"  -> Best model saved (improved {args.monitor})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    model.load_state_dict(best_model_state)
    return model

# ---------------------------- Independent Test Function ----------------------------
def evaluate_test_set(args):
    """
    Replicates the standalone test script logic.
    Loads the saved model and scaler, then predicts on the test set,
    prints metrics and saves all output files.
    """
    print("\n=============================================================")
    print("Running independent test evaluation...")
    # Set seed for reproducibility (must match training)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test features directly from the original test files
    print("Loading test data...")
    test_feat = load_features(args.test_source_esm, args.test_binder_esm, args.test_target_esm,
                              args.test_source_saprot, args.test_binder_saprot, args.test_target_saprot)
    X_test = build_features(test_feat, args.mode)
    names = test_feat['name'].values.ravel()
    y_test = test_feat['class'].values.ravel().astype(np.int64)
    print(f"Test data shape: {X_test.shape}")

    # Load scaler from disk
    scaler_path = f"{args.save_predictions}_scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test = scaler.transform(X_test)
    print("First test sample after scaler:", X_test[0, :5])

    # Load model
    input_size = X_test.shape[1]
    model = SimpleMLP(input_size, hidden_size=args.hidden_size, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(args.save_model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.save_model}")

    # DataLoader
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Predict
    all_probs = []
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            prob_class1 = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            pred = (prob_class1 >= 0.5).astype(int)
            all_probs.extend(prob_class1)
            all_preds.extend(pred)
            all_true.extend(batch_y.numpy())

    # Item-level results
    item_df = pd.DataFrame({
        'name': names,
        'pred_label': all_preds,
        'prob_positive': all_probs
    })
    y_true_item = np.array(all_true)
    item_df['true_label'] = y_true_item

    # Compute metrics at threshold 0.5
    item_metrics = compute_metrics(y_true_item, all_preds, all_probs, pos_label=1)
    print("\n========== Test Set Metrics (Item-level, threshold=0.5) ==========")
    for k, v in item_metrics.items():
        print(f"{k}: {v:.4f}")

    # Multi-threshold scanning for item-level
    thresholds = np.arange(0.0, 1.01, 0.01)
    item_metrics_list = []
    for thr in thresholds:
        y_pred = (all_probs >= thr).astype(int)
        prec = precision_score(y_true_item, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true_item, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true_item, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true_item, y_pred)
        item_metrics_list.append({'threshold': thr, 'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc})
    item_metrics_df = pd.DataFrame(item_metrics_list)
    best_idx = item_metrics_df['f1'].idxmax()
    best_item_row = item_metrics_df.loc[best_idx]
    best_item_info = {
        'threshold': best_item_row['threshold'],
        'precision': best_item_row['precision'],
        'recall': best_item_row['recall'],
        'f1': best_item_row['f1'],
        'accuracy': best_item_row['accuracy'],
        'roc_auc': item_metrics['roc_auc'],
        'auprc': item_metrics['auprc']
    }
    print(f"\n=== Item-level Best F1 @ threshold {best_item_info['threshold']:.2f} ===")
    print(f"Precision={best_item_info['precision']:.4f}, Recall={best_item_info['recall']:.4f}, F1={best_item_info['f1']:.4f}, Accuracy={best_item_info['accuracy']:.4f}")

    # Save item-level files
    prefix = args.save_predictions
    item_df.to_csv(f"{prefix}_item_predictions.csv", index=False)
    pd.DataFrame([item_metrics]).to_csv(f"{prefix}_item_overall_metrics.csv", index=False)
    pd.DataFrame([best_item_info]).to_csv(f"{prefix}_item_best_f1.csv", index=False)

    # Pair-level if requested
    if args.compute_pair_metrics:
        import sys
        # Simple bait/target extraction
        def extract_bait_target(name):
            parts = name.split('_')
            bait = parts[0]
            target = parts[-2]
            return bait, target

        bait_list = [extract_bait_target(n)[0] for n in names]
        target_list = [extract_bait_target(n)[1] for n in names]
        item_df['bait'] = bait_list
        item_df['target'] = target_list

        # Pair aggregation function (inline, identical to test script)
        def compute_pair_metrics_from_items(item_df, manual_threshold=0.5):
            pair_groups = item_df.groupby(['bait', 'target'])
            pair_records = []
            for (bait, target), group in pair_groups:
                true_label = group['true_label'].iloc[0] if 'true_label' in group else None
                any_positive = (group['prob_positive'] >= manual_threshold).any()
                pair_pred_label = 1 if any_positive else 0
                max_prob = group['prob_positive'].max()
                record = {
                    'bait': bait, 'target': target,
                    'pred_label': pair_pred_label,
                    'max_prob': max_prob, 'n_items': len(group)
                }
                if true_label is not None:
                    record['true_label'] = true_label
                pair_records.append(record)
            return pd.DataFrame(pair_records)

        pair_df = compute_pair_metrics_from_items(item_df, manual_threshold=args.manual_threshold)
        pair_df.to_csv(f"{prefix}_pair_results.csv", index=False)

        if 'true_label' in pair_df.columns:
            y_true_pair = pair_df['true_label'].values.ravel()
            y_score_pair = pair_df['max_prob'].values
            pred_pair = pair_df['pred_label'].values

            pair_metrics = compute_metrics(y_true_pair, pred_pair, y_score_pair, pos_label=1)
            roc_auc = pair_metrics['roc_auc']
            auprc = pair_metrics['auprc']
            print(f"\n=== Protein Pair Level (manual threshold = {args.manual_threshold}) ===")
            print(f"Precision={pair_metrics['precision']:.4f}, Recall={pair_metrics['recall']:.4f}, F1={pair_metrics['f1']:.4f}, Accuracy={pair_metrics['accuracy']:.4f}, ROC AUC={roc_auc:.4f}, AUPRC={auprc:.4f}")

            # Multi-threshold scanning for pair level
            thresholds = np.arange(0.0, 1.01, 0.01)
            metrics_list = []
            for thr in thresholds:
                y_pred = np.where(y_score_pair >= thr, 1, 0)
                tp = np.sum((y_true_pair == 1) & (y_pred == 1))
                fp = np.sum((y_true_pair != 1) & (y_pred == 1))
                fn = np.sum((y_true_pair == 1) & (y_pred != 1))
                tn = np.sum((y_true_pair != 1) & (y_pred != 1))
                precision = tp / (tp+fp) if (tp+fp)>0 else np.nan
                recall = tp / (tp+fn) if (tp+fn)>0 else np.nan
                f1_thr = 2 * precision * recall / (precision+recall) if (precision+recall)>0 else np.nan
                accuracy = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else np.nan
                metrics_list.append({
                    'threshold': thr, 'precision': precision, 'recall': recall,
                    'f1': f1_thr, 'accuracy': accuracy,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                })
            pair_metrics_df = pd.DataFrame(metrics_list)
            best_idx = pair_metrics_df['f1'].idxmax()
            best_pair_row = pair_metrics_df.loc[best_idx]
            print(f"\n=== Pair-level Best F1 @ threshold {best_pair_row['threshold']:.2f} ===")
            print(f"Precision={best_pair_row['precision']:.4f}, Recall={best_pair_row['recall']:.4f}, F1={best_pair_row['f1']:.4f}, Accuracy={best_pair_row['accuracy']:.4f}")

            pair_metrics_df.to_csv(f"{prefix}_pair_threshold_metrics.csv", index=False)
            pd.DataFrame([{
                'threshold': best_pair_row['threshold'],
                'precision': best_pair_row['precision'],
                'recall': best_pair_row['recall'],
                'f1': best_pair_row['f1'],
                'accuracy': best_pair_row['accuracy'],
                'roc_auc': roc_auc,
                'auprc': auprc
            }]).to_csv(f"{prefix}_pair_best_f1.csv", index=False)

    print(f"\nAll results saved with prefix: {args.save_predictions}")
    print("=============================================================\n")

# ---------------------------- Main ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train SimpleMLP for PPI prediction (positive=1, negative=0)')
    # Input files
    parser.add_argument('--train_source_esm', type=str, required=True)
    parser.add_argument('--train_binder_esm', type=str, required=True)
    parser.add_argument('--train_target_esm', type=str, required=True)
    parser.add_argument('--train_source_saprot', type=str, required=True)
    parser.add_argument('--train_binder_saprot', type=str, required=True)
    parser.add_argument('--train_target_saprot', type=str, required=True)
    parser.add_argument('--valid_source_esm', type=str, required=True)
    parser.add_argument('--valid_binder_esm', type=str, required=True)
    parser.add_argument('--valid_target_esm', type=str, required=True)
    parser.add_argument('--valid_source_saprot', type=str, required=True)
    parser.add_argument('--valid_binder_saprot', type=str, required=True)
    parser.add_argument('--valid_target_saprot', type=str, required=True)
    parser.add_argument('--test_source_esm', type=str, required=True)
    parser.add_argument('--test_binder_esm', type=str, required=True)
    parser.add_argument('--test_target_esm', type=str, required=True)
    parser.add_argument('--test_source_saprot', type=str, required=True)
    parser.add_argument('--test_binder_saprot', type=str, required=True)
    parser.add_argument('--test_target_saprot', type=str, required=True)

    parser.add_argument('--mode', type=str, default='cat_saprot_source_binder_target_esm2_match_source_binder_target',
                        choices=['esm2_source_binder_target',
                                 'saprot_source_binder_target',
                                 'esm2_match_source_binder_target',
                                 'saprot_match_source_binder_target',
                                 'cat_esm2_match_source_binder_target_saprot_match_source_binder_target',
                                 'cat_esm2_source_binder_target_saprot_match_source_binder_target',
                                 'cat_saprot_source_binder_target_esm2_match_source_binder_target',
                                 'cat_esm_source_target',
                                 'cat_saprot_source_target'])

    parser.add_argument('--balance_strategy', type=str, default='class_weight',
                        choices=['smote', 'class_weight', 'none'],
                        help='How to handle class imbalance: smote (oversample), class_weight (weighted loss), none')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight (label=1). Only used when balance_strategy=class_weight.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--monitor', type=str, default='f1', choices=['f1', 'loss', 'accuracy'])
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=12, help='Random seed for reproducibility')

    parser.add_argument('--save_model', type=str, default='best_model.pt')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Output prefix for all result files (e.g., results/test)')
    parser.add_argument('--manual_threshold', type=float, default=0.5,
                        help='Threshold for pair-level positive class (1)')

    # Additional flags for test evaluation
    parser.add_argument('--compute_pair_metrics', action='store_true',
                        help='Also compute and save protein-pair level metrics after training')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    print("Loading data...")
    train_feat = load_features(args.train_source_esm, args.train_binder_esm, args.train_target_esm,
                               args.train_source_saprot, args.train_binder_saprot, args.train_target_saprot)
    valid_feat = load_features(args.valid_source_esm, args.valid_binder_esm, args.valid_target_esm,
                               args.valid_source_saprot, args.valid_binder_saprot, args.valid_target_saprot)
    # Test features are loaded in evaluate_test_set() to ensure fresh, consistent loading

    X_train = build_features(train_feat, args.mode)
    y_train = train_feat['class'].values.ravel().astype(np.int64)
    X_valid = build_features(valid_feat, args.mode)
    y_valid = valid_feat['class'].values.ravel().astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Save scaler to file (for later use by evaluate_test_set)
    if args.save_predictions:
        with open(f"{args.save_predictions}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    else:
        print("Warning: --save_predictions is required to save scaler and output files.")
        return

    # Handle class imbalance
    class_weight = None
    if args.balance_strategy == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=args.seed, sampling_strategy='auto')
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: Train shape {X_train.shape}, class distribution {np.bincount(y_train)}")
    elif args.balance_strategy == 'class_weight':
        if args.pos_weight is not None:
            class_weight = [1.0, args.pos_weight]
            print(f"Using manual class weights: negative (0) = 1.0, positive (1) = {args.pos_weight}")
        else:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight = class_weights.tolist()
            print(f"Using automatically balanced class weights: {class_weight}")
    else:
        print("No balancing applied.")

    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
    print(f"Train class distribution (0=negative,1=positive): {np.bincount(y_train)}")
    print(f"Valid class distribution: {np.bincount(y_valid)}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                  torch.tensor(y_valid, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = X_train.shape[1]
    print(f"Input size: {input_size}")

    # Train model (no test evaluation inside)
    model = train_model(train_loader, val_loader, input_size, args, class_weight=class_weight)

    # After training, run independent test evaluation
    evaluate_test_set(args)

    print("Done.")

if __name__ == '__main__':
    main()

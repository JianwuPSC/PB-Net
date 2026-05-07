#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SimpleMLP model for pair-level PPI prediction.
After training, test evaluation is performed independently using the saved model and scaler,
exactly like the standalone test script, ensuring identical metrics.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
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

# ---------------------------- Feature Loading (pair level) ----------------------------
def load_pair_features(source_file, target_file):
    """
    Load source and target feature files (CSV, no header).
    Format: first column = name, last column = label (1 = positive, 0 = negative).
    Returns: X (concatenated features), y (labels), names (sample names)
    """
    source_df = pd.read_csv(source_file, header=None)
    target_df = pd.read_csv(target_file, header=None)

    name = source_df[0].values
    label = source_df[source_df.shape[1]-1].values  # assume 1 = positive, 0 = negative

    source_feat = source_df.iloc[:, 1:-1]
    target_feat = target_df.iloc[:, 1:-1]

    X = pd.concat([source_feat, target_feat], axis=1)
    X = X.fillna(0).values.astype(np.float32)
    y = label.astype(np.int64)
    return X, y, name

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
    best_val_metric = 0.0
    patience_counter = 0
    best_model_state = None
    best_val_metrics = None

    if args.monitor_mode == 'max':
        monitor_better = lambda x, best: x > best
    else:
        monitor_better = lambda x, best: x < best

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

        # Determine current monitored value
        if args.monitor == 'loss':
            current_val = val_loss
        elif args.monitor == 'accuracy':
            current_val = val_metrics['accuracy']
        elif args.monitor == 'f1':
            current_val = val_metrics['f1']
        elif args.monitor == 'precision':
            current_val = val_metrics['precision']
        elif args.monitor == 'recall':
            current_val = val_metrics['recall']
        else:
            raise ValueError(f"Unknown monitor: {args.monitor}")

        # Check improvement
        if args.monitor == 'loss':
            best_reference = best_val_loss
        else:
            best_reference = best_val_metric

        if monitor_better(current_val, best_reference):
            if args.monitor == 'loss':
                best_val_loss = current_val
            else:
                best_val_metric = current_val
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, args.save_model)
            best_val_metrics = val_metrics.copy()
            print(f"  -> Best model saved (improved {args.monitor} = {current_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    # Load best model state
    model.load_state_dict(best_model_state)
    return model, best_val_metrics

# ---------------------------- Independent Test Function ----------------------------
def evaluate_test_set(args):
    """
    Replicates the standalone test script logic for pair-level.
    Loads the saved model and scaler, predicts on the test set,
    prints metrics and saves all output files.
    """
    print("\n=============================================================")
    print("Running independent test evaluation...")
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Choose test files based on mode
    if args.mode == 'cat_esm_source_target':
        test_source = args.test_source_esm
        test_target = args.test_target_esm
    else:  # 'cat_saprot_source_target'
        test_source = args.test_source_saprot
        test_target = args.test_target_saprot

    # Load test features
    print("Loading test data...")
    X_test, y_test, test_names = load_pair_features(test_source, test_target)
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

    y_true = np.array(all_true)
    y_score = np.array(all_probs)

    # Metrics at threshold 0.5
    test_metrics = compute_metrics(y_true, (y_score >= 0.5).astype(int), y_score, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    pr_df.to_csv("pair_pr_curve.csv", index=False)

    print("\n========== Test Set Metrics (threshold=0.5) ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Multi-threshold scanning
    thresholds = np.arange(0.0, 1.01, 0.01)
    metrics_list = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        metrics_list.append({
            'threshold': thr,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc
        })
    metrics_df = pd.DataFrame(metrics_list)

    # Determine best threshold using the same monitor and mode as training
    best_metric_name = args.monitor if args.monitor != 'loss' else 'f1'
    if best_metric_name not in metrics_df.columns:
        best_metric_name = 'f1'
    if args.monitor_mode == 'max':
        best_idx = metrics_df[best_metric_name].idxmax()
    else:
        best_idx = metrics_df[best_metric_name].idxmin() if best_metric_name == 'loss' else metrics_df[best_metric_name].idxmax()
    best_row = metrics_df.loc[best_idx]
    best_info = {
        'threshold': best_row['threshold'],
        'precision': best_row['precision'],
        'recall': best_row['recall'],
        'f1': best_row['f1'],
        'accuracy': best_row['accuracy'],
        'selected_metric': best_metric_name,
        'selected_value': best_row[best_metric_name]
    }

    print(f"\n=== Best according to {best_metric_name} ({args.monitor_mode}) @ threshold {best_info['threshold']:.2f} ===")
    print(f"Precision={best_info['precision']:.4f}, Recall={best_info['recall']:.4f}, "
          f"F1={best_info['f1']:.4f}, Accuracy={best_info['accuracy']:.4f}")

    # Save results
    prefix = args.save_predictions
    # Predictions CSV
    pred_best = (y_score >= best_info['threshold']).astype(int)
    pred_df = pd.DataFrame({
        'name': test_names,
        'true_label': y_true,
        'pred_label_best': pred_best,
        'prob_positive': y_score
    })
    pred_df.to_csv(f"{prefix}_predictions.csv", index=False)

    # Threshold metrics CSV
    metrics_df.to_csv(f"{prefix}_threshold_metrics.csv", index=False)

    # Best threshold info CSV
    pd.DataFrame([best_info]).to_csv(f"{prefix}_best_threshold.csv", index=False)

    # Overall metrics at threshold 0.5 CSV
    pd.DataFrame([test_metrics]).to_csv(f"{prefix}_overall_metrics.csv", index=False)

    print(f"\nAll results saved with prefix: {args.save_predictions}")
    print("=============================================================\n")

# ---------------------------- Main ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train SimpleMLP for PPI prediction (pair-level, positive=1, negative=0)')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['cat_esm_source_target', 'cat_saprot_source_target'],
                        help='Feature mode: cat_esm_source_target (ESM) or cat_saprot_source_target (Saprot)')

    parser.add_argument('--train_source_esm', type=str, required=True)
    parser.add_argument('--train_target_esm', type=str, required=True)
    parser.add_argument('--train_source_saprot', type=str, required=True)
    parser.add_argument('--train_target_saprot', type=str, required=True)

    parser.add_argument('--valid_source_esm', type=str, required=True)
    parser.add_argument('--valid_target_esm', type=str, required=True)
    parser.add_argument('--valid_source_saprot', type=str, required=True)
    parser.add_argument('--valid_target_saprot', type=str, required=True)

    parser.add_argument('--test_source_esm', type=str, required=True)
    parser.add_argument('--test_target_esm', type=str, required=True)
    parser.add_argument('--test_source_saprot', type=str, required=True)
    parser.add_argument('--test_target_saprot', type=str, required=True)

    parser.add_argument('--balance_strategy', type=str, default='class_weight',
                        choices=['smote', 'class_weight', 'none'])
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight (label=1). Only used when balance_strategy=class_weight.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--monitor', type=str, default='f1', choices=['f1', 'loss', 'accuracy', 'precision', 'recall'])
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--save_model', type=str, default='best_model.pt')
    parser.add_argument('--save_predictions', type=str, default=None)

    args = parser.parse_args()
    if args.save_predictions is None:
        parser.error("--save_predictions is required for saving scaler and results.")

    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Choose files based on mode
    if args.mode == 'cat_esm_source_target':
        train_source = args.train_source_esm
        train_target = args.train_target_esm
        valid_source = args.valid_source_esm
        valid_target = args.valid_target_esm
        print("Using ESM features for training.")
    else:
        train_source = args.train_source_saprot
        train_target = args.train_target_saprot
        valid_source = args.valid_source_saprot
        valid_target = args.valid_target_saprot
        print("Using Saprot features for training.")

    print("Loading data...")
    X_train, y_train, _ = load_pair_features(train_source, train_target)
    X_valid, y_valid, _ = load_pair_features(valid_source, valid_target)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Save scaler immediately after fitting on train (and transform valid)
    with open(f"{args.save_predictions}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

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

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                  torch.tensor(y_valid, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = X_train.shape[1]
    print(f"Input size: {input_size}")

    # Train model (no test inside)
    model, best_val_metrics = train_model(train_loader, val_loader, input_size, args, class_weight=class_weight)

    if best_val_metrics is not None:
        print("\n========== Best Validation Set Metrics (according to monitor) ==========")
        for k, v in best_val_metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        print("\nWarning: No best validation metrics recorded.")

    # Now run independent test evaluation
    evaluate_test_set(args)

    print("Done.")

if __name__ == '__main__':
    main()

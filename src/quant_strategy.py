import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

# ------------------ Setup ------------------
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ------------------ Model ------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ------------------ Utils ------------------
def create_sequences(X, y, time_steps=24):
    """
    Each sequence window [i : i+time_steps] predicts label at i+time_steps.
    No future data leaks into the window because we only look BACK.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def binary_accuracy(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


# ------------------ Main ------------------
def run_quant_strategy():
    print("Starting Quant Strategy (PyTorch LSTM)...")

    input_path = "data/alpha_features.csv"
    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)  # ensure chronological order
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    target_col = "Alpha_28"
    feature_cols = [c for c in df.columns if c not in ["timestamp", target_col]]

    X_raw = df[feature_cols].values
    y_raw = df[target_col].values.astype(int)
    timestamps = df["timestamp"].values

    # ------------------ Feature Expansion ------------------
    # User requested keeping only the 9 base features.
    X_final = X_raw
    print(f"Using {X_final.shape[1]} Base Features")

    # ------------------ Temporal Split (BEFORE sequencing) ------------------
    # We define the split boundary first so we can reason clearly about what
    # data the scaler and model ever see.
    split_date = pd.Timestamp("2025-01-01")

    # Raw temporal masks on the UNSEQUENCED data (for reference if needed)
    mask_train_raw = timestamps < split_date
    mask_test_raw = timestamps >= split_date

    # ------------------ Sequencing ------------------
    SEQ_LEN = 24

    X_seq, y_seq = create_sequences(X_final, y_raw, SEQ_LEN)
    # After sequencing, each sample i corresponds to the original row at index
    # (i + SEQ_LEN).  The label timestamp is therefore timestamps[i + SEQ_LEN].
    ts_seq = timestamps[SEQ_LEN:]  # aligned label timestamps

    # Apply temporal masks on the LABEL timestamp.
    # A test sequence's label is in Jan 2025+, but its 24-hour lookback window
    # may touch late-Dec 2024 rows — that is fine, because at prediction time
    # those Dec rows are already in the past.
    mask_train = ts_seq < split_date
    mask_test = ts_seq >= split_date

    X_train_full = X_seq[mask_train]
    y_train_full = y_seq[mask_train]
    X_test = X_seq[mask_test]
    y_test = y_seq[mask_test]

    print(f"Train Shape: {X_train_full.shape}, Test Shape: {X_test.shape}")

    # ------------------ Leak Assertion ------------------
    # Verify: every timestep inside every training sequence is strictly before
    # split_date.  (The last training label is the row just before split_date,
    # so all lookback rows are also before split_date.)
    train_label_timestamps = ts_seq[mask_train]
    assert np.all(train_label_timestamps < np.datetime64(split_date)), \
        "LEAK: Some training labels fall on or after the test split date."

    # Verify: no raw test-period row was used to fit anything yet.
    # (scaler hasn't been fit — this is just a sanity gate.)
    print("✓ Temporal split integrity verified — no label leak.")

    # ------------------ Train / Val Split (temporal, no shuffle) ------------------
    # Last 20% of training data becomes validation.  shuffle=False preserves
    # chronological order so val is always *later* than train.
    split_idx = int(len(X_train_full) * 0.8)

    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------ Scaling ------------------
    # Fit ONLY on training sequences.  Val and test are transformed with the
    # same statistics — never fitted on.
    scaler = StandardScaler()
    N, T, F = X_train.shape

    # Flatten all timesteps in all training sequences for fitting.
    scaler.fit(X_train.reshape(-1, F))

    def scale(x):
        n, t, f = x.shape
        return scaler.transform(x.reshape(-1, f)).reshape(n, t, f)

    X_train = scale(X_train)
    X_val   = scale(X_val)
    X_test  = scale(X_test)
    X_all   = scale(X_seq) # Scale EVERYTHING for full output generation

    # ------------------ Tensors ------------------
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_t  = torch.FloatTensor(X_test).to(device)
    X_all_t   = torch.FloatTensor(X_all).to(device) # Full Dataset Tensor

    # ------------------ Model ------------------
    model = LSTMClassifier(input_dim=F, hidden_dim=64, num_layers=2).to(device)

    # Class-weighted loss to handle imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True  # shuffling within train is fine; temporal integrity is
                      # already guaranteed by the hard split above.
    )

    # ------------------ Training ------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/lstm_model_weights.pth"
    
    # Delete old weights if exist
    if os.path.exists(model_path):
        os.remove(model_path)

    best_val_loss = float("inf")

    print("\nEpoch | TrainL | ValL   | TrainA | ValA")
    print("-" * 50)

    for epoch in range(1, 25):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += binary_accuracy(logits, by)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation (no grad, no update)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_acc    = binary_accuracy(val_logits, y_val_t)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            saved = " <-- best saved"

        print(
            f"{epoch:03d}   | "
            f"{train_loss:.4f} | "
            f"{val_loss:.4f} | "
            f"{train_acc:.3f}  | "
            f"{val_acc:.3f}{saved}"
        )

    # ------------------ Prediction (FULL DATASET) ------------------
    # Load best checkpoint (lowest val loss)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # Predict on ALL data (Train + Val + Test) for Visualization
        logits_all = model(X_all_t)
        probs_all = torch.sigmoid(logits_all).cpu().numpy().flatten()
    
    # ------------------ Save Predictions ------------------
    out_df = pd.DataFrame({
        "timestamp": ts_seq,   # All timestamps
        "quant_prob": probs_all,
        "actual_dir": y_seq
    })

    os.makedirs("output", exist_ok=True)
    out_df.to_csv("output/quant_predictions.csv", index=False)
    print("\nFull dataset predictions saved to output/quant_predictions.csv")

    # ------------------ Test Accuracy ------------------
    # Slice out test set results from full predictions
    # We already have mask_test aligned to ts_seq
    test_probs = probs_all[mask_test]
    test_actuals = y_seq[mask_test]
    
    test_preds = (test_probs > 0.5).astype(int)
    acc = (test_preds == test_actuals).mean()
    print(f"LSTM Test Accuracy: {acc:.2%}")


if __name__ == "__main__":
    run_quant_strategy()

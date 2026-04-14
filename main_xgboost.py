import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_PATH = "submission_xgb.csv"
TARGET = "settlement_index"

DROP_COLS = [
    "id",
    "planet_name",
    TARGET
]

TEXT_COLS = [
    "exploration_log",
    "environmental_report",
    "incident_report",
    "image_prompt",
    "name",
    "description",
]

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

test_ids = test_df["id"].copy()
y = train_df[TARGET].values.astype(np.float32)

def prepare_tabular_for_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame):
    drop_for_tabular = DROP_COLS + TEXT_COLS

    train_tab = train_df.drop(columns=[c for c in drop_for_tabular if c in train_df.columns]).copy()
    test_tab = test_df.drop(columns=[c for c in drop_for_tabular if c in test_df.columns]).copy()

    numeric_cols = train_tab.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_tab.columns if c not in numeric_cols]

    for col in numeric_cols:
        med = train_tab[col].median()
        train_tab[col] = train_tab[col].fillna(med)
        test_tab[col] = test_tab[col].fillna(med)

    for col in categorical_cols:
        train_tab[col] = train_tab[col].fillna("missing").astype(str)
        test_tab[col] = test_tab[col].fillna("missing").astype(str)

    all_tab = pd.concat([train_tab, test_tab], axis=0, ignore_index=True)
    all_tab = pd.get_dummies(all_tab, columns=categorical_cols, dummy_na=False)

    train_encoded = all_tab.iloc[:len(train_tab)].copy()
    test_encoded = all_tab.iloc[len(train_tab):].copy()

    return train_encoded, test_encoded

X_all, X_test = prepare_tabular_for_xgb(train_df, test_df)

print("Encoded feature dim:", X_all.shape[1])

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE
)

X_train = X_all.iloc[train_idx].values
X_val = X_all.iloc[val_idx].values
y_train = y[train_idx]
y_val = y[val_idx]

print("Train split:", len(train_idx), "Val split:", len(val_idx))

model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=200
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"[XGBoost] Val RMSE: {val_rmse:.5f}")

# 最终再切一份更大的训练集
full_idx = np.arange(len(train_df))
full_train_idx, full_val_idx = train_test_split(
    full_idx,
    test_size=0.1,
    random_state=RANDOM_STATE
)

X_full_train = X_all.iloc[full_train_idx].values
X_full_val = X_all.iloc[full_val_idx].values
y_full_train = y[full_train_idx]
y_full_val = y[full_val_idx]

final_model = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.025,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=200
)

final_model.fit(
    X_full_train,
    y_full_train,
    eval_set=[(X_full_val, y_full_val)],
    verbose=100
)

holdout_pred = final_model.predict(X_full_val)
holdout_rmse = np.sqrt(mean_squared_error(y_full_val, holdout_pred))
print(f"[Final XGBoost] Holdout RMSE: {holdout_rmse:.5f}")

test_pred = final_model.predict(X_test.values)
test_pred = np.clip(test_pred, 0, 100)

submission = pd.DataFrame({
    "id": test_ids,
    "settlement_index": test_pred
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission to: {OUTPUT_PATH}")
print(submission.head())
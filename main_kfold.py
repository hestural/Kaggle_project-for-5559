import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
N_FOLDS = 5

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_PATH = "submission_kfold.csv"

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

def prepare_tabular(train_df, test_df):
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

    return train_tab, test_tab, categorical_cols

X_all, X_test, cat_cols = prepare_tabular(train_df, test_df)

cat_idx = [X_all.columns.get_loc(c) for c in cat_cols]


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(train_df))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n===== Fold {fold+1} =====")

    X_train = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=4000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        bagging_temperature=1.0,
        random_seed=RANDOM_STATE + fold,
        od_type="Iter",
        od_wait=200,
        verbose=200,
        task_type="GPU",
        devices="0"
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_pred = model.predict(X_val)
    oof_pred[val_idx] = val_pred

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"[Fold {fold+1}] RMSE: {rmse:.5f}")

    test_pred = model.predict(X_test)
    test_preds.append(test_pred)

oof_rmse = np.sqrt(mean_squared_error(y, oof_pred))
print(f"\n[OOF RMSE]: {oof_rmse:.5f}")

final_pred = np.mean(test_preds, axis=0)
final_pred = np.clip(final_pred, 0, 100)

submission = pd.DataFrame({
    "id": test_ids,
    "settlement_index": final_pred
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")
print(submission.head())
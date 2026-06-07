import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
COFFEE_COLS = [
    'Americano', 'Americano with Milk', 'Cappuccino',
    'Cocoa', 'Cortado', 'Espresso', 'Hot Chocolate', 'Latte'
]


# ──────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='coffee_sales_preprocessing')
    return parser.parse_args()


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_data(data_dir):
    X_train = pd.read_csv(f'{data_dir}/X_train.csv').values
    X_val   = pd.read_csv(f'{data_dir}/X_val.csv').values
    X_test  = pd.read_csv(f'{data_dir}/X_test.csv').values
    y_train = pd.read_csv(f'{data_dir}/y_train.csv', index_col=0)
    y_val   = pd.read_csv(f'{data_dir}/y_val.csv',   index_col=0)
    y_test  = pd.read_csv(f'{data_dir}/y_test.csv',  index_col=0)
    print(f"[load_data] X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────
def evaluate(y_true, y_pred, prefix=''):
    metrics = {}
    for i, col in enumerate(COFFEE_COLS):
        metrics[f'{prefix}{col}_rmse'] = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        metrics[f'{prefix}{col}_mae']  = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
        metrics[f'{prefix}{col}_r2']   = r2_score(y_true.iloc[:, i], y_pred[:, i])
    metrics[f'{prefix}overall_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[f'{prefix}overall_mae']  = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}overall_r2']   = r2_score(y_true, y_pred)
    return metrics


# ──────────────────────────────────────────────
# ARTEFAK 1: PLOT PREDIKSI VS AKTUAL
# ──────────────────────────────────────────────
def plot_prediction_vs_actual(y_true, y_pred):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, col in enumerate(COFFEE_COLS):
        axes[i].plot(y_true.iloc[:, i].values, label='Actual', color='steelblue')
        axes[i].plot(y_pred[:, i], label='Predicted', color='darkorange', linestyle='--')
        axes[i].set_title(col)
        axes[i].legend()
    plt.suptitle('Actual vs Predicted — test', fontsize=14)
    plt.tight_layout()
    path = 'plot_pred_vs_actual.png'
    plt.savefig(path)
    plt.close()
    return path


# ──────────────────────────────────────────────
# ARTEFAK 2: FEATURE IMPORTANCE
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_cols):
    importance = np.abs(model.coef_).mean(axis=0)
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='steelblue')
    plt.title('Top 20 Feature Importance (Mean |Coef|)')
    plt.xlabel('Mean Absolute Coefficient')
    plt.tight_layout()
    path = 'plot_feature_importance.png'
    plt.savefig(path)
    plt.close()
    return path


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run_modelling(data_dir):
    dagshub.init(
        repo_owner='andikaap99',
        repo_name='Eksperimen_SML_Andika-Aryadi-Putra',
        mlflow=True
    )

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir)
    feature_cols = pd.read_csv(f'{data_dir}/X_train.csv').columns.tolist()

    mlflow.set_experiment("coffee_sales_forecasting")

    with mlflow.start_run(run_name="MLR_baseline_CI"):
        mlflow.sklearn.autolog()

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_val  = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        val_metrics  = evaluate(y_val,  y_pred_val,  prefix='val_')
        test_metrics = evaluate(y_test, y_pred_test, prefix='test_')
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        path_pred = plot_prediction_vs_actual(y_test, y_pred_test)
        mlflow.log_artifact(path_pred)

        path_feat = plot_feature_importance(model, feature_cols)
        mlflow.log_artifact(path_feat)

        print("\n=== TEST METRICS ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    args = parse_args()
    run_modelling(args.data_dir)
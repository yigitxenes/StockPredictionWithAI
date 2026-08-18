import numpy as np
import optuna
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from models.walk_forward import generate_walk_forward_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)
def suggest_xgb_params(trial, task):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "random_state": 42,
    }
    if task == "classification":
        params["eval_metric"] = "logloss"
    return params

def suggest_lighgbm_params(trial, task):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    return params

def build_model(model_family, task, params):
    if model_family == "xgboost":
        return XGBClassifier(**params) if task == "classification" else XGBRegressor(**params)
    elif model_family == "lightgbm":
        return LGBMClassifier(**params) if task == "classification" else LGBMRegressor(**params)
    raise ValueError(f"Unknown model family : {model_family}")


def evaluate_classification(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    pred = model.predict(X_val)
    auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    acc = accuracy_score(y_val, pred)
    # Literatur onerisiyle uyumlu birlesik skor: AUC agirlikli, PR-AUC ve
    # accuracy'yi de dahil ederek tek bir Optuna hedefi olusturuyoruz
    combined = 0.5 * auc + 0.3 * pr_auc + 0.2 * acc
    return combined, {"auc" : auc, "pr_auc": pr_auc, "accuracy" : acc}

def evaluate_regression(model, X_val, y_val):
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    
    return -rmse, {"rmse": rmse} # Optuna works with maximums so we return negative


def tune_hyperparameters(outer_train_df, feature_cols, target_col, model_family, task,
                         n_inner_splits, n_trials, date_col = "Date"):
    """Splits the TRAIN portion of the outer fold into inner walk-forward folds to perform tuning with Optuna.
    It does not touch the outer test set in any way—
    preventing the tuning process itself from becoming a source of leakage."""
    inner_folds = generate_walk_forward_splits(outer_train_df, n_inner_splits, date_col)
    
    if len(inner_folds) < 2:
        print(f"WARNING : Number of inner folds are insufficient, continuing with base parameters.")
        return {}
    
    def objective(trial):
        params = (
            suggest_xgb_params(trial, task) if model_family == "xgboost"
            else suggest_lighgbm_params(trial, task)
        )
        fold_scores = []
        for inner_train, inner_val in inner_folds:
            X_train, y_train = inner_train[feature_cols], inner_train[target_col]
            X_val, y_val = inner_val[feature_cols], inner_val[target_col]
            
            model = build_model(model_family, task, params)
            model.fit(X_train, y_train)
            
            score, _ = (
                evaluate_classification(model, X_val, y_val) if task == "classification" else
                evaluate_regression(model, X_val, y_val)
            )
            fold_scores.append(score)
        
        return float(np.mean(fold_scores))
    
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction= "maximize", sampler= sampler)
    study.optimize(objective, n_trials= n_trials, show_progress_bar= True)
    
    return study.best_params

def run_nested_walk_forward(df, feature_cols, target_col, model_family, task, n_outer_splits, n_inner_splits,
                            n_trials, date_col = "Date"):
    """Full nested walk-forward: Performs separate tuning for each outer fold, retrains on the 
    ENTIRE outer train set using the best parameters, and evaluates on the outer test set.
    Optimal parameters may vary per fold—this is expected, 
    as shifts in optimal settings over time (concept drift) are completely natural."""
    
    outer_folds = generate_walk_forward_splits(df, n_outer_splits, date_col)
    
    all_metrics = []
    all_best_params = []
    
    for i, (outer_train, outer_test) in enumerate(outer_folds):
        print(f"Outer Fold {i + 1} / {len(outer_folds)} ")
        print(f"Tuning : {outer_train[date_col].min().date()} -> {outer_train[date_col].max().date()}")
        
        best_params = tune_hyperparameters(
            outer_train, feature_cols, target_col, model_family, task, n_inner_splits, n_trials,date_col
        )
        all_best_params.append(best_params)
        
        final_model = build_model(model_family, task, {**best_params, "random_state": 42})
        X_train, y_train = outer_train[feature_cols], outer_train[target_col]
        X_test, y_test = outer_test[feature_cols], outer_test[target_col]
        final_model.fit(X_train, y_train)
        
        _, metrics = (
            evaluate_classification(final_model, X_test, y_test) if task == "classification"
            else evaluate_regression(final_model, X_test, y_test)
        )
        
        print(f"Outer Test Result : {metrics}")
        all_metrics.append(metrics)
    
    return all_metrics, all_best_params


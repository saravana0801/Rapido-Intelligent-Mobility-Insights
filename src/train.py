import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score, confusion_matrix,
                             mean_absolute_error, precision_score, recall_score)
from sklearn.preprocessing import OrdinalEncoder

from src import data_loader, preprocessing, feature_engineering

def train_models():
    print("Preparing data...")
    dfs = data_loader.load_data()
    merged = data_loader.merge_data(dfs)
    cleaned = preprocessing.clean_data(merged)
    df = feature_engineering.create_features(cleaned)

    # ── Feature set (now includes customer & driver ratings) ─────────────────
    features = [
        'city', 'hour_of_day', 'day_of_week', 'vehicle_type', 'ride_distance_km',
        'estimated_ride_time_min', 'traffic_level', 'weather_condition',
        'surge_multiplier', 'Fare_per_KM', 'Fare_per_Min', 'Rush_Hour_Flag',
        'Long_Distance_Flag', 'Driver_Reliability_Score', 'Customer_Loyalty_Score',
        'avg_customer_rating', 'avg_driver_rating',
    ]

    cat_cols = ['city', 'day_of_week', 'vehicle_type', 'traffic_level', 'weather_condition']

    # Fill any missing rating values with median before encoding
    for col in ['avg_customer_rating', 'avg_driver_rating']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Drop features not present in the dataframe
    features = [f for f in features if f in df.columns]

    X = df[features]
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    y_outcome     = df['booking_status']
    y_fare        = df['booking_value']
    y_cust_cancel = df['customer_cancel_flag']
    y_driver_delay = df['driver_delay_flag']

    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/encoder.pkl')

    metadata = {
        "trained_at":  datetime.now().isoformat(timespec='seconds'),
        "train_rows":  int(len(df) * 0.8),
        "test_rows":   int(len(df) * 0.2),
        "total_rows":  len(df),
        "features":    features,
        "tuning":      "GridSearchCV",
        "models":      {}
    }

    # ── 1. Fare Prediction Model (Regression) ─────────────────────────────────
    print("Training Fare Prediction Model (GridSearchCV)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fare, test_size=0.2, random_state=42)

    reg_base = HistGradientBoostingRegressor(
        categorical_features=cat_indices, random_state=42)
    param_grid_reg = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_iter':      [100, 200],
    }
    reg = GridSearchCV(reg_base, param_grid_reg, cv=3, scoring='r2', n_jobs=-1)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))
    rmse_pct = rmse / (y_test.mean() + 1e-9) * 100
    print(f"  Best params : {reg.best_params_}")
    print(f"  RMSE ₹{rmse:.2f} ({rmse_pct:.1f}% of mean)  MAE ₹{mae:.2f}  R² {r2:.4f}")
    joblib.dump(reg.best_estimator_, 'models/fare_predictor.pkl')
    metadata["models"]["fare_predictor"] = {
        "type":        "regression",
        "target":      "booking_value",
        "best_params": reg.best_params_,
        "metrics":     {"rmse": round(rmse, 4), "mae": round(mae, 4),
                        "r2": round(r2, 4), "rmse_pct_of_mean": round(rmse_pct, 2)},
        "benchmark":   {"rmse_pct_target": "≤ 10%",
                        "pass": int(rmse_pct <= 10)}
    }

    # ── 2. Ride Outcome Model (Multi-Class) ───────────────────────────────────
    print("Training Ride Outcome Model (GridSearchCV)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_outcome, test_size=0.2, random_state=42)

    clf1_base = HistGradientBoostingClassifier(
        categorical_features=cat_indices, random_state=42)
    param_grid_clf = {
        'learning_rate': [0.05, 0.1],
        'max_iter':      [100, 200],
    }
    clf1 = GridSearchCV(clf1_base, param_grid_clf, cv=3,
                        scoring='f1_macro', n_jobs=-1)
    clf1.fit(X_train, y_train)
    preds = clf1.predict(X_test)
    acc1  = float(accuracy_score(y_test, preds))
    f1m1  = float(f1_score(y_test, preds, average='macro'))
    cm1   = confusion_matrix(y_test, preds).tolist()
    print(f"  Best params : {clf1.best_params_}")
    print(f"  Accuracy {acc1*100:.1f}%  F1-macro {f1m1:.4f}")
    joblib.dump(clf1.best_estimator_, 'models/outcome_predictor.pkl')
    metadata["models"]["outcome_predictor"] = {
        "type":        "multiclass",
        "target":      "booking_status",
        "classes":     sorted(y_test.unique().tolist()),
        "best_params": clf1.best_params_,
        "metrics":     {"accuracy": round(acc1, 4), "f1_macro": round(f1m1, 4)},
        "confusion_matrix": cm1,
        "benchmark":   {"accuracy_target": "85–90%",
                        "pass": int(0.85 <= acc1 <= 0.90)}
    }

    # ── 3. Customer Cancel Risk Model (Binary) ────────────────────────────────
    print("Training Customer Cancel Risk Model (GridSearchCV)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cust_cancel, test_size=0.2, random_state=42)

    clf2_base = HistGradientBoostingClassifier(
        categorical_features=cat_indices, random_state=42)
    clf2 = GridSearchCV(clf2_base, param_grid_clf, cv=3,
                        scoring='f1', n_jobs=-1)
    clf2.fit(X_train, y_train)
    preds  = clf2.predict(X_test)
    proba2 = clf2.predict_proba(X_test)[:, 1]
    acc2   = float(accuracy_score(y_test, preds))
    f1b2   = float(f1_score(y_test, preds, zero_division=0))
    prec2  = float(precision_score(y_test, preds, zero_division=0))
    rec2   = float(recall_score(y_test, preds, zero_division=0))
    auc2   = float(roc_auc_score(y_test, proba2)) if len(np.unique(y_test)) == 2 else None
    print(f"  Best params : {clf2.best_params_}")
    print(f"  Accuracy {acc2*100:.1f}%  F1 {f1b2:.4f}  AUC {auc2:.4f}")
    joblib.dump(clf2.best_estimator_, 'models/cust_cancel_predictor.pkl')
    metadata["models"]["cust_cancel_predictor"] = {
        "type":        "binary",
        "target":      "customer_cancel_flag",
        "best_params": clf2.best_params_,
        "metrics":     {"accuracy": round(acc2, 4), "f1": round(f1b2, 4),
                        "precision": round(prec2, 4), "recall": round(rec2, 4),
                        "auc": round(auc2, 4) if auc2 else None},
        "benchmark":   {"accuracy_target": "85–90%",
                        "pass": int(0.85 <= acc2 <= 0.90)}
    }

    # ── 4. Driver Delay Prediction Model (Binary) ─────────────────────────────
    print("Training Driver Delay Model (GridSearchCV)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_driver_delay, test_size=0.2, random_state=42)

    clf3_base = HistGradientBoostingClassifier(
        categorical_features=cat_indices, random_state=42)
    clf3 = GridSearchCV(clf3_base, param_grid_clf, cv=3,
                        scoring='f1', n_jobs=-1)
    clf3.fit(X_train, y_train)
    preds  = clf3.predict(X_test)
    proba3 = clf3.predict_proba(X_test)[:, 1]
    acc3   = float(accuracy_score(y_test, preds))
    f1b3   = float(f1_score(y_test, preds, zero_division=0))
    prec3  = float(precision_score(y_test, preds, zero_division=0))
    rec3   = float(recall_score(y_test, preds, zero_division=0))
    auc3   = float(roc_auc_score(y_test, proba3)) if len(np.unique(y_test)) == 2 else None
    print(f"  Best params : {clf3.best_params_}")
    print(f"  Accuracy {acc3*100:.1f}%  F1 {f1b3:.4f}  AUC {auc3:.4f}")
    joblib.dump(clf3.best_estimator_, 'models/driver_delay_predictor.pkl')
    metadata["models"]["driver_delay_predictor"] = {
        "type":        "binary",
        "target":      "driver_delay_flag",
        "best_params": clf3.best_params_,
        "metrics":     {"accuracy": round(acc3, 4), "f1": round(f1b3, 4),
                        "precision": round(prec3, 4), "recall": round(rec3, 4),
                        "auc": round(auc3, 4) if auc3 else None},
        "benchmark":   {"accuracy_target": "85–90%",
                        "pass": int(0.85 <= acc3 <= 0.90)}
    }

    # ── 5. ETA Predictor (UC2 — Improve ETA Accuracy) ─────────────────────────
    print("Training ETA Predictor — UC2 Improve ETA Accuracy (GridSearchCV)...")
    # Target: actual_ride_time_min (filled with estimated where missing)
    df['actual_ride_time_min'] = df['actual_ride_time_min'].fillna(df['estimated_ride_time_min'])
    y_eta = df['actual_ride_time_min']

    eta_features = [
        'city', 'hour_of_day', 'day_of_week', 'vehicle_type', 'ride_distance_km',
        'estimated_ride_time_min', 'traffic_level', 'weather_condition',
        'surge_multiplier', 'Rush_Hour_Flag', 'Long_Distance_Flag',
    ]
    eta_features = [f for f in eta_features if f in df.columns]
    X_eta = df[eta_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_eta, y_eta, test_size=0.2, random_state=42)

    eta_cat_idx = [eta_features.index(c) for c in
                   ['city', 'day_of_week', 'vehicle_type', 'traffic_level', 'weather_condition']
                   if c in eta_features]

    eta_base = HistGradientBoostingRegressor(
        categorical_features=eta_cat_idx, random_state=42)
    param_grid_eta = {'learning_rate': [0.05, 0.1, 0.2], 'max_iter': [100, 200]}
    eta_cv = GridSearchCV(eta_base, param_grid_eta, cv=3, scoring='r2', n_jobs=-1)
    eta_cv.fit(X_train, y_train)
    preds_eta = eta_cv.predict(X_test)
    rmse_eta  = float(np.sqrt(mean_squared_error(y_test, preds_eta)))
    mae_eta   = float(mean_absolute_error(y_test, preds_eta))
    r2_eta    = float(r2_score(y_test, preds_eta))
    eta_rmse_pct = rmse_eta / (y_test.mean() + 1e-9) * 100
    print(f"  Best params : {eta_cv.best_params_}")
    print(f"  RMSE {rmse_eta:.2f} min ({eta_rmse_pct:.1f}%)  MAE {mae_eta:.2f} min  R² {r2_eta:.4f}")
    joblib.dump(eta_cv.best_estimator_, 'models/eta_predictor.pkl')
    # Save ETA encoder mapping (reuse main encoder — same cat cols)
    joblib.dump(eta_features, 'models/eta_features.pkl')
    metadata["models"]["eta_predictor"] = {
        "type":        "regression",
        "target":      "actual_ride_time_min",
        "use_case":    "UC2 — Improve ETA Accuracy",
        "best_params": eta_cv.best_params_,
        "metrics":     {"rmse_min": round(rmse_eta, 4), "mae_min": round(mae_eta, 4),
                        "r2": round(r2_eta, 4), "rmse_pct_of_mean": round(eta_rmse_pct, 2)},
        "benchmark":   {"rmse_pct_target": "≤ 10%", "pass": int(eta_rmse_pct <= 10)}
    }

    # ── 6. Demand Level Predictor (UC3 — Dynamic Pricing / Demand Prediction) ─
    print("Training Demand Level Predictor — UC3 Dynamic Pricing (GridSearchCV)...")
    import os as _os
    demand_path = _os.path.join('Rapido_dataset', 'location_demand.csv')
    demand_df = pd.read_csv(demand_path)

    demand_features = ['city', 'pickup_location', 'hour_of_day', 'vehicle_type',
                       'avg_wait_time_min', 'avg_surge_multiplier']
    demand_cat_cols  = ['city', 'pickup_location', 'vehicle_type']

    # Fill any missing values
    for col in demand_df.select_dtypes(include='float').columns:
        demand_df[col] = demand_df[col].fillna(demand_df[col].median())

    demand_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    demand_df[demand_cat_cols] = demand_encoder.fit_transform(demand_df[demand_cat_cols])
    joblib.dump(demand_encoder, 'models/demand_encoder.pkl')
    joblib.dump(demand_features, 'models/demand_features.pkl')

    X_dem = demand_df[demand_features]
    y_dem = demand_df['demand_level']          # Low / Medium / High
    dem_cat_idx = [demand_features.index(c) for c in demand_cat_cols if c in demand_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_dem, y_dem, test_size=0.2, random_state=42)

    dem_base = HistGradientBoostingClassifier(
        categorical_features=dem_cat_idx, random_state=42)
    dem_cv = GridSearchCV(dem_base, param_grid_clf, cv=3, scoring='f1_macro', n_jobs=-1)
    dem_cv.fit(X_train, y_train)
    preds_dem = dem_cv.predict(X_test)
    acc_dem  = float(accuracy_score(y_test, preds_dem))
    f1_dem   = float(f1_score(y_test, preds_dem, average='macro', zero_division=0))
    classes_dem = sorted(y_test.unique().tolist())
    cm_dem   = confusion_matrix(y_test, preds_dem, labels=classes_dem).tolist()
    print(f"  Best params : {dem_cv.best_params_}")
    print(f"  Accuracy {acc_dem*100:.1f}%  F1-macro {f1_dem:.4f}")
    joblib.dump(dem_cv.best_estimator_, 'models/demand_predictor.pkl')
    metadata["models"]["demand_predictor"] = {
        "type":        "multiclass",
        "target":      "demand_level",
        "use_case":    "UC3 — Dynamic Pricing / Demand Prediction",
        "classes":     classes_dem,
        "best_params": dem_cv.best_params_,
        "metrics":     {"accuracy": round(acc_dem, 4), "f1_macro": round(f1_dem, 4)},
        "confusion_matrix": cm_dem,
        "benchmark":   {"accuracy_target": "85–90%", "pass": int(0.85 <= acc_dem <= 0.90)}
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ All 6 models trained with GridSearchCV and metadata saved to 'models/' directory.")
    print("   Features used:", features)

if __name__ == "__main__":
    train_models()

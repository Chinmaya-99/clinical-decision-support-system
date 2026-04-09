

import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────
CSV_PATH     = '/home/chinmaya/Desktop/coding/aiagent/health_care/data sets/kidney/kidney_disease.csv'   # ← update if needed
MODEL_OUT    = 'models/kidney_model.pkl'
FEATURES_OUT = 'models/kidney_features.pkl'

FEATURES = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine',
    'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
    'appetite', 'pedal_edema', 'anemia'
]


# ── Step 1: Load & Clean ──────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")

    # ── THE KEY FIX ──────────────────────────────────────────────────────────
    # Every string value in this CSV is wrapped in single quotes, e.g. 'ckd',
    # 'yes', 'normal'. Strip them ALL before doing any mapping. Without this,
    # map({'yes': 1}) silently returns NaN because the actual value is "'yes'".
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip()
    # ─────────────────────────────────────────────────────────────────────────

    # Rename short UCI column names to descriptive names
    df.rename(columns={
        'bp':  'blood_pressure',      'sg':  'specific_gravity',
        'al':  'albumin',             'su':  'sugar',
        'rbc': 'red_blood_cells',     'pc':  'pus_cell',
        'pcc': 'pus_cell_clumps',     'ba':  'bacteria',
        'bgr': 'blood_glucose_random','bu':  'blood_urea',
        'sc':  'serum_creatinine',    'sod': 'sodium',
        'pot': 'potassium',           'hemo':'haemoglobin',
        'pcv': 'packed_cell_volume',  'wc':  'white_blood_cell_count',
        'rc':  'red_blood_cell_count','htn': 'hypertension',
        'dm':  'diabetes_mellitus',   'cad': 'coronary_artery_disease',
        'appet':'appetite',           'pe':  'pedal_edema',
        'ane': 'anemia',
    }, inplace=True)

    # Target: 1 = CKD (positive class), 0 = no CKD
    df['target'] = (df['classification'] == 'ckd').astype(int)
    print(f"Class balance: {df['target'].value_counts().to_dict()}")

    # Encode binary yes/no columns
    for col in ['hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                'pedal_edema', 'anemia']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Encode other categoricals
    df['red_blood_cells'] = df['red_blood_cells'].map({'normal': 1, 'abnormal': 0})
    df['pus_cell']        = df['pus_cell'].map({'normal': 1, 'abnormal': 0})
    df['pus_cell_clumps'] = df['pus_cell_clumps'].map({'present': 1, 'notpresent': 0})
    df['bacteria']        = df['bacteria'].map({'present': 1, 'notpresent': 0})
    df['appetite']        = df['appetite'].map({'good': 1, 'poor': 0})

    # Some columns stored as strings — force numeric
    for col in ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values with class-aware medians for numeric cols,
    # then safety-net fill any remaining NaN with global median
    numeric_cols = df[FEATURES].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        for cls in [0, 1]:
            mask = df['target'] == cls
            median_val = df.loc[mask, col].median()
            df.loc[mask & df[col].isna(), col] = median_val
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))

    remaining_nan = df[FEATURES].isnull().sum().sum()
    print(f"Remaining NaN after imputation: {remaining_nan}")
    return df


# ── Step 2: Train ─────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                          max_depth=4, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_pipe, best_auc = None, None, 0

    print('\n── Cross-validation (5-fold ROC-AUC) ──')
    for name, clf in candidates.items():
        pipe   = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        mean_auc = scores.mean()
        print(f'  {name:<25}  AUC = {mean_auc:.4f} ± {scores.std():.4f}')
        if mean_auc > best_auc:
            best_auc, best_name, best_pipe = mean_auc, name, pipe

    print(f'\nBest model: {best_name} (AUC = {best_auc:.4f})')

    # Calibrate for reliable probability scores
    model = CalibratedClassifierCV(best_pipe, method='isotonic', cv=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f'\nTest ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}')
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))

    return model


# ── Step 3: Sanity-check predictions ─────────────────────────
def sanity_check(model):
    test_cases = {
        'Severely sick (expect High)': {
            'age': 65, 'blood_pressure': 90, 'specific_gravity': 1.005,
            'albumin': 4, 'sugar': 3, 'red_blood_cells': 0, 'pus_cell': 0,
            'pus_cell_clumps': 1, 'bacteria': 1, 'blood_glucose_random': 180,
            'blood_urea': 120, 'serum_creatinine': 9.0, 'sodium': 120,
            'potassium': 6.5, 'haemoglobin': 7.0, 'packed_cell_volume': 24,
            'white_blood_cell_count': 15000, 'red_blood_cell_count': 2.5,
            'hypertension': 1, 'diabetes_mellitus': 1, 'coronary_artery_disease': 1,
            'appetite': 0, 'pedal_edema': 1, 'anemia': 1,
        },
        'Healthy (expect Low)': {
            'age': 30, 'blood_pressure': 70, 'specific_gravity': 1.020,
            'albumin': 0, 'sugar': 0, 'red_blood_cells': 1, 'pus_cell': 1,
            'pus_cell_clumps': 0, 'bacteria': 0, 'blood_glucose_random': 90,
            'blood_urea': 20, 'serum_creatinine': 0.9, 'sodium': 140,
            'potassium': 4.0, 'haemoglobin': 14.5, 'packed_cell_volume': 44,
            'white_blood_cell_count': 7000, 'red_blood_cell_count': 4.8,
            'hypertension': 0, 'diabetes_mellitus': 0, 'coronary_artery_disease': 0,
            'appetite': 1, 'pedal_edema': 0, 'anemia': 0,
        },
    }

    print('\n── Sanity checks ──')
    for label, patient in test_cases.items():
        X_p  = pd.DataFrame([patient])[FEATURES]
        prob = float(model.predict_proba(X_p)[0][1])
        risk = 'High' if prob >= 0.70 else 'Moderate' if prob >= 0.40 else 'Low'
        print(f'  {label}')
        print(f'    → prob={prob:.4f}  risk={risk}\n')


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Kidney Disease Model Training ===\n')

    df    = load_and_clean(CSV_PATH)
    model = train(df)
    sanity_check(model)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model,    MODEL_OUT)
    joblib.dump(FEATURES, FEATURES_OUT)
    print(f'\n✓ Saved: {MODEL_OUT}')
    print(f'✓ Saved: {FEATURES_OUT}')
    print('\nDone. Replace your old kidney_model.pkl with this one.')

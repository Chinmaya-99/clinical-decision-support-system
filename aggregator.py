import os
import numpy as np
import pandas as pd
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Model Registry ─────────────────────────────────────────
# Add new diseases here as you build more models.
# Each entry tells the aggregator which features that model needs.

MODEL_REGISTRY = {
    'Diabetes': {
        'path': 'models/diabetes_model.pkl',
        'features': [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ],
        'feature_defaults': {
            'Pregnancies': 0, 'Glucose': 100, 'BloodPressure': 72,
            'SkinThickness': 20, 'Insulin': 80, 'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.5, 'Age': 35
        }
    },
    'Heart Disease': {
        'path': 'models/heart_disease_model.pkl',
        'features': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
            'age_thalach_ratio', 'oldpeak_cat', 'age_hr_risk'
        ],
        'feature_defaults': {
            'age': 40, 'sex': 0, 'cp': 0, 'trestbps': 120, 'chol': 200,
            'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 1,
            'age_thalach_ratio': 0.27, 'oldpeak_cat': 0, 'age_hr_risk': 0
        }
    },
    'Kidney Disease': {
        'path': 'models/kidney_model.pkl',
        'features': [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
            'blood_glucose_random', 'blood_urea', 'serum_creatinine',
            'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
            'white_blood_cell_count', 'red_blood_cell_count',
            'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
            'appetite', 'pedal_edema', 'anemia'
        ],
        'feature_defaults': {
            'age': 40,
            'blood_pressure': 80,
            'specific_gravity': 1.020,
            'albumin': 0,
            'sugar': 0,
            'red_blood_cells': 1,         # 1 = normal, 0 = abnormal
            'pus_cell': 1,                # 1 = normal, 0 = abnormal
            'pus_cell_clumps': 0,         # 0 = notpresent, 1 = present
            'bacteria': 0,                # 0 = notpresent, 1 = present
            'blood_glucose_random': 100,
            'blood_urea': 25,
            'serum_creatinine': 1.0,
            'sodium': 140,
            'potassium': 4.5,
            'haemoglobin': 13.5,
            'packed_cell_volume': 44,
            'white_blood_cell_count': 7000,
            'red_blood_cell_count': 4.5,
            'hypertension': 0,            # 0 = no, 1 = yes
            'diabetes_mellitus': 0,
            'coronary_artery_disease': 0,
            'appetite': 1,                # 1 = good, 0 = poor
            'pedal_edema': 0,
            'anemia': 0
        }
    }
}


# ── 2. Feature Engineering ────────────────────────────────────

def engineer_heart_features(data: dict) -> dict:
    data = data.copy()
    data['age_thalach_ratio'] = data.get('age', 40) / (data.get('thalach', 150) + 1)
    oldpeak = data.get('oldpeak', 0)
    if oldpeak <= 0.5:
        data['oldpeak_cat'] = 0
    elif oldpeak <= 2.0:
        data['oldpeak_cat'] = 1
    else:
        data['oldpeak_cat'] = 2
    data['age_hr_risk'] = int(data.get('age', 40) > 55 and data.get('thalach', 150) < 140)
    return data


FEATURE_ENGINEERS = {
    'Heart Disease': engineer_heart_features,
    # Add more as you build models: 'Kidney Disease': engineer_kidney_features
}


# ── 3. Feature validation ─────────────────────────────────────
# FIX: was incorrectly defined inside ModelLoader with a 'self' parameter,
# causing predict_single to crash silently on every call.
# Now a plain standalone function — no 'self', called directly.

def validate_and_fill(patient_data: dict, disease: str) -> pd.DataFrame:
    """Build the feature DataFrame for a single disease model.
    Uses patient values where provided, falls back to clinical defaults."""
    config   = MODEL_REGISTRY[disease]
    features = config['features']
    defaults = config['feature_defaults']

    # Apply feature engineering if needed (e.g. heart disease derived features)
    engineer = FEATURE_ENGINEERS.get(disease)
    if engineer:
        patient_data = engineer(patient_data)

    # Build row: use patient value if present and not None, else default
    row = {}
    for feat in features:
        if feat in patient_data and patient_data[feat] is not None:
            row[feat] = patient_data[feat]
        else:
            row[feat] = defaults.get(feat, 0)

    return pd.DataFrame([row])[features]


# ── 4. Model Loader ───────────────────────────────────────────

class ModelLoader:
    """Loads and caches all available models at startup."""

    def __init__(self):
        self.models = {}
        self._load_all()

    def _load_all(self):
        print("Loading disease models...")
        for disease, config in MODEL_REGISTRY.items():
            path = os.path.join(BASE_DIR, config['path'])
            print("Looking for model at:", path)
            if os.path.exists(path):
                try:
                    self.models[disease] = joblib.load(path)
                    print(f"  ✓ {disease} model loaded")
                except Exception as e:
                    print(f"  ✗ {disease} failed to load: {e}")
            else:
                print(f"  - {disease} model not found (skipping)")

        if not self.models:
            raise RuntimeError("No models found! Train at least one model first.")

        print(f"\nReady: {len(self.models)} model(s) active\n")

    def available_diseases(self):
        return list(self.models.keys())


# ── 5. Single-model prediction ────────────────────────────────
# FIX: was calling ModelLoader.validate_and_fill(patient_data, disease)
# which passed patient_data as 'self' — now calls the standalone function.

def predict_single(model, patient_data: dict, disease: str) -> dict:
    """Run one model and return a standardized result dict."""
    try:
        X     = validate_and_fill(patient_data, disease)   # FIX: direct call
        proba = model.predict_proba(X)[0]

        # Always take the positive class probability (index 1)
        prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        prob = round(prob, 4)

        if prob >= 0.70:
            risk_level = 'High'
        elif prob >= 0.40:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        return {
            'disease':     disease,
            'probability': prob,
            'confidence':  f"{prob * 100:.1f}%",
            'risk_level':  risk_level,
            'status':      'ok'
        }
    except Exception as e:
        # FIX: print full traceback so errors are visible in terminal
        traceback.print_exc()
        return {
            'disease':     disease,
            'probability': 0.0,
            'confidence':  'N/A',
            'risk_level':  'Unknown',
            'status':      f'error: {e}'
        }


# ── 6. Aggregator ─────────────────────────────────────────────

class ClinicalDecisionAggregator:
    """
    The central brain of the CDSS.
    Runs patient data through all loaded models,
    then returns a ranked, filtered prediction report.
    """

    def __init__(self, threshold: float = 0.30):
        """
        threshold: minimum probability to include in the flagged output.
                   Diseases below this are treated as low risk.
                   Default 0.30 keeps moderate and high risks visible.
        """
        self.loader    = ModelLoader()
        self.threshold = threshold

    def predict(self, patient_data: dict, top_n: int = 3) -> dict:
        """
        Main entry point.

        Args:
            patient_data : flat dict of all available patient features
            top_n        : how many diseases to return in the ranked list

        Returns:
            Full prediction report dict
        """
        raw_results = []

        for disease, model in self.loader.models.items():
            result = predict_single(model, patient_data, disease)
            raw_results.append(result)

        # Sort by probability descending
        raw_results.sort(key=lambda x: x['probability'], reverse=True)

        # FIX: was `flagged = raw_results` — threshold was never applied.
        # Now correctly filters to only conditions at or above the threshold.
        flagged  = [r for r in raw_results if r['probability'] >= self.threshold]
        low_risk = [r for r in raw_results if r['probability'] < self.threshold]

        top_predictions = flagged[:top_n]

        report = {
            'timestamp':       datetime.now().isoformat(),
            'models_run':      len(raw_results),
            'top_predictions': top_predictions,
            'all_results':     raw_results,
            'low_risk_count':  len(low_risk),
            'disclaimer': (
                "⚠ DISCLAIMER: This system is for educational and research purposes only. "
                "These predictions are NOT a medical diagnosis. "
                "Always consult a qualified healthcare professional."
            )
        }

        return report

    def format_report(self, report: dict) -> str:
        """Pretty-print the prediction report to the console."""
        lines = [
            "\n" + "="*55,
            "   CLINICAL DECISION SUPPORT — PREDICTION REPORT",
            "="*55,
            f"  Timestamp : {report['timestamp'][:19]}",
            f"  Models run: {report['models_run']}",
            "-"*55,
        ]

        if not report['top_predictions']:
            lines.append("  No significant risk detected above threshold.")
        else:
            lines.append("  TOP PREDICTED CONDITIONS:\n")
            for i, pred in enumerate(report['top_predictions'], 1):
                bar_len = int(pred['probability'] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lines += [
                    f"  {i}. {pred['disease']}",
                    f"     Risk     : {pred['risk_level']}",
                    f"     Confidence: {pred['confidence']}",
                    f"     [{bar}]",
                    ""
                ]

        lines += [
            "-"*55,
            f"  Low-risk conditions: {report['low_risk_count']}",
            "="*55,
            f"\n  {report['disclaimer']}\n",
        ]

        return "\n".join(lines)


# ── 7. Test helper ────────────────────────────────────────────

def simulate_models_for_testing():
    """
    Creates dummy sklearn pipelines so you can test the
    aggregator before all real models are trained.
    Run this once, then replace with real trained models.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    os.makedirs('models', exist_ok=True)

    for disease, config in MODEL_REGISTRY.items():
        path = config['path']
        if os.path.exists(path):
            print(f"  Skipping {disease} (model already exists)")
            continue

        n_features = len(config['features'])
        X, y = make_classification(
            n_samples=300, n_features=n_features,
            n_informative=min(5, n_features), random_state=42
        )

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500))
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, path)
        print(f"  ✓ Simulated {disease} model saved to {path}")

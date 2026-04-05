# predict_api.py
# ============================================================
# Clean API layer — import this into your Streamlit app
# ============================================================

from aggregator import ClinicalDecisionAggregator

# Load once at import time (cached by Python's module system)
_aggregator = None

def get_aggregator():
    global _aggregator
    if _aggregator is None:
        _aggregator = ClinicalDecisionAggregator(threshold=0.30)
    return _aggregator


def run_prediction(patient_data: dict, top_n: int = 3) -> dict:
    """
    Main function your Streamlit app calls.
    
    Args:
        patient_data : dict of patient values from form inputs
        top_n        : number of top diseases to return

    Returns:
        {
          'top_predictions': [
              {'disease': str, 'probability': float,
               'confidence': str, 'risk_level': str}, ...
          ],
          'all_results': [...],
          'disclaimer':  str,
          'models_run':  int
        }
    """
    agg = get_aggregator()
    return agg.predict(patient_data, top_n=top_n)

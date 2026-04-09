# app.py
# ============================================================
# Clinical Decision Support System — Streamlit Web App
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
from datetime import datetime
import json
import os

# Import your aggregator (must be in the same folder)
from predict_api import run_prediction, get_aggregator


# ── 1. Page Configuration ─────────────────────────────────────
# Must be the FIRST streamlit call in the file
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── 2. Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Risk badge colors */
    .badge-high     { background:#fee2e2; color:#991b1b; padding:3px 10px;
                      border-radius:12px; font-size:12px; font-weight:600; }
    .badge-moderate { background:#fef3c7; color:#92400e; padding:3px 10px;
                      border-radius:12px; font-size:12px; font-weight:600; }
    .badge-low      { background:#d1fae5; color:#065f46; padding:3px 10px;
                      border-radius:12px; font-size:12px; font-weight:600; }

    /* Disclaimer box */
    .disclaimer {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        font-size: 13px; color: #78350f; margin-top: 1rem;
    }

    /* Result card */
    .result-card {
        background: white; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 16px 20px;
        margin-bottom: 10px;
    }

    /* Sidebar model status */
    .model-ok  { color: #16a34a; font-size:13px; }
    .model-off { color: #9ca3af; font-size:13px; }

    /* Remove default streamlit padding */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── 3. Session State ──────────────────────────────────────────
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists values across those reruns —
# like a mini-database that lives as long as the browser tab is open.

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []   # list of past reports

if 'last_report' not in st.session_state:
    st.session_state.last_report = None        # most recent prediction


# ── 4. Load aggregator (cached — only runs once) ──────────────
@st.cache_resource
def load_aggregator():
    """
    @st.cache_resource means Streamlit runs this function exactly
    once and reuses the result on every rerun.
    Without it, models would reload from disk on every button click.
    """
    return get_aggregator()

aggregator = load_aggregator()


# ── 5. Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 CDSS")
    st.caption("Clinical Decision Support System")
    st.divider()

    # Show which models are loaded
    st.subheader("Model Status")
    all_diseases = ['Diabetes', 'Heart Disease', 'Kidney Disease', 'Anemia', 'Infection']
    loaded = aggregator.loader.available_diseases()

    for disease in all_diseases:
        if disease in loaded:
            st.markdown(f'<p class="model-ok">✓ {disease}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="model-off">○ {disease} (not loaded)</p>', unsafe_allow_html=True)

    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["Predict", "History", "About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(f"Models active: {len(loaded)} / {len(all_diseases)}")
    st.caption(f"Session: {datetime.now().strftime('%d %b %Y')}")


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
if page == "Predict":

    st.header("Patient Assessment")
    st.caption("Fill in available test results. Missing fields use safe clinical defaults.")

    # ── Input form split into tabs by category ────────────────
    tab_vitals, tab_blood, tab_cardiac, tab_kidney, tab_symptoms = st.tabs([
        "Vitals", "Blood Panel", "Cardiac / ECG", "Kidney / Urine", "Symptoms"
    ])

    patient = {}   # this dict grows as the user fills in each tab

    # ── Tab 1: Vitals ─────────────────────────────────────────
    with tab_vitals:
        st.subheader("Basic vitals")
        col1, col2, col3 = st.columns(3)

        with col1:
            patient['age'] = st.number_input("Age (years)", 1, 120, 40)
            patient['Age'] = patient['age']   # diabetes model uses 'Age' (capital)
            patient['sex'] = st.selectbox("Sex", ["Male", "Female"])
            patient['sex'] = 1 if patient['sex'] == "Male" else 0
            patient['gender'] = patient['sex']

        with col2:
            patient['BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0, step=0.1)
            patient['trestbps'] = st.number_input("Resting blood pressure (mmHg)", 80, 220, 120)
            patient['BloodPressure'] = patient['trestbps']

        with col3:
            patient['temperature'] = st.number_input("Body temperature (°C)", 35.0, 42.0, 37.0, step=0.1)
            patient['has_fatigue'] = int(st.checkbox("Fatigue present"))
            patient['has_chills'] = int(st.checkbox("Chills present"))

    # ── Tab 2: Blood Panel ────────────────────────────────────
    with tab_blood:
        st.subheader("Blood test results")
        col1, col2, col3 = st.columns(3)

        with col1:
            patient['Glucose'] = st.number_input("Glucose (mg/dL)", 50, 500, 100)
            patient['blood_glucose_random'] = patient['Glucose']
            patient['Insulin'] = st.number_input("Insulin (µU/mL)", 0, 900, 80)
            patient['Pregnancies'] = st.number_input("Pregnancies", 0, 20, 0)

        with col2:
            patient['chol'] = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            patient['fbs'] = int(st.checkbox("Fasting blood sugar > 120 mg/dL"))
            patient['SkinThickness'] = st.number_input("Skin thickness (mm)", 0, 100, 20)

        with col3:
            patient['hemoglobin'] = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 13.5, step=0.1)
            patient['haemoglobin'] = patient['hemoglobin']
            patient['DiabetesPedigreeFunction'] = st.number_input(
                "Diabetes pedigree function", 0.0, 2.5, 0.5, step=0.01
            )
            patient['wbc_count'] = st.number_input("WBC count (cells/µL)", 1000, 30000, 7000, step=100)

        st.subheader("CBC (Complete Blood Count)")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient['mch']  = st.number_input("MCH (pg)", 10.0, 50.0, 27.0, step=0.1)
        with col2:
            patient['mchc'] = st.number_input("MCHC (g/dL)", 20.0, 40.0, 33.0, step=0.1)
        with col3:
            patient['mcv']  = st.number_input("MCV (fL)", 50.0, 130.0, 85.0, step=0.1)

    # ── Tab 3: Cardiac / ECG ──────────────────────────────────
    with tab_cardiac:
        st.subheader("Cardiac indicators")
        col1, col2 = st.columns(2)

        with col1:
            cp_map = {
                "Typical angina (0)": 0,
                "Atypical angina (1)": 1,
                "Non-anginal pain (2)": 2,
                "Asymptomatic (3)": 3
            }
            patient['cp'] = cp_map[st.selectbox("Chest pain type", list(cp_map.keys()))]
            patient['thalach'] = st.number_input("Max heart rate achieved", 60, 220, 150)
            patient['exang'] = int(st.checkbox("Exercise-induced angina"))
            patient['oldpeak'] = st.number_input(
                "ST depression (oldpeak)", 0.0, 7.0, 0.0, step=0.1
            )

        with col2:
            slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
            patient['slope'] = slope_map[st.selectbox("ST slope", list(slope_map.keys()))]
            patient['ca']   = st.slider("Major vessels coloured by fluoroscopy", 0, 4, 0)
            thal_map = {"Normal (1)": 1, "Fixed defect (2)": 2, "Reversible defect (3)": 3}
            patient['thal'] = thal_map[st.selectbox("Thalassemia", list(thal_map.keys()))]
            restecg_map = {"Normal (0)": 0, "ST-T abnormality (1)": 1, "LVH (2)": 2}
            patient['restecg'] = restecg_map[st.selectbox("Resting ECG", list(restecg_map.keys()))]

    # ── Tab 4: Kidney / Urine ─────────────────────────────────
    with tab_kidney:
        st.subheader("Kidney & urine panel")
        col1, col2, col3 = st.columns(3)

        with col1:
            patient['blood_urea']        = st.number_input("Blood urea (mg/dL)", 5, 200, 30)
            patient['serum_creatinine']  = st.number_input("Serum creatinine (mg/dL)", 0.4, 15.0, 1.0, step=0.1)
            patient['sodium']            = st.number_input("Sodium (mEq/L)", 110, 160, 140)
            patient['potassium']         = st.number_input("Potassium (mEq/L)", 2.0, 7.0, 4.5, step=0.1)

        with col2:
            patient['albumin']           = st.slider("Albumin in urine (0–5)", 0, 5, 0)
            patient['sugar']             = st.slider("Sugar in urine (0–5)", 0, 5, 0)
            patient['specific_gravity']  = st.number_input("Specific gravity", 1.005, 1.030, 1.020, step=0.001)
            patient['packed_cell_volume']= st.number_input("Packed cell volume (%)", 10, 60, 44)

        with col3:
            patient['white_blood_cell_count'] = st.number_input("WBC in urine (cells/cumm)", 1000, 30000, 8000, step=100)
            patient['red_blood_cell_count']   = st.number_input("RBC count (millions/cumm)", 1.0, 8.0, 4.5, step=0.1)
            patient['blood_pressure']         = st.number_input("Diastolic BP (mmHg)", 50, 150, 80)

        st.subheader("Clinical flags")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient['hypertension']          = int(st.checkbox("Hypertension"))
            patient['diabetes_mellitus']     = int(st.checkbox("Diabetes mellitus"))
        with col2:
            patient['coronary_artery_disease']= int(st.checkbox("Coronary artery disease"))
            patient['pedal_edema']            = int(st.checkbox("Pedal edema"))
        with col3:
            patient['anemia']    = int(st.checkbox("Anemia (known)"))
            patient['appetite']  = int(st.selectbox("Appetite", ["Good (1)", "Poor (0)"]) == "Good (1)")
            patient['red_blood_cells'] = int(st.selectbox(
                "RBC in urine", ["Normal (1)", "Abnormal (0)"]) == "Normal (1)")
            patient['pus_cell']  = int(st.selectbox(
                "Pus cells", ["Normal (1)", "Abnormal (0)"]) == "Normal (1)")
            patient['pus_cell_clumps'] = int(st.checkbox("Pus cell clumps"))
            patient['bacteria']        = int(st.checkbox("Bacteria present"))

    # ── Tab 5: Symptoms ───────────────────────────────────────
    with tab_symptoms:
        st.subheader("Reported symptoms")
        col1, col2 = st.columns(2)

        with col1:
            patient['fever_days'] = st.slider("Fever duration (days)", 0, 14, 0)
            patient['crp']  = st.number_input("CRP level (mg/L)", 0.0, 200.0, 5.0, step=0.5)
            patient['esr']  = st.number_input("ESR (mm/hr)", 0, 150, 15)

        with col2:
            patient['neutrophils_pct']  = st.slider("Neutrophils (%)", 0, 100, 60)
            patient['lymphocytes_pct']  = st.slider("Lymphocytes (%)", 0, 100, 30)

    st.divider()

    # ── Predict Button ────────────────────────────────────────
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        predict_clicked = st.button(
            "Run Prediction", type="primary", use_container_width=True
        )
    with col_clear:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.last_report = None
            st.rerun()

    # ── Run prediction when button clicked ───────────────────
    if predict_clicked:
        with st.spinner("Running models..."):
            report = run_prediction(patient, top_n=5)
            st.session_state.last_report = report

            # Save to history with timestamp
            history_entry = {
                'time':    datetime.now().strftime('%H:%M:%S'),
                'report':  report,
                'patient': {k: v for k, v in patient.items()
                            if k in ['age', 'Glucose', 'BMI', 'hemoglobin']}
            }
            st.session_state.prediction_history.append(history_entry)

    # ── Display Results ───────────────────────────────────────
    if st.session_state.last_report:
        report = st.session_state.last_report

        st.subheader("Prediction Results")

        # ── Summary metric cards ──────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Models run",       report['models_run'])
        m2.metric("Conditions flagged", len(report['top_predictions']))
        top = report['top_predictions']
        if top:
            m3.metric(
                "Top condition",
                top[0]['disease'],
                f"{top[0]['confidence']} confidence"
            )

        st.divider()

        # ── Results + Chart side by side ──────────────────────
        res_col, chart_col = st.columns([1, 1])

        with res_col:
            st.subheader("Ranked conditions")

            if not report['top_predictions']:
                st.success("No conditions flagged above threshold. Low risk profile.")
            else:
                for pred in report['top_predictions']:
                    risk = pred['risk_level']
                    badge_class = {
                        'High': 'badge-high',
                        'Moderate': 'badge-moderate',
                        'Low': 'badge-low'
                    }.get(risk, 'badge-low')

                    prob_pct = pred['probability'] * 100

                    # Color the progress bar by risk
                    bar_color = {'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#22c55e'}.get(risk, '#6b7280')

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                            <span style="font-size:15px; font-weight:600">{pred['disease']}</span>
                            <span class="{badge_class}">{risk} Risk</span>
                        </div>
                        <div style="font-size:13px; color:#6b7280; margin-bottom:6px">
                            Confidence: <strong>{pred['confidence']}</strong>
                        </div>
                        <div style="background:#f3f4f6; border-radius:4px; height:8px; overflow:hidden">
                            <div style="width:{prob_pct:.1f}%; height:100%; background:{bar_color}; border-radius:4px"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Low-risk count
                    if report['low_risk_count'] > 0:
                        st.caption(f"{report['low_risk_count']} condition(s) below threshold — low risk.")

        with chart_col:
            st.subheader("Confidence overview")

            all_results = report['all_results']
            if all_results:
                df_chart = pd.DataFrame(all_results)
                df_chart = df_chart[df_chart['status'] == 'ok'].sort_values(
                    'probability', ascending=True
                )

                # Color bars by risk level
                color_map = {'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#22c55e'}
                colors = [color_map.get(r, '#94a3b8') for r in df_chart['risk_level']]

                # fig = go.Figure(go.Bar(
                #     x=df_chart['probability'] * 100,
                #     y=df_chart['disease'],
                #     orientation='h',
                #     marker_color=colors,
                #     text=[f"{p*100:.1f}%" for p in df_chart['probability']],
                #     textposition='outside'
                # ))
                # fig.update_layout(
                #     xaxis_title="Confidence (%)",
                #     xaxis=dict(range=[0, 110]),
                #     yaxis_title="",
                #     plot_bgcolor='rgba(0,0,0,0)',
                #     paper_bgcolor='rgba(0,0,0,0)',
                #     margin=dict(l=10, r=40, t=10, b=30),
                #     height=300,
                #     showlegend=False
                # )
                # st.plotly_chart(fig, use_container_width=True)


                # Prepare data
                df_chart['Confidence (%)'] = df_chart['probability'] * 100
                df_chart = df_chart[['disease', 'Confidence (%)']].set_index('disease')

                # Show chart
                st.bar_chart(df_chart)

        # ── Disclaimer ────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            <strong>Medical disclaimer:</strong> This system is for educational and 
            research purposes only. These predictions are <strong>NOT</strong> a medical 
            diagnosis. Always consult a qualified healthcare professional before making 
            any medical decisions.
        </div>
        """, unsafe_allow_html=True)

        # ── Export report ─────────────────────────────────────
        st.divider()
        export_data = {
            'timestamp': report['timestamp'],
            'predictions': report['top_predictions'],
            'disclaimer': report['disclaimer']
        }
        st.download_button(
            label="Download Report (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"cdss_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# ══════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "History":
    st.header("Prediction History")
    st.caption("All predictions made in this session")

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Go to the Predict page to get started.")
    else:
        for i, entry in enumerate(reversed(st.session_state.prediction_history)):
            with st.expander(f"Assessment at {entry['time']} — {len(entry['report']['top_predictions'])} condition(s) flagged"):
                top = entry['report']['top_predictions']
                if top:
                    for pred in top:
                        st.write(f"**{pred['disease']}** — {pred['confidence']} ({pred['risk_level']} risk)")
                else:
                    st.write("No conditions above threshold.")

                # Show the key patient values that were entered
                st.caption("Key values entered:")
                st.json(entry['patient'])


# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.header("About this system")

    st.markdown("""
    ### What this is
    A multi-model clinical decision support system built with Python and scikit-learn.
    It uses structured patient data — blood tests, vitals, ECG values, and symptoms —
    to predict the likelihood of several common conditions.

    ### How it works
    1. Patient data is entered through the form
    2. Each loaded disease model runs independently
    3. The aggregator ranks predictions by confidence
    4. Results are displayed with risk levels and confidence scores

    ### Disease models
    | Condition | Key features used |
    |-----------|-------------------|
    | Diabetes | Glucose, BMI, insulin, age, pedigree |
    | Heart Disease | ECG values, chest pain, cholesterol, ST depression |
    | Kidney Disease | Creatinine, urea, sodium, albumin, specific gravity |
    | Anemia | Hemoglobin, MCV, MCH, MCHC |
    | Infection | WBC, CRP, ESR, temperature, fever duration |

    ### Ethical notice
    This tool is built for **educational and research purposes only**.
    It should never replace the judgment of a qualified medical professional.
    All results carry an explicit disclaimer and must be interpreted in clinical context.
    """)

    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit · Plotly")
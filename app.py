import streamlit as st
import joblib
import numpy as np

model = joblib.load("final_model.pkl")

st.title("Huntington Disease Stage Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=30, max_value=80, value=55)
    cag_repeat = st.number_input("HTT CAG Repeat Length", min_value=35, max_value=80, value=57)
    chorea_score = st.number_input("Chorea Score", min_value=0.0, max_value=10.0, value=5.0, format="%.2f")
    brain_volume_loss = st.number_input("Brain Volume Loss", min_value=2.0, max_value=8.5, value=5.0, format="%.2f")

with col2:
    functional_capacity = st.number_input("Functional Capacity", min_value=0, max_value=100, value=50)
    htt_gene_expression = st.number_input("HTT Gene Expression Level", min_value=0.1, max_value=2.5, value=1.3, format="%.2f")
    protein_aggregation = st.number_input("Protein Aggregation Level", min_value=0.1, max_value=5.0, value=2.5, format="%.2f")

if st.button("Predict"):
    data = np.array([[age, cag_repeat, chorea_score, brain_volume_loss,
                      functional_capacity, htt_gene_expression, protein_aggregation]])

    prediction = model.predict(data)

    stage_map = {
        0: "Early Stage",
        1: "Middle Stage",
        2: "Advanced Stage"
    }

    st.success(f"Huntington Disease Stage: {stage_map[prediction[0]]}")

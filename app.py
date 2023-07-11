import numpy as np
import pandas as pd
import streamlit as st

from model import DATA_PATH, FEATURE_INFO_PATH, FeatureInfo, generate_models

# Title and description
st.image("https://drive.google.com/uc?export=view&id=1ijBEINIw0Uv6u9imv7geshpPhYU4L2yy")
st.title("Property Improvement Predictor")
st.write("Predict the resale value of properties based on renovations")

# Load feature info
FEATURE_INFO = FeatureInfo(FEATURE_INFO_PATH)

# Generate models at the start of the app
model, test_results = generate_models(DATA_PATH, FEATURE_INFO)

# Property features form
with st.expander("Property Features", expanded=True):
    st.header("Enter Property Details")

    input_data = {}
    # Generate input fields
    for feature in FEATURE_INFO.get_features():
        if FEATURE_INFO.get_input(feature) == "number_input":
            input_data[feature] = st.number_input(
                label=FEATURE_INFO.get_label(feature),
                min_value=FEATURE_INFO.get_min(feature),
                max_value=FEATURE_INFO.get_max(feature),
                value=2000,
            )

        elif FEATURE_INFO.get_input(feature) == "slider":
            input_data[feature] = st.slider(
                label=FEATURE_INFO.get_label(feature),
                min_value=FEATURE_INFO.get_min(feature),
                max_value=FEATURE_INFO.get_max(feature),
                value=1,
            )

        elif FEATURE_INFO.get_input(feature) == "selectbox":
            input_data[feature] = st.selectbox(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )

        elif FEATURE_INFO.get_input(feature) == "select_slider":
            input_data[feature] = st.select_slider(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )

        elif FEATURE_INFO.get_input(feature) == "radio":
            input_data[feature] = st.radio(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )


# Submit button
if st.button("Predict"):
    # Convert labels into values for categorical features
    for feature in FEATURE_INFO.get_categorical_features():
        input_data[feature] = FEATURE_INFO.map_label_to_value(
            feature, input_data[feature]
        )


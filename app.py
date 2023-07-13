import streamlit as st

from model import (
    DATA_PATH,
    FEATURE_INFO_PATH,
    FeatureInfo,
    find_price_increases,
    generate_models,
)

# Set page config
st.set_page_config(page_title="Renovation Recommender", layout="wide")

# Title and description
st.title("Property Improvement Predictor")
st.write("Suggests property improvements to increase resale value.")

# Set number of columns
NUMBER_OF_COLUMNS = 3

# Load feature info
FEATURE_INFO = FeatureInfo(FEATURE_INFO_PATH)

# Generate models at the start of the app
model, test_results = generate_models(DATA_PATH, FEATURE_INFO)


def feature_average(feature):
    """Returns the average value for a feature."""
    return (
        FEATURE_INFO.get_max(feature) - FEATURE_INFO.get_min(feature)
    ) // 2 + FEATURE_INFO.get_min(feature)


# Property features form
st.header("Look up renovation suggestions for your property:")
with st.form("input"):
    input_data = {}
    cols_form = st.columns(NUMBER_OF_COLUMNS)

    # Generate input fields
    for i, feature in enumerate(FEATURE_INFO.get_features()):
        # Cycle between columns
        col = cols_form[i % NUMBER_OF_COLUMNS]

        # Generate input fields based on feature input type
        # Number input
        if FEATURE_INFO.get_input(feature) == "number_input":
            input_data[feature] = col.number_input(
                label=FEATURE_INFO.get_label(feature),
                min_value=FEATURE_INFO.get_min(feature),
                max_value=FEATURE_INFO.get_max(feature),
                # Set default value to the average value
                value=feature_average(feature),
            )
        # Slider input
        elif FEATURE_INFO.get_input(feature) == "slider":
            input_data[feature] = col.slider(
                label=FEATURE_INFO.get_label(feature),
                min_value=FEATURE_INFO.get_min(feature),
                max_value=FEATURE_INFO.get_max(feature),
                value=feature_average(feature),
            )
        # Selectbox input
        elif FEATURE_INFO.get_input(feature) == "selectbox":
            input_data[feature] = col.selectbox(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )
        # Select slider input
        elif FEATURE_INFO.get_input(feature) == "select_slider":
            input_data[feature] = col.select_slider(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )
        # Radio input
        elif FEATURE_INFO.get_input(feature) == "radio":
            input_data[feature] = col.radio(
                label=FEATURE_INFO.get_label(feature),
                options=FEATURE_INFO.get_labels(feature),
            )

    submit = st.form_submit_button(type="primary", help="Find renovation suggestions")

if submit:
    with st.spinner("Predicting..."):
        # Copy input data to avoid modifying on user action
        data_submitted = input_data.copy()

        # Convert labels into values for categorical features
        data_pred = FEATURE_INFO.convert_labels_to_values(data_submitted)

        # Predict the price increase
        predictions = find_price_increases(model, data_pred, FEATURE_INFO)
        predictions = predictions.sort_values(
            "Price Increase", ascending=False
        ).reset_index(drop=True)

    cols_results = st.columns([0.4, 0.4, 0.2], gap="large")

    with cols_results[0]:
        st.subheader("Predicted Price Increase")
        st.bar_chart(
            predictions,
            x="Feature",
            y="Price Increase",
        )

    with cols_results[1]:
        # Format price increase column as currency
        st.subheader("  ")
        predictions["Price Increase"] = predictions["Price Increase"].map(
            lambda x: f"${round(x,-2):,.0f}"
        )
        # Display formatted prediction columns
        st.dataframe(predictions[["Renovation", "Price Increase"]], hide_index=True)

    with cols_results[2]:
        st.subheader("Model Metrics")
        # Test results
        # Output Number format = 0.00E+00
        st.metric(
            label="R Squared Score",
            value=f"{test_results['r2']:.2e}",
        )
        st.metric(
            label="Mean Square Error",
            value=f"{test_results['mse']:.2e}",
        )
        st.metric(
            label="Mean Absolute Error",
            value=f"{test_results['mae']:.2e}",
        )

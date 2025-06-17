import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# Suppress openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Page Configuration
st.set_page_config(
    page_title="Self-Healing Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Description
st.title("Self-Healing Concrete Strength Predictor")
st.markdown(
    """
    This application predicts the compressive strength of self-healing concrete based on various mix parameters.
    Upload your data or use the default dataset to train the model and make predictions.
"""
)


# --- Helper Functions ---
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            # Try to find the data file in different possible locations
            possible_paths = [
                "data/concrete_mix_data.xlsx",
                "concrete_mix_data.xlsx",
                os.path.join(os.path.dirname(__file__), "data", "concrete_mix_data.xlsx"),
                os.path.join(os.path.dirname(__file__), "concrete_mix_data.xlsx"),
                "/mount/src/strength_predictor/data/concrete_mix_data.xlsx",  # Streamlit Cloud path
                "/mount/src/strength_predictor/concrete_mix_data.xlsx"  # Streamlit Cloud path
            ]
            
            df = None
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_excel(path)
                        break
                except Exception:
                    continue
                    
            if df is None:
                st.error("Default data file not found. Please upload your data file.")
                return None
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def encode_features(df):
    try:
        label_cols = ["MA", "TE", "Fiber_Type"]
        label_encoders = {}
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        return df, label_encoders
    except Exception as e:
        st.error(f"Error encoding features: {str(e)}")
        return None, None


def train_model(df):
    try:
        X = df.drop(columns=["SNO", "Strength_MPa"])
        y = df["Strength_MPa"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, X_train, X_test, y_train, y_test, y_pred, mse, r2
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None, None, None


def predict_custom_strength(
    model, label_encoders, ma, pma, pca, te, w_c_ratio, curing_days, fiber_type
):
    try:
        ma_encoded = label_encoders["MA"].transform([ma])[0]
        te_encoded = label_encoders["TE"].transform([te])[0]
        fiber_encoded = label_encoders["Fiber_Type"].transform([fiber_type])[0]
        input_df = pd.DataFrame(
            [
                {
                    "MA": ma_encoded,
                    "PMA": pma,
                    "PCA": pca,
                    "TE": te_encoded,
                    "W_C_Ratio": w_c_ratio,
                    "Curing_Days": curing_days,
                    "Fiber_Type": fiber_encoded,
                }
            ]
        )
        prediction = model.predict(input_df)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


# --- Sidebar ---
with st.sidebar:
    st.header("Data Management")
    
    # Data Format Guide
    with st.expander("📋 Data Format Guide", expanded=True):
        st.markdown("""
        ### Required Excel File Format:
        
        Your Excel file should contain the following columns:
        
        | Column Name | Description | Type |
        |------------|-------------|------|
        | SNO | Serial Number | Number |
        | MA | Mineral Admixture | Text |
        | PMA | Percentage of Mineral Admixture | Number |
        | PCA | Percentage of Crystalline | Number |
        | TE | Type of Exposure | Text |
        | W_C_Ratio | Water-Cement Ratio | Number |
        | Curing_Days | Number of Curing Days | Number |
        | Fiber_Type | Type of Fiber Used | Text |
        | Strength_MPa | Compressive Strength | Number |
        
        ### Example Values:
        - **MA**: Fly Ash, GGBS, Silica Fume
        - **TE**: Indoor, Outdoor, Marine
        - **Fiber_Type**: Steel, Glass, Polypropylene
        
        ### Notes:
        - All numerical values should be positive
        - PMA should be between 0-100
        - PCA should be between 0-10
        - W/C Ratio should be between 0.2-1.0
        - Curing Days should be between 1-365
        """)

    # Data Upload Section
    st.subheader("1️⃣ Upload Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        st.success("✅ File uploaded successfully!")

    # Download Section
    st.subheader("2️⃣ Download Results")
    if 'comparison' in locals() and isinstance(comparison, pd.DataFrame):
        st.download_button(
            label="📥 Download Sample Predictions (CSV)",
            data=comparison.to_csv(index=False),
            file_name="sample_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Run predictions first to enable download")

# --- Main Content ---
# Load and Process Data
with st.spinner("Loading and processing data..."):
    df = load_data(uploaded_file)
    if df is not None:
        df, label_encoders = encode_features(df)
        if df is not None and label_encoders is not None:
            model, X_train, X_test, y_train, y_test, y_pred, mse, r2 = train_model(df)
        else:
            st.error("Failed to process the data. Please check your input data format.")
            st.stop()
    else:
        st.error("No data available. Please upload a valid Excel file.")
        st.stop()

# Model Evaluation Section
st.header("📈 Model Performance")
col1, col2, col3 = st.columns(3)

# Safe metric display with error handling
with col1:
    if mse is not None:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    else:
        st.metric("Mean Squared Error", "N/A")

with col2:
    if r2 is not None:
        st.metric("R² Score", f"{r2:.2f}")
    else:
        st.metric("R² Score", "N/A")

with col3:
    if mse is not None:
        accuracy = (1 - mse/100) if mse < 100 else 0
        st.metric("Accuracy", f"{accuracy:.1%}")
    else:
        st.metric("Accuracy", "N/A")

# Sample Predictions
if all(x is not None for x in [y_test, y_pred]):
    st.header("🔍 Sample Predictions")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        st.dataframe(comparison.head(10), use_container_width=True, height=400)

    # Visualizations
    st.header("📊 Visualizations")
    tab1, tab2 = st.tabs(["Bar Chart", "Scatter Plot"])

    with tab1:
        st.write("**Actual vs Predicted (First 10 Samples)**")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        comparison_plot = comparison.head(10)
        comparison_plot.plot(kind="bar", ax=ax1)
        plt.title("Actual vs Predicted Compressive Strength")
        plt.xlabel("Sample")
        plt.ylabel("Strength (MPa)")
        plt.xticks(rotation=0)
        plt.legend(["Actual", "Predicted"])
        st.pyplot(fig1)

    with tab2:
        st.write("**All Test Results**")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(y_test, y_pred, alpha=0.7)
        ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
        plt.xlabel("Actual Strength (MPa)")
        plt.ylabel("Predicted Strength (MPa)")
        plt.title("Actual vs Predicted Scatter Plot")
        plt.grid(True)
        st.pyplot(fig2)
else:
    st.warning("Model predictions are not available. Please check your data and try again.")

# Custom Prediction Section
if model is not None and label_encoders is not None:
    st.header("🧪 Predict Custom Mix Strength")
    st.markdown("Enter the parameters for your concrete mix to predict its strength:")

    with st.form("custom_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Material Properties")
            ma = st.selectbox(
                "Mineral Admixture (MA)", label_encoders["MA"].classes_.tolist()
            )
            te = st.selectbox(
                "Type of Exposure (TE)", label_encoders["TE"].classes_.tolist()
            )
            fiber_type = st.selectbox(
                "Fiber Type", label_encoders["Fiber_Type"].classes_.tolist()
            )

        with col2:
            st.subheader("Mix Proportions")
            pma = st.slider("% MA", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
            pca = st.slider(
                "% Crystalline", min_value=0.0, max_value=10.0, value=1.0, step=0.1
            )

        with col3:
            st.subheader("Curing Conditions")
            w_c_ratio = st.slider(
                "W/C Ratio", min_value=0.2, max_value=1.0, value=0.4, step=0.01
            )
            curing_days = st.slider(
                "Curing Days", min_value=1, max_value=365, value=28, step=1
            )

        submitted = st.form_submit_button("Predict Strength")

    if submitted:
        with st.spinner("Calculating prediction..."):
            pred_strength = predict_custom_strength(
                model, label_encoders, ma, pma, pca, te, w_c_ratio, curing_days, fiber_type
            )
            if pred_strength is not None:
                st.success(f"Predicted Compressive Strength: {pred_strength} MPa")
                # Show prediction confidence
                if mse is not None:
                    confidence = max(0, 100 - (mse * 100))
                    st.info(f"Prediction Confidence: {confidence:.1f}%")
            else:
                st.error("Failed to make prediction. Please check your input values.")
else:
    st.warning("Model is not available. Please upload and process your data first.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built by Agha Faseeh | Self-Healing Concrete Strength Predictor</p>
    </div>
""",
    unsafe_allow_html=True,
)

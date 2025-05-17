import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime
from docx import Document
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)

# Constants
GRID_EMISSION_FACTOR = 0.5  # kg CO2/kWh for grid
SOLAR_EMISSION_FACTOR = 0.05  # kg CO2/kWh for solar
TREES_PER_TON_CO2 = 48  # Trees per ton of CO2 absorbed
MAX_ENERGY_INCREASE = 1.1  # Limit optimized energy to 10% above baseline

def process_uploaded_data(file):
    """Process uploaded Excel/CSV file and map to internal format."""
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file format. Please upload .xlsx or .csv.")

        required_cols = ['Timestamp', 'Total Consumption (kWh)', 'Machine 1 (kWh)', 'Machine 2 (kWh)', 'HVAC (kWh)', 'Lighting (kWh)', 'Other (kWh)']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.rename(columns={
            'Total Consumption (kWh)': 'Total_Energy',
            'Machine 1 (kWh)': 'Energy_Heavy',
            'Machine 2 (kWh)': 'Energy_Medium',
            'HVAC (kWh)': 'HVAC_Energy',
            'Lighting (kWh)': 'Lighting_Energy',
            'Other (kWh)': 'Other_Energy'
        })

        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
        df['Hour'] = df['Timestamp'].dt.hour
        df['Is_Weekday'] = df['Day_of_Week'] < 5
        df['Is_Working_Hours'] = df['Is_Weekday'] & (df['Hour'] >= 8) & (df['Hour'] < 18)

        conditions = [
            (~df['Is_Weekday']),
            (df['Is_Weekday'] & (df['Hour'] >= 8) & (df['Hour'] < 16)),
            (df['Is_Weekday'] & (df['Hour'] >= 16) & (df['Hour'] < 24))
        ]
        choices = ['weekend', 'day', 'night']
        df['Shift'] = np.select(conditions, choices, default='overnight')

        df['Heavy_On'] = (df['Energy_Heavy'] / df['Energy_Heavy'].max() * 5).round().clip(0, 5)
        df['Medium_On'] = (df['Energy_Medium'] / df['Energy_Medium'].max() * 10).round().clip(0, 10)

        shift_map_heavy = {'day': 5, 'night': 2, 'overnight': 0, 'weekend': 0}
        shift_map_medium = {'day': 10, 'night': 10, 'overnight': 2, 'weekend': 2}
        df['Intended_Heavy_On'] = df['Shift'].map(shift_map_heavy)
        df['Intended_Medium_On'] = df['Shift'].map(shift_map_medium)

        df['Temperature'] = 30 + 5 * np.sin(2 * np.pi * (df['Hour'] - 14) / 24) + np.random.normal(0, 1, len(df))
        df['HVAC_Inefficient'] = (df['HVAC_Energy'] > (20 + 10 * np.maximum(df['Temperature'] - 22, 0))).astype(int)

        df['Solar_Available'] = np.where((df['Hour'] >= 6) & (df['Hour'] <= 18), 100 * np.sin(np.pi * (df['Hour'] - 6) / 12), 0)
        df['Solar_Used'] = np.minimum(df['Solar_Available'], df['Total_Energy'])
        df['Grid_Energy'] = np.maximum(0, df['Total_Energy'] - df['Solar_Used'])

        df['CO2_Emissions'] = df['Grid_Energy'] * GRID_EMISSION_FACTOR + df['Solar_Used'] * SOLAR_EMISSION_FACTOR

        logger.info("Processed uploaded data with columns: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Failed to process uploaded data: %s", e)
        st.error(f"Failed to process uploaded data: {e}")
        return None

def simulate_factory_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours=720):
    """Simulate factory energy consumption with inefficiencies and solar data."""
    try:
        timestamps = pd.date_range(start="2025-05-17 00:00", periods=hours, freq="H")
        df = pd.DataFrame({"Timestamp": timestamps})
        df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
        df["Hour"] = df["Timestamp"].dt.hour
        df["Is_Weekday"] = df["Day_of_Week"] < 5

        conditions = [
            (~df["Is_Weekday"]),
            (df["Is_Weekday"] & (df["Hour"] >= 8) & (df["Hour"] < 16)),
            (df["Is_Weekday"] & (df["Hour"] >= 16) & (df["Hour"] < 24))
        ]
        choices = ["weekend", "day", "night"]
        df["Shift"] = np.select(conditions, choices, default="overnight")

        shift_map_heavy = {"day": num_heavy, "night": num_heavy // 2, "overnight": 0, "weekend": 0}
        shift_map_medium = {"day": num_medium, "night": num_medium, "overnight": num_medium // 5, "weekend": num_medium // 5}
        df["Intended_Heavy_On"] = df["Shift"].map(shift_map_heavy)
        df["Intended_Medium_On"] = df["Shift"].map(shift_map_medium)

        df["Heavy_On"] = df["Intended_Heavy_On"] + np.random.binomial(num_heavy - df["Intended_Heavy_On"], p_ineff)
        df["Medium_On"] = df["Intended_Medium_On"] + np.random.binomial(num_medium - df["Intended_Medium_On"], p_ineff)
        df["Energy_Heavy"] = df["Heavy_On"] * energy_heavy
        df["Energy_Medium"] = df["Medium_On"] * energy_medium

        df["Temperature"] = 30 + 5 * np.sin(2 * np.pi * (df["Hour"] - 14) / 24) + np.random.normal(0, 1, len(df))
        df["HVAC_Inefficient"] = np.random.choice([0, 1], size=len(df), p=[1 - q_ineff, q_ineff])
        df["HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - (22 - 2 * df["HVAC_Inefficient"]), 0)

        df["Is_Working_Hours"] = df["Is_Weekday"] & (df["Hour"] >= 8) & (df["Hour"] < 18)
        df["Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, np.where(np.random.random(len(df)) < r_ineff, 50, 10))

        df["Solar_Available"] = np.where((df["Hour"] >= 6) & (df["Hour"] <= 18), 100 * np.sin(np.pi * (df["Hour"] - 6) / 12), 0)
        df["Solar_Used"] = np.minimum(df["Solar_Available"], df["Energy_Heavy"] + df["Energy_Medium"] + df["HVAC_Energy"] + df["Lighting_Energy"])
        df["Grid_Energy"] = np.maximum(0, df["Energy_Heavy"] + df["Energy_Medium"] + df["HVAC_Energy"] + df["Lighting_Energy"] - df["Solar_Used"])

        df["Total_Energy"] = df[["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]].sum(axis=1)
        df["CO2_Emissions"] = df["Grid_Energy"] * GRID_EMISSION_FACTOR + df["Solar_Used"] * SOLAR_EMISSION_FACTOR

        df["Efficient_Heavy_On"] = df["Intended_Heavy_On"]
        df["Efficient_Medium_On"] = df["Intended_Medium_On"]
        df["Efficient_Energy_Heavy"] = df["Efficient_Heavy_On"] * energy_heavy
        df["Efficient_Energy_Medium"] = df["Efficient_Medium_On"] * energy_medium
        df["Efficient_HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - 22, 0)
        df["Efficient_Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, 10)
        df["Efficient_Total_Energy"] = df[["Efficient_Energy_Heavy", "Efficient_Energy_Medium", "Efficient_HVAC_Energy", "Efficient_Lighting_Energy"]].sum(axis=1)
        df["Efficient_CO2_Emissions"] = df["Efficient_Total_Energy"] * GRID_EMISSION_FACTOR

        logger.info("Simulation completed with columns: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        st.error(f"Simulation failed: {e}")
        return None

class CarbonOptimizer:
    """Simple Q-learning for carbon-aware scheduling."""
    def __init__(self, actions=["shift_heavy", "shift_medium", "reduce_load", "no_action"], lr=0.1, gamma=0.9):
        self.q_table = {}
        self.actions = actions
        self.lr = lr
        self.gamma = gamma

    def get_state(self, shift, hour, solar_available):
        return (shift, hour, int(solar_available > 0))

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_table.get(state, {a: 0 for a in self.actions})
        return max(q_values, key=q_values.get)

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.lr * (reward + self.gamma * next_max_q - current_q)

@st.cache_data
def get_simulated_data(_num_heavy, _energy_heavy, _num_medium, _energy_medium, _p_ineff, _q_ineff, _r_ineff, _hours):
    return simulate_factory_data(_num_heavy, _energy_heavy, _num_medium, _energy_medium, _p_ineff, _q_ineff, _r_ineff, _hours)

@st.cache_resource
def train_model(df, target, model_type, cache_key):
    try:
        data = df[["Shift", "Hour", "Day_of_Week", target]].dropna()
        X = data[["Shift", "Hour", "Day_of_Week"]]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = ColumnTransformer(
            transformers=[("shift", OneHotEncoder(drop="first", sparse_output=False), ["Shift"])],
            remainder="passthrough"
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type == "Random Forest" else LinearRegression()
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        joblib.dump(pipeline, f"model_{target}_{model_type.replace(' ', '_')}.joblib")
        return pipeline, {"MAE": mae, "MSE": mse, "R2": r2}
    except Exception as e:
        logger.error("Model training failed: %s", e)
        st.error(f"Model training failed: {e}")
        return None, None

def create_docx_from_markdown(markdown_text):
    """Convert markdown text to a .docx file."""
    try:
        doc = Document()
        lines = markdown_text.split('\n')
        for line in lines:
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                doc.add_heading(line[4:], level=3)
            elif line.startswith('- '):
                doc.add_paragraph(line[2:], style='ListBullet')
            else:
                doc.add_paragraph(line)
        output = BytesIO()
        doc.save(output)
        output.seek(0)
        return output
    except Exception as e:
        logger.error("Failed to create .docx: %s", e)
        st.error(f"Failed to create .docx: {e}")
        return None

def detect_inefficiencies(df):
    """Detect inefficiencies in energy consumption data."""
    inefficiencies = []
    
    heavy_ineff = df[(df['Heavy_On'] > df['Intended_Heavy_On']) & (df['Shift'].isin(['overnight', 'weekend']))]
    medium_ineff = df[(df['Medium_On'] > df['Intended_Medium_On']) & (df['Shift'].isin(['overnight', 'weekend']))]
    if not heavy_ineff.empty:
        inefficiencies.append(f"Heavy machines running unnecessarily during {len(heavy_ineff)} off-hours (overnight/weekend).")
    if not medium_ineff.empty:
        inefficiencies.append(f"Medium machines running unnecessarily during {len(medium_ineff)} off-hours (overnight/weekend).")

    hvac_ineff = df[df['HVAC_Inefficient'] == 1]
    if not hvac_ineff.empty:
        inefficiencies.append(f"HVAC operating inefficiently (low temperature threshold) for {len(hvac_ineff)} hours.")

    lighting_ineff = df[(~df['Is_Working_Hours']) & (df['Lighting_Energy'] > 10)]
    if not lighting_ineff.empty:
        inefficiencies.append(f"Lighting left on unnecessarily during {len(lighting_ineff)} non-working hours.")

    return inefficiencies

def recommend_actions(inefficiencies):
    """Recommend actions based on detected inefficiencies."""
    actions = []
    for ineff in inefficiencies:
        if "Heavy machines" in ineff:
            actions.append("Implement strict shutdown protocols for heavy machines during overnight and weekend shifts.")
        if "Medium machines" in ineff:
            actions.append("Schedule medium machines to operate only during intended shifts (day/night).")
        if "HVAC" in ineff:
            actions.append("Reset HVAC systems to a 22°C threshold to avoid overcooling.")
        if "Lighting" in ineff:
            actions.append("Install automated lighting controls to turn off lights outside working hours (8 AM–6 PM weekdays).")
    return actions

def main():
    st.set_page_config(page_title="Factory Energy & Carbon Optimizer", layout="wide")
    st.title("Factory Energy & Carbon Optimizer")

    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Results", "Carbon Impact", "Documentation"])

    with tab1:
        st.header("Factory Setup")
        data_source = st.radio("Select Data Source", ["Simulate Data", "Upload Data"])

        if data_source == "Simulate Data":
            col1, col2 = st.columns(2)
            with col1:
                num_heavy = st.number_input("Number of Heavy Machines", min_value=1, value=5, step=1)
                energy_heavy = st.number_input("Energy per Heavy Machine (kW)", min_value=0.1, value=20.0, step=0.5)
                num_medium = st.number_input("Number of Medium Machines", min_value=1, value=10, step=1)
                energy_medium = st.number_input("Energy per Medium Machine (kW)", min_value=0.1, value=10.0, step=0.5)
            with col2:
                p_ineff = st.slider("Machine Inefficiency Probability", 0.0, 1.0, 0.1, 0.01)
                q_ineff = st.slider("HVAC Inefficiency Probability", 0.0, 1.0, 0.2, 0.01)
                r_ineff = st.slider("Lighting Inefficiency Probability", 0.0, 1.0, 0.1, 0.01)
                duration = st.selectbox("Simulation Duration", ["1 Week (168h)", "1 Month (720h)", "3 Months (2160h)"], index=1)
                hours = {"1 Week (168h)": 168, "1 Month (720h)": 720, "3 Months (2160h)": 2160}[duration]
        else:
            uploaded_file = st.file_uploader("Upload Energy Data (Excel/CSV)", type=['xlsx', 'csv'])
            num_heavy = 5
            energy_heavy = 20.0
            num_medium = 10
            energy_medium = 10.0
            p_ineff = q_ineff = r_ineff = 0.1
            hours = None

        cost_per_kwh = st.number_input("Cost per kWh ($)", min_value=0.0, value=0.15, step=0.01)
        model_type = st.selectbox("Prediction Model", ["Random Forest", "Linear Regression"])

        if st.button("Run"):
            with st.spinner("Processing..."):
                if data_source == "Simulate Data":
                    df = get_simulated_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours)
                else:
                    if uploaded_file is None:
                        st.error("Please upload a file.")
                        return
                    df = process_uploaded_data(uploaded_file)

                if df is not None:
                    st.session_state["df"] = df
                    st.session_state["params"] = {
                        "num_heavy": num_heavy,
                        "energy_heavy": energy_heavy,
                        "num_medium": num_medium,
                        "energy_medium": energy_medium,
                        "cost_per_kwh": cost_per_kwh,
                        "model_type": model_type,
                        "data_source": data_source
                    }
                    st.success("Data processed successfully!")
                    st.rerun()

    with tab2:
        if "df" in st.session_state and "params" in st.session_state:
            df = st.session_state["df"]
            params = st.session_state["params"]
            st.header("Optimization Results")
            required_cols = ["Shift", "Hour", "Solar_Available", "CO2_Emissions", "Intended_Heavy_On", "Intended_Medium_On",
                           "Day_of_Week", "Heavy_On", "Medium_On", "Energy_Heavy", "Energy_Medium", "HVAC_Energy",
                           "Lighting_Energy", "Temperature", "HVAC_Inefficient", "Is_Working_Hours", "Solar_Used", "Grid_Energy"]
            if not all(col in df.columns for col in required_cols):
                st.error("Missing required columns. Please re-run the simulation or upload valid data.")
            else:
                try:
                    st.subheader("Detected Inefficiencies")
                    inefficiencies = detect_inefficiencies(df)
                    if inefficiencies:
                        for ineff in inefficiencies:
                            st.write(f"- {ineff}")
                    else:
                        st.write("No significant inefficiencies detected.")

                    st.subheader("Recommended Actions")
                    actions = recommend_actions(inefficiencies)
                    if actions:
                        for action in actions:
                            st.write(f"- {action}")
                    else:
                        st.write("No actions required.")

                    with st.spinner("Optimizing..."):
                        heavy_model, heavy_metrics = train_model(df, "Intended_Heavy_On", params["model_type"], "heavy")
                        medium_model, medium_metrics = train_model(df, "Intended_Medium_On", params["model_type"], "medium")

                    if heavy_model and medium_model:
                        st.subheader("Model Accuracy")
                        st.markdown("**Heavy Machines Model**")
                        st.table({
                            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "R-squared (R²)"],
                            "Value": [f"{heavy_metrics['MAE']:.4f}", f"{heavy_metrics['MSE']:.4f}", f"{heavy_metrics['R2']:.4f}"]
                        })
                        st.markdown("**Medium Machines Model**")
                        st.table({
                            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "R-squared (R²)"],
                            "Value": [f"{medium_metrics['MAE']:.4f}", f"{medium_metrics['MSE']:.4f}", f"{medium_metrics['R2']:.4f}"]
                        })

                        optimizer = CarbonOptimizer()
                        for i in range(min(100, len(df) - 1)):
                            state = optimizer.get_state(df["Shift"].iloc[i], df["Hour"].iloc[i], df["Solar_Available"].iloc[i])
                            action = optimizer.get_action(state)
                            reward = -df["CO2_Emissions"].iloc[i] / (df["Total_Energy"].iloc[i] + 1)  # Normalize reward by energy
                            next_state = optimizer.get_state(df["Shift"].iloc[i + 1], df["Hour"].iloc[i + 1], df["Solar_Available"].iloc[i + 1])
                            optimizer.update(state, action, reward, next_state)

                        df["Predicted_Heavy_On"] = heavy_model.predict(df[["Shift", "Hour", "Day_of_Week"]]).round().clip(0, params["num_heavy"])
                        df["Predicted_Medium_On"] = medium_model.predict(df[["Shift", "Hour", "Day_of_Week"]]).round().clip(0, params["num_medium"])
                        df["Optimized_Heavy_On"] = df["Predicted_Heavy_On"].copy()
                        df["Optimized_Medium_On"] = df["Predicted_Medium_On"].copy()

                        for i in range(len(df)):
                            state = optimizer.get_state(df["Shift"].iloc[i], df["Hour"].iloc[i], df["Solar_Available"].iloc[i])
                            action = optimizer.get_action(state, epsilon=0.1)
                            if action == "shift_heavy" and df["Solar_Available"].iloc[i] > 0 and df["Hour"].iloc[i] < 18:
                                df.loc[i, "Optimized_Heavy_On"] = min(df["Heavy_On"].iloc[i], df["Predicted_Heavy_On"].iloc[i])
                            elif action == "shift_medium" and df["Solar_Available"].iloc[i] > 0 and df["Hour"].iloc[i] < 18:
                                df.loc[i, "Optimized_Medium_On"] = min(df["Medium_On"].iloc[i], df["Predicted_Medium_On"].iloc[i])
                            elif action == "reduce_load":
                                df.loc[i, "Optimized_Heavy_On"] = max(0, df["Optimized_Heavy_On"].iloc[i] - 1)
                                df.loc[i, "Optimized_Medium_On"] = max(0, df["Optimized_Medium_On"].iloc[i] - 1)

                        df["Optimized_Energy_Heavy"] = df["Optimized_Heavy_On"] * params["energy_heavy"]
                        df["Optimized_Energy_Medium"] = df["Optimized_Medium_On"] * params["energy_medium"]
                        df["Optimized_HVAC_Energy"] = np.where(
                            df["HVAC_Inefficient"] == 0, df["HVAC_Energy"], 20 + 10 * np.maximum(df["Temperature"] - 22, 0)
                        )
                        df["Optimized_Lighting_Energy"] = np.where(df["Is_Working_Hours"] | (df["Lighting_Energy"] == 10), df["Lighting_Energy"], 10)
                        df["Optimized_Total_Energy"] = df[["Optimized_Energy_Heavy", "Optimized_Energy_Medium", "Optimized_HVAC_Energy", "Optimized_Lighting_Energy"]].sum(axis=1)
                        df["Optimized_Solar_Used"] = np.minimum(df["Solar_Available"], df["Optimized_Total_Energy"])
                        df["Optimized_Grid_Energy"] = np.maximum(0, df["Optimized_Total_Energy"] - df["Optimized_Solar_Used"])
                        df["Optimized_CO2_Emissions"] = df["Optimized_Grid_Energy"] * GRID_EMISSION_FACTOR + df["Optimized_Solar_Used"] * SOLAR_EMISSION_FACTOR

                        # Cap optimized energy to avoid excessive increases
                        df["Optimized_Total_Energy"] = np.where(df["Optimized_Total_Energy"] > df["Total_Energy"] * MAX_ENERGY_INCREASE,
                                                              df["Total_Energy"], df["Optimized_Total_Energy"])
                        df["Optimized_Grid_Energy"] = np.maximum(0, df["Optimized_Total_Energy"] - df["Optimized_Solar_Used"])
                        df["Optimized_CO2_Emissions"] = df["Optimized_Grid_Energy"] * GRID_EMISSION_FACTOR + df["Optimized_Solar_Used"] * SOLAR_EMISSION_FACTOR

                        baseline_energy = df["Total_Energy"].sum()
                        optimized_energy = df["Optimized_Total_Energy"].sum()
                        energy_savings = baseline_energy - optimized_energy
                        savings_percent = (energy_savings / baseline_energy * 100) if baseline_energy > 0 else 0
                        cost_savings = energy_savings * params["cost_per_kwh"]
                        baseline_co2 = df["CO2_Emissions"].sum()
                        optimized_co2 = df["Optimized_CO2_Emissions"].sum()
                        co2_savings = baseline_co2 - optimized_co2
                        trees_equivalent = (co2_savings / 1000) * TREES_PER_TON_CO2

                        st.subheader("Summary")
                        st.table({
                            "Metric": ["Baseline Energy (kWh)", "Optimized Energy (kWh)", "Energy Savings (kWh)", "Savings (%)",
                                       "Cost Savings ($)", "Baseline CO2 (kg)", "Optimized CO2 (kg)", "CO2 Savings (kg)",
                                       "Trees Equivalent"],
                            "Value": [f"{baseline_energy:.2f}", f"{optimized_energy:.2f}", f"{energy_savings:.2f}",
                                      f"{savings_percent:.2f}", f"{cost_savings:.2f}", f"{baseline_co2:.2f}",
                                      f"{optimized_co2:.2f}", f"{co2_savings:.2f}", f"{trees_equivalent:.2f}"]
                        })

                        st.subheader("Energy Usage")
                        fig_energy = go.Figure()
                        fig_energy.add_trace(go.Scatter(x=df["Timestamp"], y=df["Total_Energy"], name="Baseline", line=dict(color="red")))
                        fig_energy.add_trace(go.Scatter(x=df["Timestamp"], y=df["Optimized_Total_Energy"], name="Optimized", line=dict(color="green")))
                        fig_energy.update_layout(
                            title="Energy Usage (kW)",
                            yaxis_title="Energy (kW)",
                            xaxis_title="Timestamp",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_energy, use_container_width=True)

                        st.subheader("Energy Breakdown")
                        fig_breakdown = go.Figure()
                        for col, name, color in [("Energy_Heavy", "Heavy Machines", "blue"), ("Energy_Medium", "Medium Machines", "orange"),
                                               ("HVAC_Energy", "HVAC", "green"), ("Lighting_Energy", "Lighting", "purple")]:
                            fig_breakdown.add_trace(go.Scatter(x=df["Timestamp"], y=df[col], name=name, stackgroup="one", line=dict(color=color)))
                        fig_breakdown.update_layout(
                            title="Baseline Energy Breakdown (kW)",
                            yaxis_title="Energy (kW)",
                            xaxis_title="Timestamp",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            showlegend=True
                        )
                        st.plotly_chart(fig_breakdown, use_container_width=True)

                        st.subheader("Daily Savings")
                        df["Date"] = df["Timestamp"].dt.date
                        daily_savings = df.groupby("Date").apply(lambda x: x["Total_Energy"].sum() - x["Optimized_Total_Energy"].sum()).reset_index(name="Savings")
                        daily_savings = daily_savings[daily_savings["Savings"] != 0]  # Filter out zero savings
                        fig_savings = px.bar(daily_savings, x="Date", y="Savings", title="Daily Energy Savings (kWh)", color_discrete_sequence=["teal"])
                        fig_savings.update_layout(
                            yaxis_title="Savings (kWh)",
                            xaxis_title="Date",
                            hovermode="x unified",
                            showlegend=False
                        )
                        st.plotly_chart(fig_savings, use_container_width=True)

                        st.subheader("Export")
                        csv = df.to_csv(index=False)
                        st.download_button("Download Data as CSV", csv, f"factory_energy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

                except Exception as e:
                    st.error(f"Optimization error: {e}")
                    logger.error(f"Optimization error: {e}")

    with tab3:
        if "df" in st.session_state:
            df = st.session_state["df"]
            st.header("Carbon Impact")
            if "CO2_Emissions" not in df.columns or "Optimized_CO2_Emissions" not in df.columns:
                st.error("Carbon data missing. Please re-run the simulation or upload valid data.")
            else:
                try:
                    st.subheader("Emissions Over Time")
                    fig_co2_time = go.Figure()
                    fig_co2_time.add_trace(go.Scatter(x=df["Timestamp"], y=df["CO2_Emissions"], name="Baseline", line=dict(color="red")))
                    fig_co2_time.add_trace(go.Scatter(x=df["Timestamp"], y=df["Optimized_CO2_Emissions"], name="Optimized", line=dict(color="green")))
                    fig_co2_time.update_layout(
                        title="CO2 Emissions Over Time (kg)",
                        yaxis_title="CO2 Emissions (kg)",
                        xaxis_title="Timestamp",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_co2_time, use_container_width=True)

                    st.subheader("Emissions Breakdown")
                    co2_breakdown = {
                        "Heavy Machines": (df["Energy_Heavy"] * GRID_EMISSION_FACTOR).sum(),
                        "Medium Machines": (df["Energy_Medium"] * GRID_EMISSION_FACTOR).sum(),
                        "HVAC": (df["HVAC_Energy"] * GRID_EMISSION_FACTOR).sum(),
                        "Lighting": (df["Lighting_Energy"] * GRID_EMISSION_FACTOR).sum()
                    }
                    fig_pie = px.pie(names=list(co2_breakdown.keys()), values=list(co2_breakdown.values()), title="Baseline CO2 Breakdown")
                    fig_pie.update_layout(
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.subheader("Daily CO2 Emissions Comparison")
                    df["Date"] = df["Timestamp"].dt.date
                    daily_co2 = df.groupby("Date").agg({
                        "CO2_Emissions": "sum",
                        "Optimized_CO2_Emissions": "sum"
                    }).reset_index()
                    daily_co2 = daily_co2[(daily_co2["CO2_Emissions"] > 0) | (daily_co2["Optimized_CO2_Emissions"] > 0)]  # Filter out zero values
                    fig_daily_co2 = go.Figure()
                    fig_daily_co2.add_trace(go.Bar(
                        x=daily_co2["Date"],
                        y=daily_co2["CO2_Emissions"],
                        name="Baseline",
                        marker_color="red"
                    ))
                    fig_daily_co2.add_trace(go.Bar(
                        x=daily_co2["Date"],
                        y=daily_co2["Optimized_CO2_Emissions"],
                        name="Optimized",
                        marker_color="green"
                    ))
                    fig_daily_co2.update_layout(
                        title="Daily CO2 Emissions Comparison (kg)",
                        yaxis_title="CO2 Emissions (kg)",
                        xaxis_title="Date",
                        barmode="group",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_daily_co2, use_container_width=True)

                    baseline_co2 = df["CO2_Emissions"].sum()
                    optimized_co2 = df["Optimized_CO2_Emissions"].sum()
                    co2_savings = baseline_co2 - optimized_co2
                    trees_equivalent = (co2_savings / 1000) * TREES_PER_TON_CO2

                    st.subheader("Sustainability Impact")
                    st.markdown(f"""
                    - **CO2 Savings**: {co2_savings:.2f} kg
                    - **Trees Equivalent**: {trees_equivalent:.2f} trees
                    - **Goal**: Supports UN SDG 7 & 13 by reducing industrial emissions.
                    """)

                except Exception as e:
                    st.error(f"Carbon impact error: {e}")
                    logger.error(f"Carbon impact error: {e}")

    with tab4:
        st.header("Documentation")
        documentation_text = """
        ### Factory Energy & Carbon Optimizer

        **Overview**  
        The Factory Energy & Carbon Optimizer is an AI-driven tool designed to simulate or analyze energy consumption and carbon emissions in a factory environment. It supports both synthetic data generation and user-uploaded data (Excel/CSV) to identify inefficiencies (e.g., idle machines, overheating HVAC) and recommend actions to reduce energy waste and CO2 emissions. The tool aligns with UN Sustainable Development Goals (SDGs) 7 and 13. Built as a Streamlit web application, it provides an interactive interface for configuring simulations, uploading data, viewing results, and exporting data.

        **Data Sources**  
        - **Simulated Data**: Generates synthetic hourly energy consumption data for heavy/medium machines, HVAC, and lighting, with configurable inefficiencies and solar availability.  
        - **Uploaded Data**: Accepts Excel/CSV files with columns: `Timestamp`, `Total Consumption (kWh)`, `Machine 1 (kWh)`, `Machine 2 (kWh)`, `HVAC (kWh)`, `Lighting (kWh)`, `Other (kWh)`. The data is mapped to internal formats for analysis and optimization.

        **Simulation Assumptions**  
        - **Time Period**: Starts May 17, 2025, with durations (1 week: 168 hours, 1 month: 720 hours, 3 months: 2160 hours). For uploaded data, uses provided timestamps.  
        - **Shifts and Occupancy**:  
          - Day Shift: Weekdays, 8 AM–4 PM, full operation.  
          - Night Shift: Weekdays, 4 PM–12 AM, reduced heavy machines.  
          - Overnight Shift: Weekdays, 12 AM–8 AM, minimal operation.  
          - Weekend: Minimal operation.  
        - **Emission Factors**: Grid (0.5 kg CO2/kWh), Solar (0.05 kg CO2/kWh).  
        - **Equipment Profiles**: Configurable for simulation; for uploaded data, assumes 5 heavy machines (20 kW each) and 10 medium machines (10 kW each).  
        - **Inefficiencies**: Detected in uploaded data (e.g., machines running off-hours, inefficient HVAC, lighting overuse).  
        - **Solar Availability**: Peaks at 100 kW from 6 AM–6 PM, zero at night.  
        - **Carbon Offset**: 48 trees absorb 1 ton of CO2 annually.

        **Features**  
        - **Data Processing**: Handles both simulated and uploaded data, with validation and mapping.  
        - **Inefficiency Detection**: Identifies wasteful energy use (e.g., machines running off-hours).  
        - **Optimization**: Uses AI to schedule machines efficiently and shift loads to solar hours.  
        - **Visualization**: Interactive plots for energy usage, breakdown, savings, and CO2 emissions.  
        - **Export Options**: CSV for data, DOCX for documentation.

        **Usage**  
        1. Select data source (simulate or upload) in the Simulation tab.  
        2. Configure parameters or upload a file and run the analysis.  
        3. View inefficiencies, recommendations, and optimized results in the Results tab.  
        4. Analyze carbon impact in the Carbon Impact tab.  
        5. Export data or documentation as needed.

        This tool demonstrates AI-driven sustainability for industrial operations.
        """
        st.markdown(documentation_text)

        st.subheader("Export Documentation")
        docx_file = create_docx_from_markdown(documentation_text)
        if docx_file:
            st.download_button(
                label="Download Documentation as DOCX",
                data=docx_file,
                file_name=f"factory_energy_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if __name__ == "__main__":
    main()

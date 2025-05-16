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
from sklearn.metrics import mean_absolute_error
import joblib
import logging
from datetime import datetime
import threading
import time
import requests
from prophet import Prophet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)

# Constants (moved to config-like structure)
CONFIG = {
    "GRID_EMISSION_FACTOR": 0.5,  # kg CO2/kWh for grid
    "SOLAR_EMISSION_FACTOR": 0.05,  # kg CO2/kWh for solar
    "TREES_PER_TON_CO2": 48,  # Trees per ton of CO2 absorbed
    "CARBON_CREDIT_PRICE": 20.0  # $/ton CO2
}

def simulate_factory_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours=720):
    """
    Simulates factory energy consumption with inefficiencies and solar data.

    Args:
        num_heavy (int): Number of heavy machines.
        energy_heavy (float): Energy consumption per heavy machine (kW).
        num_medium (int): Number of medium machines.
        energy_medium (float): Energy consumption per medium machine (kW).
        p_ineff (float): Machine inefficiency probability.
        q_ineff (float): HVAC inefficiency probability.
        r_ineff (float): Lighting inefficiency probability.
        hours (int): Simulation duration in hours.

    Returns:
        pd.DataFrame: Simulated data with energy and emissions.
    """
    try:
        timestamps = pd.date_range(start="2025-06-01 00:00", periods=hours, freq="H")
        df = pd.DataFrame({"Timestamp": timestamps})
        df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
        df["Hour"] = df["Timestamp"].dt.hour
        df["Is_Weekday"] = df["Day_of_Week"] < 5

        # Shift logic
        conditions = [
            (~df["Is_Weekday"]),
            (df["Is_Weekday"] & (df["Hour"] >= 8) & (df["Hour"] < 16)),
            (df["Is_Weekday"] & (df["Hour"] >= 16) & (df["Hour"] < 24))
        ]
        choices = ["weekend", "day", "night"]
        df["Shift"] = np.select(conditions, choices, default="overnight")

        # Machine schedules
        shift_map_heavy = {"day": num_heavy, "night": num_heavy // 2, "overnight": 0, "weekend": 0}
        shift_map_medium = {"day": num_medium, "night": num_medium, "overnight": num_medium // 5, "weekend": num_medium // 5}
        df["Intended_Heavy_On"] = df["Shift"].map(shift_map_heavy)
        df["Intended_Medium_On"] = df["Shift"].map(shift_map_medium)

        # Inefficiencies
        df["Heavy_On"] = df["Intended_Heavy_On"] + np.random.binomial(num_heavy - df["Intended_Heavy_On"], p_ineff)
        df["Medium_On"] = df["Intended_Medium_On"] + np.random.binomial(num_medium - df["Intended_Medium_On"], p_ineff)
        df["Energy_Heavy"] = df["Heavy_On"] * energy_heavy
        df["Energy_Medium"] = df["Medium_On"] * energy_medium

        # HVAC and temperature
        df["Temperature"] = 30 + 5 * np.sin(2 * np.pi * (df["Hour"] - 14) / 24) + np.random.normal(0, 1, len(df))
        df["HVAC_Inefficient"] = np.random.choice([0, 1], size=len(df), p=[1 - q_ineff, q_ineff])
        df["HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - (22 - 2 * df["HVAC_Inefficient"]), 0)

        # Lighting
        df["Is_Working_Hours"] = df["Is_Weekday"] & (df["Hour"] >= 8) & (df["Hour"] < 18)
        df["Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, np.where(np.random.random(len(df)) < r_ineff, 50, 10))

        # Solar energy
        df["Solar_Available"] = np.where((df["Hour"] >= 6) & (df["Hour"] <= 18), 100 * np.sin(np.pi * (df["Hour"] - 6) / 12), 0)
        df["Solar_Used"] = np.minimum(df["Solar_Available"], df["Energy_Heavy"] + df["Energy_Medium"] + df["HVAC_Energy"] + df["Lighting_Energy"])
        df["Grid_Energy"] = np.maximum(0, df["Energy_Heavy"] + df["Energy_Medium"] + df["HVAC_Energy"] + df["Lighting_Energy"] - df["Solar_Used"])

        # Totals and emissions
        df["Total_Energy"] = df[["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]].sum(axis=1)
        df["CO2_Emissions"] = df["Grid_Energy"] * CONFIG["GRID_EMISSION_FACTOR"] + df["Solar_Used"] * CONFIG["SOLAR_EMISSION_FACTOR"]

        # Efficient baseline
        df["Efficient_Heavy_On"] = df["Intended_Heavy_On"]
        df["Efficient_Medium_On"] = df["Intended_Medium_On"]
        df["Efficient_Energy_Heavy"] = df["Efficient_Heavy_On"] * energy_heavy
        df["Efficient_Energy_Medium"] = df["Efficient_Medium_On"] * energy_medium
        df["Efficient_HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - 22, 0)
        df["Efficient_Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, 10)
        df["Efficient_Total_Energy"] = df[["Efficient_Energy_Heavy", "Efficient_Energy_Medium", "Efficient_HVAC_Energy", "Efficient_Lighting_Energy"]].sum(axis=1)
        df["Efficient_CO2_Emissions"] = df["Efficient_Total_Energy"] * CONFIG["GRID_EMISSION_FACTOR"]

        logger.info("Simulation completed with columns: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        st.error(f"Simulation failed: {e}. Please check input values and try again.")
        return None

class CarbonOptimizer:
    """
    Simple Q-learning for carbon-aware scheduling.

    Args:
        actions (list): List of possible actions.
        lr (float): Learning rate.
        gamma (float): Discount factor.
    """
    def __init__(self, actions=["shift_heavy", "shift_medium", "reduce_hvac", "no_action"], lr=0.1, gamma=0.9):
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

@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def get_simulated_data(_num_heavy, _energy_heavy, _num_medium, _energy_medium, _p_ineff, _q_ineff, _r_ineff, _hours):
    return simulate_factory_data(_num_heavy, _energy_heavy, _num_medium, _energy_medium, _p_ineff, _q_ineff, _r_ineff, _hours)

@st.cache_resource
def train_model(_df, target, model_type, cache_key):
    """
    Trains a model to predict target variable.

    Args:
        _df (pd.DataFrame): Input data.
        target (str): Target column name.
        model_type (str): Model type ("Random Forest", "Linear Regression", "Prophet").
        cache_key (str): Cache key for model.

    Returns:
        tuple: Trained model and MAE.
    """
    try:
        data = _df[["Shift", "Hour", "Day_of_Week", "Timestamp", target]].dropna()
        if model_type == "Prophet":
            prophet_df = data[["Timestamp", target]].rename(columns={"Timestamp": "ds", target: "y"})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=24, freq="H")
            forecast = model.predict(future)
            y_pred = forecast["yhat"][:len(data)]
            mae = mean_absolute_error(data[target], y_pred)
            joblib.dump(model, f"model_{target}_Prophet.joblib")
            return model, mae
        else:
            X = data[["Shift", "Hour", "Day_of_Week"]]
            y = data[target]
            preprocessor = ColumnTransformer(
                transformers=[("shift", OneHotEncoder(drop="first", sparse_output=False), ["Shift"])],
                remainder="passthrough"
            )
            model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type == "Random Forest" else LinearRegression()
            pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            mae = mean_absolute_error(y, y_pred)
            joblib.dump(pipeline, f"model_{target}_{model_type.replace(' ', '_')}.joblib")
            return pipeline, mae
    except Exception as e:
        logger.error("Model training failed: %s", e)
        st.error(f"Model training failed: {e}. Please ensure data is complete.")
        return None, None

def generate_pdf_report(summary_data, filename):
    """
    Generates a PDF report of simulation results.

    Args:
        summary_data (dict): Summary metrics.
        filename (str): Output PDF filename.
    """
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Factory Energy & Carbon Optimizer Report", styles['Title']))
        elements.append(Spacer(1, 12))

        data = [[k, v] for k, v in summary_data.items()]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)

        doc.build(elements)
        return filename
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        st.error(f"PDF generation failed: {e}")
        return None

def stream_iot_data(params, hours):
    """
    Simulates IoT data streaming by periodically updating simulation data.

    Args:
        params (dict): Simulation parameters.
        hours (int): Simulation duration in hours.
    """
    while st.session_state.get("streaming", False):
        df = simulate_factory_data(
            params["num_heavy"], params["energy_heavy"], params["num_medium"],
            params["energy_medium"], params["p_ineff"], params["q_ineff"], params["r_ineff"], hours
        )
        st.session_state["df"] = df
        time.sleep(300)  # Update every 5 minutes

def render_simulation_tab():
    """Renders the simulation tab with form-based inputs."""
    st.header("Factory Setup")
    if "factories" not in st.session_state:
        st.session_state["factories"] = {}
    factory_name = st.text_input("Factory Name", value="Factory_1")
    with st.form("simulation_form"):
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
        cost_per_kwh = st.number_input("Cost per kWh ($)", min_value=0.0, value=0.15, step=0.01)
        model_type = st.selectbox("Prediction Model", ["Random Forest", "Linear Regression", "Prophet"])
        submitted = st.form_submit_button("Run Simulation")

        if submitted:
            with st.spinner("Simulating..."):
                progress = st.progress(0)
                df = get_simulated_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours)
                progress.progress(100)
                if df is not None:
                    st.session_state["factories"][factory_name] = df
                    st.session_state["params"] = {
                        "num_heavy": num_heavy, "energy_heavy": energy_heavy, "num_medium": num_medium,
                        "energy_medium": energy_medium, "p_ineff": p_ineff, "q_ineff": q_ineff, "r_ineff": r_ineff,
                        "cost_per_kwh": cost_per_kwh, "model_type": model_type, "hours": hours
                    }
                    st.session_state["current_factory"] = factory_name
                    st.success(f"Simulation for {factory_name} completed!")
                    st.rerun()

    if st.button("Start IoT Stream"):
        st.session_state["streaming"] = True
        threading.Thread(target=stream_iot_data, args=(st.session_state["params"], hours), daemon=True).start()
    if st.button("Stop IoT Stream"):
        st.session_state["streaming"] = False

def render_results_tab():
    """Renders the results tab with optimization and visualizations."""
    if "factories" not in st.session_state or not st.session_state["factories"]:
        st.warning("No simulation data available. Run a simulation first.")
        return
    factory_name = st.selectbox("Select Factory", list(st.session_state["factories"].keys()))
    df = st.session_state["factories"].get(factory_name)
    params = st.session_state.get("params", {})
    if df is None:
        st.error("No data for selected factory.")
        return

    st.header("Optimization Results")
    required_cols = ["Shift", "Hour", "Solar_Available", "CO2_Emissions", "Intended_Heavy_On", "Intended_Medium_On",
                    "Day_of_Week", "Heavy_On", "Medium_On", "Energy_Heavy", "Energy_Medium", "HVAC_Energy",
                    "Lighting_Energy", "Temperature", "HVAC_Inefficient", "Is_Working_Hours", "Solar_Used", "Grid_Energy"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns. Please re-run the simulation.")
        return

    try:
        with st.spinner("Optimizing..."):
            progress = st.progress(0)
            heavy_model, heavy_mae = train_model(df, "Intended_Heavy_On", params["model_type"], f"heavy_{factory_name}")
            medium_model, medium_mae = train_model(df, "Intended_Medium_On", params["model_type"], f"medium_{factory_name}")
            progress.progress(50)

            if heavy_model and medium_model:
                optimizer = CarbonOptimizer()
                for i in range(min(100, len(df) - 1)):
                    state = optimizer.get_state(df["Shift"].iloc[i], df["Hour"].iloc[i], df["Solar_Available"].iloc[i])
                    action = optimizer.get_action(state)
                    reward = -df["CO2_Emissions"].iloc[i]
                    next_state = optimizer.get_state(df["Shift"].iloc[i + 1], df["Hour"].iloc[i + 1], df["Solar_Available"].iloc[i + 1])
                    optimizer.update(state, action, reward, next_state)
                joblib.dump(optimizer.q_table, f"q_table_{factory_name}.joblib")

                if params["model_type"] == "Prophet":
                    prophet_df = df[["Timestamp", "Intended_Heavy_On"]].rename(columns={"Timestamp": "ds", "Intended_Heavy_On": "y"})
                    future = heavy_model.make_future_dataframe(periods=24, freq="H")
                    heavy_forecast = heavy_model.predict(future)
                    df["Predicted_Heavy_On"] = heavy_forecast["yhat"][:len(df)].round().clip(0, params["num_heavy"])
                    prophet_df = df[["Timestamp", "Intended_Medium_On"]].rename(columns={"Timestamp": "ds", "Intended_Medium_On": "y"})
                    future = medium_model.make_future_dataframe(periods=24, freq="H")
                    medium_forecast = medium_model.predict(future)
                    df["Predicted_Medium_On"] = medium_forecast["yhat"][:len(df)].round().clip(0, params["num_medium"])
                else:
                    df["Predicted_Heavy_On"] = heavy_model.predict(df[["Shift", "Hour", "Day_of_Week"]]).round().clip(0, params["num_heavy"])
                    df["Predicted_Medium_On"] = medium_model.predict(df[["Shift", "Hour", "Day_of_Week"]]).round().clip(0, params["num_medium"])

                df["Optimized_Heavy_On"] = df["Predicted_Heavy_On"].copy()
                df["Optimized_Medium_On"] = df["Predicted_Medium_On"].copy()

                for i in range(len(df)):
                    state = optimizer.get_state(df["Shift"].iloc[i], df["Hour"].iloc[i], df["Solar_Available"].iloc[i])
                    action = optimizer.get_action(state, epsilon=0)
                    if action == "shift_heavy" and df["Solar_Available"].iloc[i] > 0:
                        df.loc[i, "Optimized_Heavy_On"] = min(df["Heavy_On"].iloc[i], df["Predicted_Heavy_On"].iloc[i] + 1)
                    elif action == "shift_medium" and df["Solar_Available"].iloc[i] > 0:
                        df.loc[i, "Optimized_Medium_On"] = min(df["Medium_On"].iloc[i], df["Predicted_Medium_On"].iloc[i] + 1)
                    elif action == "reduce_hvac" and df["HVAC_Inefficient"].iloc[i] == 1:
                        df.loc[i, "HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"].iloc[i] - 22, 0)

                df["Optimized_Energy_Heavy"] = df["Optimized_Heavy_On"] * params["energy_heavy"]
                df["Optimized_Energy_Medium"] = df["Optimized_Medium_On"] * params["energy_medium"]
                df["Optimized_HVAC_Energy"] = np.where(
                    df["HVAC_Inefficient"] == 0, df["HVAC_Energy"], 20 + 10 * np.maximum(df["Temperature"] - 22, 0)
                )
                df["Optimized_Lighting_Energy"] = np.where(df["Is_Working_Hours"] | (df["Lighting_Energy"] == 10), df["Lighting_Energy"], 10)
                df["Optimized_Total_Energy"] = df[["Optimized_Energy_Heavy", "Optimized_Energy_Medium", "Optimized_HVAC_Energy", "Optimized_Lighting_Energy"]].sum(axis=1)
                df["Optimized_Grid_Energy"] = np.maximum(0, df["Optimized_Total_Energy"] - df["Solar_Used"])
                df["Optimized_CO2_Emissions"] = df["Optimized_Grid_Energy"] * CONFIG["GRID_EMISSION_FACTOR"] + df["Solar_Used"] * CONFIG["SOLAR_EMISSION_FACTOR"]

                progress.progress(100)

                # Metrics
                baseline_energy = df["Total_Energy"].sum()
                optimized_energy = df["Optimized_Total_Energy"].sum()
                energy_savings = baseline_energy - optimized_energy
                savings_percent = (energy_savings / baseline_energy * 100) if baseline_energy > 0 else 0
                cost_savings = energy_savings * params["cost_per_kwh"]
                baseline_co2 = df["CO2_Emissions"].sum()
                optimized_co2 = df["Optimized_CO2_Emissions"].sum()
                co2_savings = baseline_co2 - optimized_co2
                trees_equivalent = (co2_savings / 1000) * CONFIG["TREES_PER_TON_CO2"]
                carbon_credit_earnings = (co2_savings / 1000) * CONFIG["CARBON_CREDIT_PRICE"]

                # Summary
                summary_data = {
                    "Baseline Energy (kWh)": f"{baseline_energy:.2f}",
                    "Optimized Energy (kWh)": f"{optimized_energy:.2f}",
                    "Energy Savings (kWh)": f"{energy_savings:.2f}",
                    "Savings (%)": f"{savings_percent:.2f}",
                    "Cost Savings ($)": f"{cost_savings:.2f}",
                    "Baseline CO2 (kg)": f"{baseline_co2:.2f}",
                    "Optimized CO2 (kg)": f"{optimized_co2:.2f}",
                    "CO2 Savings (kg)": f"{co2_savings:.2f}",
                    "Trees Equivalent": f"{trees_equivalent:.2f}",
                    "Carbon Credit Earnings ($)": f"{carbon_credit_earnings:.2f}"
                }
                st.subheader("Summary")
                st.table(summary_data)

                st.subheader("Model Performance")
                st.write(f"Heavy Machine MAE: {heavy_mae:.2f}")
                st.write(f"Medium Machine MAE: {medium_mae:.2f}")

                # Visualizations
                st.subheader("Savings Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=savings_percent,
                    title={"text": "Energy Savings (%)"}, gauge={"axis": {"range": [0, 100]}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.subheader("Energy Usage")
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(x=df["Timestamp"], y=df["Total_Energy"], name="Baseline", line=dict(color="red")))
                fig_energy.add_trace(go.Scatter(x=df["Timestamp"], y=df["Optimized_Total_Energy"], name="Optimized", line=dict(color="green")))
                fig_energy.update_layout(title="Energy Usage (kW)", yaxis_title="Energy (kW)")
                st.plotly_chart(fig_energy, use_container_width=True)

                st.subheader("Energy Breakdown")
                fig_breakdown = go.Figure()
                for col, name, color in [("Energy_Heavy", "Heavy Machines", "blue"), ("Energy_Medium", "Medium Machines", "orange"),
                                       ("HVAC_Energy", "HVAC", "green"), ("Lighting_Energy", "Lighting", "purple")]:
                    fig_breakdown.add_trace(go.Scatter(x=df["Timestamp"], y=df[col], name=name, stackgroup="one", line=dict(color=color)))
                fig_breakdown.update_layout(title="Baseline Energy Breakdown (kW)", yaxis_title="Energy (kW)")
                st.plotly_chart(fig_breakdown, use_container_width=True)

                st.subheader("3D Energy Usage")
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=df["Hour"], y=df["Day_of_Week"], z=df["Total_Energy"],
                    mode="markers", marker=dict(size=5, color=df["CO2_Emissions"], colorscale="Viridis")
                )])
                fig_3d.update_layout(title="3D Energy Usage by Hour and Day", scene=dict(xaxis_title="Hour", yaxis_title="Day of Week", zaxis_title="Energy (kW)"))
                st.plotly_chart(fig_3d)

                st.subheader("Daily Savings")
                df["Date"] = df["Timestamp"].dt.date
                daily_savings = df.groupby("Date").apply(lambda x: x["Total_Energy"].sum() - x["Optimized_Total_Energy"].sum()).reset_index(name="Savings")
                fig_savings = px.bar(daily_savings, x="Date", y="Savings", title="Daily Energy Savings (kWh)", color_discrete_sequence=["teal"])
                st.plotly_chart(fig_savings, use_container_width=True)

                st.subheader("Export")
                csv = df.to_csv(index=False)
                st.download_button("Download Data as CSV", csv, f"factory_energy_{factory_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                pdf_filename = f"factory_report_{factory_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                if generate_pdf_report(summary_data, pdf_filename):
                    with open(pdf_filename, "rb") as f:
                        st.download_button("Download Report as PDF", f, pdf_filename, "application/pdf")

    except Exception as e:
        st.error(f"Optimization error: {e}. Please check data integrity.")
        logger.error(f"Optimization error: {e}")

def render_carbon_tab():
    """Renders the carbon impact tab."""
    if "factories" not in st.session_state or not st.session_state["factories"]:
        st.warning("No simulation data available. Run a simulation first.")
        return
    factory_name = st.selectbox("Select Factory", list(st.session_state["factories"].keys()))
    df = st.session_state["factories"].get(factory_name)
    if df is None:
        st.error("No data for selected factory.")
        return

    st.header("Carbon Impact")
    if "CO2_Emissions" not in df.columns or "Optimized_CO2_Emissions" not in df.columns:
        st.error("Carbon data missing. Please re-run the simulation.")
        return

    try:
        baseline_co2 = df["CO2_Emissions"].sum()
        optimized_co2 = df["Optimized_CO2_Emissions"].sum()
        co2_savings = baseline_co2 - optimized_co2
        trees_equivalent = (co2_savings / 1000) * CONFIG["TREES_PER_TON_CO2"]

        st.subheader("Emissions Over Time")
        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(x=df["Timestamp"], y=df["CO2_Emissions"], name="Baseline", line=dict(color="red")))
        fig_co2.add_trace(go.Scatter(x=df["Timestamp"], y=df["Optimized_CO2_Emissions"], name="Optimized", line=dict(color="green")))
        fig_co2.update_layout(title="CO2 Emissions (kg)", yaxis_title="CO2 (kg)")
        st.plotly_chart(fig_co2, use_container_width=True)

        st.subheader("Emissions Breakdown")
        co2_breakdown = {
            "Heavy Machines": (df["Energy_Heavy"] * CONFIG["GRID_EMISSION_FACTOR"]).sum(),
            "Medium Machines": (df["Energy_Medium"] * CONFIG["GRID_EMISSION_FACTOR"]).sum(),
            "HVAC": (df["HVAC_Energy"] * CONFIG["GRID_EMISSION_FACTOR"]).sum(),
            "Lighting": (df["Lighting_Energy"] * CONFIG["GRID_EMISSION_FACTOR"]).sum()
        }
        fig_pie = px.pie(names=list(co2_breakdown.keys()), values=list(co2_breakdown.values()), title="Baseline CO2 Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Sustainability Impact")
        st.markdown(f"""
        - **CO2 Savings**: {co2_savings:.2f} kg
        - **Trees Equivalent**: {trees_equivalent:.2f} trees
        - **Carbon Credit Earnings**: ${(co2_savings / 1000) * CONFIG["CARBON_CREDIT_PRICE"]:.2f}
        - **Goal**: Supports UN SDG 7 (Affordable and Clean Energy) and 13 (Climate Action) by reducing industrial emissions.
        """)
    except Exception as e:
        st.error(f"Carbon impact error: {e}. Please check data integrity.")
        logger.error(f"Carbon impact error: {e}")

def render_documentation_tab():
    """Renders the documentation tab with pitch deck."""
    st.header("Documentation")
    st.markdown("""
    ### Factory Energy & Carbon Optimizer
    Optimizes factory energy use and carbon emissions using machine learning, solar integration, and IoT simulation.

    **Problem**:
    Factories contribute significantly to global emissions due to inefficient energy use. Reducing energy consumption and emissions is critical for sustainability.

    **Solution**:
    This app simulates factory operations, optimizes machine scheduling using ML (Random Forest, Linear Regression, Prophet) and Q-learning, and integrates solar energy to minimize grid reliance.

    **Features**:
    - Simulates energy consumption for multiple factories with inefficiencies.
    - Optimizes operations to reduce energy and CO2 emissions using ML and Q-learning.
    - Visualizes energy usage, savings, and emissions with interactive charts (including 3D plots and gauges).
    - Supports mock IoT streaming for real-time data updates.
    - Exports results as CSV and PDF reports.
    - Estimates carbon credit earnings.

    **Inputs**:
    - Machine counts and energy consumption.
    - Inefficiency probabilities for machines, HVAC, and lighting.
    - Simulation duration and cost per kWh.
    - Factory-specific configurations.

    **Outputs**:
    - Summary table with energy, cost, and CO2 savings.
    - Interactive visualizations for energy trends, daily savings, and emissions.
    - Downloadable CSV and PDF reports.

    **Sustainability Impact**:
    - Reduces CO2 emissions through solar optimization and efficient scheduling.
    - Aligns with UN SDG 7 & 13 by promoting clean energy and climate action.
    - Equivalent tree planting impact for CO2 savings.
    - Potential for carbon credit monetization.

    **Future Enhancements**:
    - Integration with real-time grid carbon intensity APIs.
    - Real IoT connectivity with SCADA systems.
    - Blockchain-based carbon credit tracking.
    - Scalability to multi-factory networks.

    **Pitch Deck**:
    - **Innovation**: Combines ML, Q-learning, and IoT simulation for sustainable factory operations.
    - **Impact**: Reduces energy costs and emissions, supporting global sustainability goals.
    - **Scalability**: Applicable to real-world factories with IoT and API integrations.
    - **Demo**: [Insert link to video demo hosted on YouTube]
    """)

def main():
    st.set_page_config(page_title="Factory Energy & Carbon Optimizer", layout="wide", initial_sidebar_state="expanded")
    st.title("Factory Energy & Carbon Optimizer")

    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Results", "Carbon Impact", "Documentation"])
    with tab1:
        render_simulation_tab()
    with tab2:
        render_results_tab()
    with tab3:
        render_carbon_tab()
    with tab4:
        render_documentation_tab()

if __name__ == "__main__":
    main()

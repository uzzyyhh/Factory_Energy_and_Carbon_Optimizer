import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)

GRID_EMISSION_FACTOR, SOLAR_EMISSION_FACTOR, TREES_PER_TON_CO2 = 0.5, 0.05, 48

def simulate_factory_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours=720):
    """Simulate factory energy data with inefficiencies."""
    try:
        df = pd.DataFrame({"Timestamp": pd.date_range("2025-06-01", periods=hours, freq="H")})
        df["Day_of_Week"], df["Hour"], df["Is_Weekday"] = df["Timestamp"].dt.dayofweek, df["Timestamp"].dt.hour, df["Day_of_Week"] < 5
        df["Shift"] = np.select([(df["Is_Weekday"] & df["Hour"].between(8, 15)), (df["Is_Weekday"] & df["Hour"].between(16, 23))], ["day", "night"], "overnight")
        df["Shift"] = np.where(~df["Is_Weekday"], "weekend", df["Shift"])
        shift_map = {"day": [num_heavy, num_medium], "night": [num_heavy//2, num_medium], "overnight": [0, num_medium//5], "weekend": [0, num_medium//5]}
        df[["Intended_Heavy_On", "Intended_Medium_On"]] = [df["Shift"].map({k: v[i] for k, v in shift_map.items()}) for i in range(2)]
        df[["Heavy_On", "Medium_On"]] = [df[f"Intended_{m}_On"] + np.random.binomial(num_heavy if m == "Heavy" else num_medium - df[f"Intended_{m}_On"], p_ineff) for m in ["Heavy", "Medium"]]
        df[["Energy_Heavy", "Energy_Medium"]] = [df[f"{m}_On"] * (energy_heavy if m == "Heavy" else energy_medium) for m in ["Heavy", "Medium"]]
        df["Temperature"] = 30 + 5 * np.sin(2 * np.pi * (df["Hour"] - 14) / 24) + np.random.normal(0, 1, len(df))
        df["HVAC_Inefficient"] = np.random.choice([0, 1], len(df), p=[1-q_ineff, q_ineff])
        df["HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - (22 - 2 * df["HVAC_Inefficient"]), 0)
        df["Is_Working_Hours"] = df["Is_Weekday"] & df["Hour"].between(8, 17)
        df["Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, np.where(np.random.random(len(df)) < r_ineff, 50, 10))
        df["Solar_Available"] = np.where(df["Hour"].between(6, 18), 100 * np.sin(np.pi * (df["Hour"] - 6) / 12), 0)
        df["Solar_Used"] = np.minimum(df["Solar_Available"], df[["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]].sum(axis=1))
        df["Grid_Energy"] = np.maximum(0, df[["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]].sum(axis=1) - df["Solar_Used"])
        df["Total_Energy"] = df[["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]].sum(axis=1)
        df["CO2_Emissions"] = df["Grid_Energy"] * GRID_EMISSION_FACTOR + df["Solar_Used"] * SOLAR_EMISSION_FACTOR
        df[["Efficient_Heavy_On", "Efficient_Medium_On"]] = df[["Intended_Heavy_On", "Intended_Medium_On"]]
        df[["Efficient_Energy_Heavy", "Efficient_Energy_Medium"]] = [df[f"Efficient_{m}_On"] * (energy_heavy if m == "Heavy" else energy_medium) for m in ["Heavy", "Medium"]]
        df["Efficient_HVAC_Energy"] = 20 + 10 * np.maximum(df["Temperature"] - 22, 0)
        df["Efficient_Lighting_Energy"] = np.where(df["Is_Working_Hours"], 50, 10)
        df["Efficient_Total_Energy"] = df[["Efficient_Energy_Heavy", "Efficient_Energy_Medium", "Efficient_HVAC_Energy", "Efficient_Lighting_Energy"]].sum(axis=1)
        df["Efficient_CO2_Emissions"] = df["Efficient_Total_Energy"] * GRID_EMISSION_FACTOR
        logger.info("Simulation completed with shape %s", df.shape)
        return df
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        st.error(f"Simulation failed: {e}")
        return None

class CarbonOptimizer:
    """Q-learning for carbon-aware scheduling."""
    def __init__(self): self.q_table, self.actions = {}, ["shift_heavy", "shift_medium", "no_action"]
    def get_state(self, shift, hour, solar): return (shift, hour, int(solar > 0))
    def get_action(self, state, epsilon=0.1): return np.random.choice(self.actions) if np.random.random() < epsilon else max(self.q_table.get(state, {a: 0 for a in self.actions}), key=lambda x: self.q_table.get(state, {a: 0 for a in self.actions})[x])
    def update(self, state, action, reward, next_state): self.q_table.setdefault(state, {a: 0 for a in self.actions}); self.q_table.setdefault(next_state, {a: 0 for a in self.actions}); self.q_table[state][action] += 0.1 * (reward + 0.9 * max(self.q_table[next_state].values(), default=0) - self.q_table[state][action])

@st.cache_data
def get_simulated_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours):
    return simulate_factory_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours)

def main():
    st.set_page_config(page_title="Factory Energy & Carbon Optimizer", layout="wide")
    st.title("Factory Energy & Carbon Optimizer")

    if st.sidebar.button("Clear Cache"): st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Results", "Carbon Impact", "Documentation"])

    with tab1:
        st.header("Factory Setup")
        col1, col2 = st.columns(2)
        with col1: num_heavy, energy_heavy, num_medium, energy_medium = [st.number_input(f"Number of {m} Machines", 1, value=v if "Heavy" in m else v2, step=1 if "Number" in m else 0.5) for m, v, v2 in [("Heavy", 5, 20.0), ("Medium", 10, 10.0)]]
        with col2: p_ineff, q_ineff, r_ineff = [st.slider(f"{c} Inefficiency Probability", 0.0, 1.0, v, 0.01) for c, v in [("Machine", 0.1), ("HVAC", 0.2), ("Lighting", 0.1)]]; hours = {"1 Week (168h)": 168, "1 Month (720h)": 720, "3 Months (2160h)": 2160}[st.selectbox("Duration", list({"1 Week (168h)": 168, "1 Month (720h)": 720, "3 Months (2160h)": 2160}.keys()), 1)]; cost_per_kwh = st.number_input("Cost per kWh ($)", 0.0, value=0.15, step=0.01); model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])
        if st.button("Run Simulation"):
            with st.spinner("Simulating..."):
                df = get_simulated_data(num_heavy, energy_heavy, num_medium, energy_medium, p_ineff, q_ineff, r_ineff, hours)
                if df is not None:
                    st.session_state.update(df=df, params={"num_heavy": num_heavy, "energy_heavy": energy_heavy, "num_medium": num_medium, "energy_medium": energy_medium, "cost_per_kwh": cost_per_kwh})
                    st.session_state["show_success"] = st.session_state["switch_to_results"] = True
                else: st.error("Simulation failed. Check logs.")

        if st.session_state.get("show_success"): st.success("Simulation completed!"); st.session_state["show_success"] = False; st.rerun()
        if st.session_state.get("switch_to_results"):
            st.components.v1.html("<script>document.addEventListener('DOMContentLoaded',function(){const t=document.querySelectorAll('button[role=\"tab\"]');t[1]?.click()});</script>", height=0)
            st.session_state["switch_to_results"] = False

    with tab2:
        if "df" in st.session_state and "params" in st.session_state:
            df, params = st.session_state["df"], st.session_state["params"]
            if all(col in df.columns for col in ["Shift", "Hour", "Solar_Available", "CO2_Emissions", "Intended_Heavy_On", "Intended_Medium_On", "Day_of_Week", "Heavy_On", "Medium_On", "Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy", "Temperature", "HVAC_Inefficient", "Is_Working_Hours", "Solar_Used", "Grid_Energy"]):
                with st.spinner("Optimizing..."):
                    optimizer = CarbonOptimizer()
                    for i in range(min(100, len(df)-1)): 
                        state = optimizer.get_state(*df.iloc[i][["Shift", "Hour", "Solar_Available"]])
                        action = optimizer.get_action(state)
                        reward = -df["CO2_Emissions"].iloc[i]
                        next_state = optimizer.get_state(*df.iloc[i+1][["Shift", "Hour", "Solar_Available"]])
                        optimizer.update(state, action, reward, next_state)
                    
                    df["Optimized_Heavy_On"] = df["Intended_Heavy_On"].clip(0, params["num_heavy"])
                    df["Optimized_Medium_On"] = df["Intended_Medium_On"].clip(0, params["num_medium"])
                    
                    for i in range(len(df)):
                        state = optimizer.get_state(*df.iloc[i][["Shift", "Hour", "Solar_Available"]])
                        action = optimizer.get_action(state, 0)
                        if action == "shift_heavy" and df["Solar_Available"].iloc[i] > 0:
                            df.loc[i, "Optimized_Heavy_On"] = min(df["Heavy_On"].iloc[i], df["Optimized_Heavy_On"].iloc[i] + 1)
                        elif action == "shift_medium" and df["Solar_Available"].iloc[i] > 0:
                            df.loc[i, "Optimized_Medium_On"] = min(df["Medium_On"].iloc[i], df["Optimized_Medium_On"].iloc[i] + 1)
                    
                    df["Optimized_Energy_Heavy"], df["Optimized_Energy_Medium"] = df["Optimized_Heavy_On"] * params["energy_heavy"], df["Optimized_Medium_On"] * params["energy_medium"]
                    df["Optimized_HVAC_Energy"] = np.where(df["HVAC_Inefficient"] == 0, df["HVAC_Energy"], 20 + 10 * np.maximum(df["Temperature"] - 22, 0))
                    df["Optimized_Lighting_Energy"] = np.where(df["Is_Working_Hours"] | (df["Lighting_Energy"] == 10), df["Lighting_Energy"], 10)
                    df["Optimized_Total_Energy"] = df[["Optimized_Energy_Heavy", "Optimized_Energy_Medium", "Optimized_HVAC_Energy", "Optimized_Lighting_Energy"]].sum(axis=1)
                    df["Optimized_Grid_Energy"] = np.maximum(0, df["Optimized_Total_Energy"] - df["Solar_Used"])
                    df["Optimized_CO2_Emissions"] = df["Optimized_Grid_Energy"] * GRID_EMISSION_FACTOR + df["Solar_Used"] * SOLAR_EMISSION_FACTOR

                baseline_energy, optimized_energy = df["Total_Energy"].sum(), df["Optimized_Total_Energy"].sum()
                energy_savings, savings_percent = baseline_energy - optimized_energy, (energy_savings / baseline_energy * 100) if baseline_energy else 0
                cost_savings = energy_savings * params["cost_per_kwh"]
                baseline_co2, optimized_co2 = df["CO2_Emissions"].sum(), df["Optimized_CO2_Emissions"].sum()
                co2_savings, trees_equivalent = baseline_co2 - optimized_co2, (co2_savings / 1000) * TREES_PER_TON_CO2

                st.subheader("Summary")
                st.table({"Metric": ["Baseline Energy (kWh)", "Optimized Energy (kWh)", "Energy Savings (kWh)", "Savings (%)", "Cost Savings ($)", "Baseline CO2 (kg)", "Optimized CO2 (kg)", "CO2 Savings (kg)", "Trees Equivalent"], "Value": [f"{v:.2f}" for v in [baseline_energy, optimized_energy, energy_savings, savings_percent, cost_savings, baseline_co2, optimized_co2, co2_savings, trees_equivalent]]})

                st.subheader("Energy Usage")
                st.plotly_chart(go.Figure([go.Scatter(x=df["Timestamp"], y=df[c], name=c, line=dict(color="red" if c == "Total_Energy" else "green")) for c in ["Total_Energy", "Optimized_Total_Energy"]], layout={"title": "Energy Usage (kW)", "yaxis_title": "Energy (kW)"}), use_container_width=True)

                st.subheader("Energy Breakdown")
                st.plotly_chart(go.Figure([go.Scatter(x=df["Timestamp"], y=df[c], name=c.split("_")[0], stackgroup="one", line=dict(color={"Heavy": "blue", "Medium": "orange", "HVAC": "green", "Lighting": "purple"}[c.split("_")[0]])) for c in ["Energy_Heavy", "Energy_Medium", "HVAC_Energy", "Lighting_Energy"]], layout={"title": "Baseline Energy Breakdown (kW)", "yaxis_title": "Energy (kW)"}), use_container_width=True)

                st.subheader("Daily Savings")
                df["Date"] = df["Timestamp"].dt.date
                st.plotly_chart(px.bar(df.groupby("Date").apply(lambda x: x["Total_Energy"].sum() - x["Optimized_Total_Energy"].sum()).reset_index(name="Savings"), x="Date", y="Savings", title="Daily Energy Savings (kWh)", color_discrete_sequence=["teal"]), use_container_width=True)

                st.subheader("Export")
                st.download_button("Download Data as CSV", df.to_csv(index=False), f"factory_energy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

    with tab3:
        if "df" in st.session_state:
            df = st.session_state["df"]
            if "CO2_Emissions" in df.columns and "Optimized_CO2_Emissions" in df.columns:
                baseline_co2, optimized_co2 = df["CO2_Emissions"].sum(), df["Optimized_CO2_Emissions"].sum()
                co2_savings, trees_equivalent = baseline_co2 - optimized_co2, (co2_savings / 1000) * TREES_PER_TON_CO2

                st.subheader("Emissions Over Time")
                st.plotly_chart(go.Figure([go.Scatter(x=df["Timestamp"], y=df[c], name=c, line=dict(color="red" if c == "CO2_Emissions" else "green")) for c in ["CO2_Emissions", "Optimized_CO2_Emissions"]], layout={"title": "CO2 Emissions (kg)", "yaxis_title": "CO2 (kg)"}), use_container_width=True)

                st.subheader("Emissions Breakdown")
                co2_breakdown = {k: (df[f"Energy_{k}"] * GRID_EMISSION_FACTOR).sum() for k in ["Heavy", "Medium", "HVAC", "Lighting"]}
                st.plotly_chart(px.pie(names=list(co2_breakdown.keys()), values=list(co2_breakdown.values()), title="Baseline CO2 Breakdown"), use_container_width=True)

                st.subheader("Sustainability Impact")
                st.markdown(f"- **CO2 Savings**: {co2_savings:.2f} kg\n- **Trees Equivalent**: {trees_equivalent:.2f} trees\n- **Goal**: Supports UN SDG 7 & 13")

    with tab4:
        st.header("Documentation")
        st.markdown("""
        ### Factory Energy & Carbon Optimizer
        Optimizes energy and carbon using solar and efficiency.

        **Features**: Simulates energy with inefficiencies, optimizes with Q-learning, visualizes savings, exports CSV.

        **Inputs**: Machine counts, energy use, inefficiency probabilities, duration, cost.

        **Outputs**: Energy and CO2 savings, plots, data export.

        **Sustainability**: Reduces emissions with solar, scalable to IoT.

        **Future**: Real-time grid data, IoT, carbon credits.
        """)

if __name__ == "__main__":
    main()

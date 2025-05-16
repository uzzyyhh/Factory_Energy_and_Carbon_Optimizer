import pytest
import pandas as pd
from app import simulate_factory_data

def test_simulate_factory_data():
    df = simulate_factory_data(5, 20, 10, 10, 0.1, 0.2, 0.1, 24)
    assert df is not None, "Simulation failed to return a DataFrame"
    assert df.shape[0] == 24, "Incorrect number of rows"
    assert "CO2_Emissions" in df.columns, "CO2_Emissions column missing"
    assert "Total_Energy" in df.columns, "Total_Energy column missing"
    assert (df["Total_Energy"] >= 0).all(), "Negative energy values detected"

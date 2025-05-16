# Factory Energy & Carbon Optimizer

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

An AI-driven tool for simulating and optimizing factory energy consumption and carbon emissions using machine learning and reinforcement learning.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Approach](#technical-approach)
- [Sustainability Impact](#sustainability-impact)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

The Factory Energy & Carbon Optimizer is a Streamlit web application that:
- Simulates factory energy consumption with realistic inefficiencies
- Uses machine learning (Random Forest/Linear Regression) to predict optimal machine schedules
- Applies Q-learning to minimize carbon emissions by shifting loads to solar-heavy hours
- Provides visualizations of energy usage and carbon impact
- Supports UN Sustainable Development Goals (SDGs) 7 (Clean Energy) and 13 (Climate Action)

## Features

- **Realistic Simulation** of:
  - Heavy and medium machinery
  - HVAC systems with temperature dependencies
  - Lighting systems
  - Solar energy availability
  - Operational shifts (day/night/weekend)

- **AI/ML Optimization**:
  - Machine learning models for energy prediction
  - Reinforcement learning for carbon-aware scheduling
  - Inefficiency detection and correction

- **Interactive Visualizations**:
  - Energy usage over time
  - Component breakdowns
  - CO2 emissions comparisons
  - Daily savings metrics

- **Export Capabilities**:
  - CSV data export
  - DOCX documentation generation

## Installation

### 1. Clone the repository:
```
git clone https://github.com/uzzyyhh/Factory_Energy_and_Carbon_Optimizer.git
cd Factory_Energy_and_Carbon_Optimizer
```
### Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install dependencies:
```
pip install -r requirements.txt
```
### Usage

#### Run the Streamlit application:
```
streamlit run app.py
```

### Configure your factory parameters in the web interface:
- Number and energy consumption of machines
- Inefficiency probabilities
- Simulation duration
- Cost per kWh
- Model type

### View results across four tabs:
- Simulation: Configure and run simulations
- Results: View optimization metrics and energy visualizations
- Carbon Impact: Analyze CO2 reductions and sustainability impact

### Documentation: Access detailed system documentation

- Technical Approach
- Data Simulation

### Generates synthetic factory data with:

- Time-based features (hour, day of week)
- Shift patterns
- Machine inefficiencies
- Temperature-dependent HVAC usage
- Solar availability curves
- Machine Learning
- Random Forest/Linear Regression models predict:
- Optimal number of active machines
- Energy consumption patterns
- Features include shift type, hour, and day of week
- Optimization
  
### Q-learning agent with:

- States: (shift, hour, solar availability)
- Actions: Shift heavy/medium machines or no action
- Reward: Negative CO2 emissions
- Learning rate: 0.1, discount factor: 0.9

### Sustainability Impact:
- Energy Efficiency: Typically achieves 5-20% energy reductions
- Carbon Reduction: Can reduce CO2 emissions by 5,000-15,000 kg over 3 months
- nEquivalent to planting 240-720 trees annually

### Supports UN SDGs:
- SDG 7: Affordable and Clean Energy
- SDG 13: Climate Action

### Future Improvements
- Add seasonal variation models
- Integrate real-time IoT data
- Implement deep reinforcement learning
- Add multi-factory optimization
- Incorporate carbon credit tracking

### License
- This project is licensed under the MIT License - see the LICENSE file for details.


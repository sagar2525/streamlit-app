# NexGen Logistics Intelligence System

## 🚀 Project Overview
This project is a **Predictive Delivery & Customer Experience Intelligence System** designed for NexGen Logistics. It moves operations from reactive fire-fighting to proactive decision-making.

**Key Capabilities:**
1.  **Predictive Risk**: Forecasts delivery delays *before* they occur using ML (Random Forest).
2.  **Customer Intelligence**: Identifies at-risk customers and correlates delays with satisfaction.
3.  **Prescriptive Actions**: Recommends specific operational interventions (e.g., "Assign Newer Vehicle", "Escalate Priority").

## 📂 Project Structure
```
project/
│── app.py                # Main Streamlit Application
│── train_models.py       # Script to train ML models
│── requirements.txt      # Dependencies
│── data/                 # Raw CSV datasets
│── models/               # Saved ML models (.joblib)
│── utils/
│   │── data_loader.py    # Data ingestion
│   │── feature_eng.py    # Feature engineering pipeline
│   │── model_utils.py    # Model training & inference logic
│   │── decision_logic.py # Business rules engine
```

## 🛠️ Setup & Usage

### 1. Installation
Ensure Python 3.8+ is installed.
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
(Optional if models/ directory is already populated)
```bash
python train_models.py
```
This script will:
- Load data from `data/`
- Build the analytical master dataset
- Train Delay Prediction & Customer Risk models
- Save artifacts to `models/`

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

## 🧠 Model Logic
- **Delay Prediction**: Uses Route Risk (Traffic/Weather), Distance, and Vehicle Suitability to predict probability of delay.
- **Customer Risk**: Assesses risk based on Customer Segment history and current order experience.
- **Decision Engine**: Rules-based layer that translates Risk Probabilities into Business Actions (e.g., If Delay Risk > 60% AND Critical Customer -> Recommend Standard Priority Escalation).

## 📊 Dashboard Pages
1.  **ℹ️ About**: Project context, problem statement, and solution approach.
2.  **📊 Executive Overview**: High-level KPIs and Cost/Risk aggregation.
3.  **🔮 Predictive Risk**: Detailed view of orders predicted to be delayed.
4.  **😊 Customer Experience**: Analysis of dissatisfaction risk and ratings.
5.  **⭐ Operational Control Tower**: The central action hub with Simulator and Batch Operations.

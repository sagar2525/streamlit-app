import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def preprocess_for_modeling(master_df):
    """
    Encodes categorical variables and handles missing values for modeling.
    """
    df = master_df.copy()
    
    # Select features for Delay Prediction
    # Target: is_delayed
    # Features: Distance, Priority, Route Risk, Vehicle Score, Origin, Destination, Traffic, Weather
    
    # Select features for Customer Risk
    # Target: customer_dissatisfaction_risk
    # Features: Priority, Segment Stats, Delay (historical/actual for training), Order Value
    
    # Handle Categoricals
    le = LabelEncoder()
    cat_cols = ['Priority', 'Origin', 'Destination', 'Product_Category', 'Customer_Segment', 'Weather_Impact']
    
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            # fillna first
            df[col] = df[col].fillna('Unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Fill remaining numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df, encoders

def train_delay_model(df):
    """
    Trains valid Random Forest model for Delay Prediction.
    """
    target = 'is_delayed'
    features = [
        'Distance_KM', 'route_risk_score', 'vehicle_suitability_score', 
        'Traffic_Delay_Minutes', 'Priority', 'Origin', 'Product_Category'
    ]
    
    # Filter only rows where we have the target (Delivery Performance data exists)
    train_df = df.dropna(subset=[target])
    
    X = train_df[features]
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("--- Delay Prediction Model ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, importances

def train_customer_risk_model(df):
    """
    Trains Gradient Boosting model for Customer Dissatisfaction Risk.
    """
    target = 'customer_dissatisfaction_risk'
    features = [
        'segment_avg_rating', 'segment_recommend_pct', 'delay_days', 
        'Priority', 'Order_Value_INR'
    ]
    
    # Filter rows with feedback
    train_df = df.dropna(subset=[target])
    
    X = train_df[features]
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print("--- Customer Risk Model ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

def save_artifacts(delay_model, risk_model, encoders):
    joblib.dump(delay_model, os.path.join(MODELS_DIR, 'delay_model.joblib'))
    joblib.dump(risk_model, os.path.join(MODELS_DIR, 'risk_model.joblib'))
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.joblib'))
    print("Models and encoders saved.")

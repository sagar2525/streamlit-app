import pandas as pd
import numpy as np

def recommend_action(row):
    """
    Generates rule-based operational recommendations.
    Input: Row with [delay_prob, route_risk, vehicle_score, customer_risk]
    Output: {Action, Reason, Expected_Impact}
    """
    actions = []
    
    # Extract key metrics
    # Note: 'delay_prob' come from model inference
    delay_prob = row.get('delay_probability', 0) * 100
    route_risk = row.get('route_risk_score', 0)
    vehicle_score = row.get('vehicle_suitability_score', 0)
    is_critical_cust = row.get('customer_dissatisfaction_risk', 0)
    
    # RULE 1: High Delay Probability + Critical Customer -> Urgent
    if delay_prob > 60 and is_critical_cust:
        return {
            "Action": "Escalate to Express & Prioritize",
            "Reason": "High delay risk for at-risk customer",
            "Cost_Impact": "High (+20%)",
            "Svc_Impact": "Significant Risk Reduction"
        }
        
    # RULE 2: High Route Risk -> Review Route
    if route_risk > 70:
         return {
            "Action": "Re-route / Monitor Traffic",
            "Reason": "Severe weather or traffic detected",
            "Cost_Impact": "Neutral",
            "Svc_Impact": "Avoid Potential 4hr+ Delay"
        }

    # RULE 3: Low Vehicle Score + Moderate Delay Risk -> Upgrade Vehicle
    if vehicle_score < 40 and delay_prob > 40:
        return {
            "Action": "Reassign to Newer Vehicle",
            "Reason": "Vehicle suitability is low for this lane",
            "Cost_Impact": "Medium (+5%)",
            "Svc_Impact": "Improve Reliability"
        }
        
    # RULE 4: High Customer Risk (Historical) -> Proactive Comm
    if is_critical_cust:
        return {
            "Action": "Proactive Status Update",
            "Reason": "Customer has history of dissatisfaction",
            "Cost_Impact": "Low",
            "Svc_Impact": "Trust Building"
        }

    return {
        "Action": "Standard Dispatch",
        "Reason": "Risk within acceptable limits",
        "Cost_Impact": "None",
        "Svc_Impact": "Standard SLA"
    }

def apply_decision_logic(df):
    """
    Applies logic to entire dataframe.
    """
    recommendations = df.apply(recommend_action, axis=1)
    rec_df = pd.DataFrame(recommendations.tolist())
    return pd.concat([df.reset_index(drop=True), rec_df], axis=1)

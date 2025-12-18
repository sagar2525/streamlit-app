import pandas as pd
import numpy as np

def create_delivery_features(delivery_df):
    """
    Calculates delivery delay and delay status.
    """
    df = delivery_df.copy()
    df['delay_days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)
    return df

def create_route_features(routes_df):
    """
    Calculates route risk score based on traffic and weather.
    """
    df = routes_df.copy()
    
    # Normalize inputs for risk score (simple heuristic)
    # Traffic: 0-120 mins -> 0-1
    traffic_score = df['Traffic_Delay_Minutes'] / df['Traffic_Delay_Minutes'].max()
    
    # Weather: Map categorical to numeric risk
    weather_map = {'Clear': 0, 'Rain': 0.5, 'Storm': 1.0, 'Fog': 0.7, 'Cloudy': 0.2} 
    # Fallback to 0 if unknown
    weather_risk = df['Weather_Impact'].map(weather_map).fillna(0)
    
    # Composite Risk Score (0-100 scale)
    df['route_risk_score'] = ((traffic_score * 0.6) + (weather_risk * 0.4)) * 100
    return df

def create_vehicle_features(fleet_df, delivery_df):
    """
    Aggregates vehicle features by Type and maps to orders via Delivery Carrier.
    """
    # 1. Aggregate fleet stats by Vehicle_Type
    type_stats = fleet_df.groupby('Vehicle_Type').agg({
        'Age_Years': 'mean',
        'Fuel_Efficiency_KM_per_L': 'mean',
        'Capacity_KG': 'mean',
        'CO2_Emissions_Kg_per_KM': 'mean'
    }).reset_index()
    
    # 2. Create Suitability Score (Inverse of Age, Proportional to Efficiency)
    # Simple heuristic: Higher efficiency & lower age = better score
    # Normalize Age (invert): 0-10 years
    age_score = 1 - (type_stats['Age_Years'] / 15).clip(upper=1)
    eff_score = type_stats['Fuel_Efficiency_KM_per_L'] / type_stats['Fuel_Efficiency_KM_per_L'].max()
    
    type_stats['vehicle_suitability_score'] = ((age_score + eff_score) / 2) * 100
    
    # 3. Map to Delivery DF
    # Issue: Delivery has 'Carrier' (e.g. GlobalTransit), Fleet has 'Vehicle_Type' (e.g. Large_Truck)
    # We need a mapping. Since this is a prototype, we will infer/assign a mapping based on inspection or random assignment
    # ACTUAL DATA SAYS: Carrier = ['GlobalTransit', 'Speedy', 'EcoDeliver', 'ReliableExpress']
    # FLEET SAYS: Type = ['Refrigerated', 'Small_Van', 'Large_Truck', 'Medium_Truck', 'Express_Bike']
    
    # Heuristic Mapping
    carrier_map = {
        'GlobalTransit': 'Large_Truck',
        'Speedy': 'Express_Bike',
        'EcoDeliver': 'Small_Van',
        'ReliableExpress': 'Medium_Truck'
    }
    
    delivery_df_mod = delivery_df.copy()
    delivery_df_mod['Vehicle_Type_Mapped'] = delivery_df_mod['Carrier'].map(carrier_map).fillna('Medium_Truck')
    
    # Now merge on Mapped Type
    vehicle_map = delivery_df_mod[['Order_ID', 'Vehicle_Type_Mapped']]
    vehicle_features = vehicle_map.merge(type_stats, left_on='Vehicle_Type_Mapped', right_on='Vehicle_Type', how='left')
    
    return vehicle_features.drop(columns=['Vehicle_Type_Mapped', 'Vehicle_Type'])


def create_customer_features(feedback_df, orders_df):
    """
    Creates customer risk profile based on Customer_Segment historical performance.
    Since we lack Customer_ID on Orders, we assume risk is segment-based 
    OR we infer customer risk from the specific order's feedback if available (for training)
    and segment averages for future prediction.
    
    For this prototype: We will enable 'Order Level' risk targets.
    """
    # feedback_df has Order_ID, Rating, Issue_Category
    # Merge feedback to orders to get Segment info
    merged = pd.merge(orders_df[['Order_ID', 'Customer_Segment']], feedback_df, on='Order_ID', how='left')
    
    # Calculate Segment-Level Risk stats
    segment_stats = merged.groupby('Customer_Segment').agg({
        'Rating': 'mean',
        'Would_Recommend': lambda x: (x == 'Yes').mean() * 100
    }).rename(columns={'Rating': 'segment_avg_rating', 'Would_Recommend': 'segment_recommend_pct'}).reset_index()
    
    # Merge segment stats back to orders
    merged = merged.merge(segment_stats, on='Customer_Segment', how='left')

    # Calculate specific Order dissatisfaction (Target variable for Risk Model)
    # Low Rating (<3) = High Dissatisfaction
    merged['customer_dissatisfaction_risk'] = (merged['Rating'] <= 3).astype(int)
    
    # Return both the per-order target and the segment features
    return merged[['Order_ID', 'customer_dissatisfaction_risk', 'segment_avg_rating', 'segment_recommend_pct']]

def create_cost_features(cost_df):
    """
    Aggregates total cost per order and preserves components.
    """
    df = cost_df.copy()
    cost_cols = [c for c in df.columns if 'Cost' in c or 'Fee' in c or 'Insurance' in c or 'Overhead' in c]
    df['total_cost'] = df[cost_cols].sum(axis=1)
    # Return Order_ID, total_cost, and all component columns
    return_cols = ['Order_ID', 'total_cost'] + cost_cols
    # Ensure no duplicates if Order_ID was in cost_cols (unlikely but safe)
    return_cols = list(set(return_cols)) 
    return df[return_cols]

def build_master_dataset(data_dict):
    """
    Orchestrates the feature engineering and merging.
    """
    orders = data_dict['orders']
    
    # 1. Base Features
    # Convert dates to day of week, month
    orders['order_dow'] = orders['Order_Date'].dt.dayofweek
    orders['order_month'] = orders['Order_Date'].dt.month
    
    # 2. Delivery Features
    del_feat = create_delivery_features(data_dict['delivery'])
    
    # 3. Route Features
    route_feat = create_route_features(data_dict['routes'])
    
    # 4. Vehicle Features
    veh_feat = create_vehicle_features(data_dict['fleet'], data_dict['delivery'])
    
    # 5. Customer Features
    cust_feat = create_customer_features(data_dict['feedback'], orders)
    
    # 6. Cost Features
    cost_feat = create_cost_features(data_dict['costs'])
    
    # --- MERGE ALL ---
    master = orders.merge(del_feat, on='Order_ID', how='left')
    master = master.merge(route_feat, on='Order_ID', how='left')
    master = master.merge(veh_feat, on='Order_ID', how='left')
    master = master.merge(cust_feat, on='Order_ID', how='left')
    master = master.merge(cost_feat, on='Order_ID', how='left')
    
    # Fill risks for new orders (simulated) with 0 or mean
    master['route_risk_score'] = master['route_risk_score'].fillna(master['route_risk_score'].median())
    master['vehicle_suitability_score'] = master['vehicle_suitability_score'].fillna(master['vehicle_suitability_score'].mean())
    
    return master

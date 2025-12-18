import pandas as pd
import os

DATA_DIR = 'data'

def load_dataset(filename):
    """Loads a CSV dataset from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def load_all_data():
    """
    Loads all 7 logistics datasets and returns them as a dictionary.
    Keys: orders, delivery, routes, fleet, warehouse, feedback, costs
    """
    data = {}
    datasets = {
        'orders': 'orders.csv',
        'delivery': 'delivery_performance.csv',
        'routes': 'routes_distance.csv',
        'fleet': 'vehicle_fleet.csv',
        'warehouse': 'warehouse_inventory.csv',
        'feedback': 'customer_feedback.csv',
        'costs': 'cost_breakdown.csv'
    }

    for key, filename in datasets.items():
        try:
            data[key] = load_dataset(filename)
            # Standardize date columns if applicable
            for col in data[key].columns:
                if 'Date' in col:
                    data[key][col] = pd.to_datetime(data[key][col], errors='coerce')
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            data[key] = None
            
    return data

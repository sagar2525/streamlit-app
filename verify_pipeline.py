import pandas as pd
from utils.data_loader import load_all_data
from utils.feature_eng import build_master_dataset

def verify():
    print("Loading data...")
    try:
        data = load_all_data()
        for k, v in data.items():
            if v is not None:
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: FAILED TO LOAD")
        
        print("\nBuilding master dataset...")
        master = build_master_dataset(data)
        print(f"Master Dataset Shape: {master.shape}")
        print("Columns:", list(master.columns))
        
        # Check for Critical Features
        critical_cols = ['is_delayed', 'route_risk_score', 'vehicle_suitability_score', 'customer_dissatisfaction_risk', 'total_cost']
        missing = [c for c in critical_cols if c not in master.columns]
        
        if missing:
            print(f"FAILED: Missing critical columns: {missing}")
        else:
            print("SUCCESS: All critical features created.")
            print(master[critical_cols].describe())
            
        # Check for NaNs in critical columns
        nans = master[critical_cols].isna().sum()
        if nans.sum() > 0:
            print("\nWARNING: NaNs found in critical columns:")
            print(nans[nans > 0])
        
        # Save for inspection
        master.to_csv('master_debug.csv', index=False)
        print("\nSaved 'master_debug.csv' for inspection.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()

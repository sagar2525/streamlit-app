from utils.data_loader import load_all_data
from utils.feature_eng import build_master_dataset
from utils.model_utils import preprocess_for_modeling, train_delay_model, train_customer_risk_model, save_artifacts

def main():
    print("1. Loading Data...")
    data = load_all_data()
    
    print("2. Building Master Dataset...")
    master = build_master_dataset(data)
    
    print("3. Preprocessing...")
    processed_df, encoders = preprocess_for_modeling(master)
    
    print("4. Training Delay Model...")
    delay_model, importances = train_delay_model(processed_df)
    print("\nTop Features (Delay):\n", importances.head())
    
    print("\n5. Training Customer Risk Model...")
    risk_model = train_customer_risk_model(processed_df)
    
    print("\n6. Saving Artifacts...")
    save_artifacts(delay_model, risk_model, encoders)
    
if __name__ == "__main__":
    main()

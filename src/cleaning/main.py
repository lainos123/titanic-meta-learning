from data_loader import load_and_combine_data, export_cleaned_data
from feature_engineering import (
    extract_name_features, 
    extract_cabin_features, 
    create_family_features
)
from missing_value_handler import (
    fill_missing_to_median, 
    fill_missing_to_mode
)
from normalisation import normalise_features
from target_encoding import (
    encode_name_features,
    encode_cabin_features
)
from age_preds import predict_missing_ages

def clean_data(df):
    """Main function to clean and process the data."""
    print("\n=== Starting Data Cleaning Pipeline ===")
    
    # Basic preprocessing
    print("\nPerforming basic preprocessing:")
    print("- Processing Embarked column...")
    df = fill_missing_to_mode(df, column='Embarked')
    
    # Initial feature engineering (needed for age prediction)
    print("\nExtracting features needed for age prediction:")
    print("- Extracting name features...")
    df = extract_name_features(df)
    
    # Age prediction
    print("\nPredicting missing ages...")
    df = predict_missing_ages(df)
    
    # Remaining feature engineering
    print("\nPerforming remaining feature engineering:")
    print("- Extracting cabin features...")
    df = extract_cabin_features(df)
    print("- Creating family features...")
    df = create_family_features(df)
    
    # Target encoding
    print("\nPerforming target encoding:")
    print("- Encoding name-derived features...")
    df = encode_name_features(df)
    print("- Encoding cabin features...")
    df = encode_cabin_features(df)
    
    # Final preprocessing
    print("\nPerforming final preprocessing:")
    columns_to_fill = ['Fare', 'Title_encoded', 'LastName_encoded']
    print(f"- Filling missing values with median for columns: {', '.join(columns_to_fill)}")
    df = fill_missing_to_median(df, columns_to_fill)
    print("- Dropping high cardinality columns (Ticket)")
    df = df.drop(['Ticket'], axis=1)
    
    print("\n✓ Data cleaning completed successfully")
    return df

def process_data():
    """Main function to run the entire data processing pipeline."""
    print("\n=== Starting Data Processing Pipeline ===")
    
    # Load data
    print("\nLoading and combining data...")
    combined = load_and_combine_data()
    print(f"- Loaded {len(combined)} total rows")
    
    # Process data
    combined = clean_data(combined)
    
    print("\nNormalising features...")
    combined = normalise_features(combined)
    
    # Export results
    print("\nExporting processed data...")
    train_df, test_df = export_cleaned_data(combined)
    print(f"- Exported {len(train_df)} training samples")
    print(f"- Exported {len(test_df)} test samples")
    
    print("\n✓ Data processing pipeline completed successfully")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = process_data() 
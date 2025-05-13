import pandas as pd
import os

def load_and_combine_data(train_path='./data/raw/train.csv', test_path='./data/raw/test.csv'):
    """Load train and test datasets and combine them into a single dataframe."""
    print(f"- Loading training data from {train_path}")
    train = pd.read_csv(train_path) 
    print(f"- Loading test data from {test_path}")
    test = pd.read_csv(test_path)
    
    print("- Adding source indicators")
    train['source'] = 'train'
    test['source'] = 'test'
    
    print("- Combining datasets")
    return pd.concat([train, test], sort=False)

def export_cleaned_data(combined_df, output_dir='./data/processed'):
    """Export the cleaned data into separate train and test CSV files."""
    print(f"- Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("- Splitting into train and test sets")
    train = combined_df[combined_df['source'] == 'train'].drop(columns=['source'])
    test = combined_df[combined_df['source'] == 'test'].drop(columns=['source'])
    
    train_path = f'{output_dir}/train.csv'
    test_path = f'{output_dir}/test.csv'
    print(f"- Saving training data to {train_path}")
    train.to_csv(train_path, index=False)
    print(f"- Saving test data to {test_path}")
    test.to_csv(test_path, index=False)
    
    return train, test 
import pandas as pd

def encode_categorical_by_target(df, categorical_col, target_col='Survived'):
    """
    Encode a categorical column using the mean of the target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_col (str): Name of categorical column to encode
        target_col (str): Name of target column to use for encoding
    """
    encoded_col_name = f"{categorical_col}_encoded"
    n_categories = df[categorical_col].nunique()
    print(f"  Encoding {n_categories} unique values in {categorical_col}")
    df[encoded_col_name] = df.groupby(categorical_col)[target_col].transform("mean")
    return df

def encode_name_features(df):
    """Encode name-derived features using target encoding."""
    df = encode_categorical_by_target(df, 'LastName')
    df = encode_categorical_by_target(df, 'Title')
    df = df.drop(['LastName', 'Title', 'Name'], axis=1)
    return df

def encode_cabin_features(df):
    """Encode cabin features using target encoding."""
    df['Cabin_letter'] = df['Cabin_letter'].fillna('Missing')
    df = encode_categorical_by_target(df, 'Cabin_letter')
    df = df.drop(['Cabin_letter'], axis=1)
    return df 
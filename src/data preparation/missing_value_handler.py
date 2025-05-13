def fill_missing_to_mode(df, column):
    """
    Fill missing values of specified column with the most common value.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    """
    n_missing = df[column].isna().sum()
    if n_missing > 0:
        print(f"  Found {n_missing} missing values in {column}")
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
        print(f"  Filled with mode value: {mode_value}")
    return df

def fill_missing_to_median(df, columns_to_fill):
    """
    Fill missing values in specified columns with their respective medians.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_fill (list): List of columns to fill with median values
    """
    for col in columns_to_fill:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  Filling {n_missing} missing values in {col}")
            df[col] = df[col].fillna(df[col].median())
    return df 
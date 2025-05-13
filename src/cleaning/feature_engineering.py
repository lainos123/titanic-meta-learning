import pandas as pd
import numpy as np

def extract_name_features(df):
    """Extract title and last name features from the Name column."""
    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0]
    df['LastName'] = df['Name'].str.split(',').str[0].str.split('.').str[0]
    return df

def extract_cabin_features(df):
    """Extract cabin information."""
    def extract_cabin_info(cabin_entry):
        if pd.isna(cabin_entry):
            return cabin_entry  # Just return nan
        cabins = cabin_entry.split()
        first_cabin = cabins[0]
        cabin_letter = first_cabin[0]
        return cabin_letter  # Return just the letter, not a Series
    
    print("  Processing cabin information...")
    df['Cabin_letter'] = df['Cabin'].apply(extract_cabin_info)
    print(f"  Found {df['Cabin_letter'].nunique()} unique cabin letters")
    df = df.drop(['Cabin'], axis=1)
    return df

def create_family_features(df):
    """Create and process family-related features."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df 
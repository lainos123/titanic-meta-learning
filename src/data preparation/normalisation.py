from sklearn.preprocessing import StandardScaler

def normalise_features(df, cols_to_scale=None):
    """Normalise specified numerical features using StandardScaler."""
    if cols_to_scale is None:
        cols_to_scale = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 
                        'LastName_encoded', 'Title_encoded', 'Cabin_letter_encoded']
    print(f"- Normalising columns: {', '.join(cols_to_scale)}")
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df 
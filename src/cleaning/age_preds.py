# lets use a Gradient Boosting Regressor to predict the missing values of age 
# using columns Pclass, Sex, SibSp, Parch, Embarked, Title.

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

def predict_missing_ages(df):
    """
    Predict missing age values in the dataset using XGBoost.
    
    Args:
        df: pandas DataFrame containing the features needed for age prediction
            (must have columns: Age, Pclass, Sex, SibSp, Parch, Embarked, Title, Fare)
    
    Returns:
        DataFrame with predicted ages filled in
    """
    # Create a copy of the data we'll use for prediction
    df_ageboost = df.copy()
    age_data = df_ageboost[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Fare']].copy()

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_emb = LabelEncoder()
    le_title = LabelEncoder()

    age_data['Sex'] = le_sex.fit_transform(age_data['Sex'])
    age_data['Embarked'] = le_emb.fit_transform(age_data['Embarked'])
    age_data['Title'] = le_title.fit_transform(age_data['Title'])

    # Split into two sets: known age and unknown age
    known_age = age_data[age_data['Age'].notnull()]
    unknown_age = age_data[age_data['Age'].isnull()]

    # Prepare the training data
    X_train = known_age.drop('Age', axis=1)
    y_train = known_age['Age']

    # Train XGBoost model
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_reg.fit(X_train, y_train)

    # Predict missing ages
    X_test = unknown_age.drop('Age', axis=1)
    predicted_ages = xgb_reg.predict(X_test)

    # Fill the missing values in the original dataframe
    df_ageboost.loc[df_ageboost['Age'].isnull(), 'Age'] = predicted_ages

    # Print some statistics about the predictions
    print("\nMissing values after prediction:")
    print(df_ageboost['Age'].isnull().sum())
    print("\nAge statistics after imputation:")
    print(df_ageboost['Age'].describe().round(2))

    return df_ageboost

# Example usage:
# df_with_predicted_ages = predict_missing_ages(df)


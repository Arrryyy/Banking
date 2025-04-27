import pandas as pd
from sklearn.preprocessing import StandardScaler
import os 

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, sep=';')  # <-- important fix

    # Step 2: Drop 'duration' (high leakage, not realistic for real prediction)
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    # Step 3: Encode categorical variables
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Step 4: Handle special variables
    df['pdays_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
    df = df.drop(columns=['pdays'])  # Drop original pdays after feature creation

    # Step 5: Scale numeric features
    numeric_cols = ['age', 'campaign', 'previous', 
                    'emp.var.rate', 'cons.price.idx', 
                    'cons.conf.idx', 'euribor3m', 'nr.employed']

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 6: Encode target
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Step 7: Ensure output folder exists, then save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved at {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/bank-additional-full.csv"
    output_path = "data/processed/bank_processed.csv"
    preprocess_data(input_path, output_path)
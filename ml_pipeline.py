import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Huntington_Disease_Dataset.csv')

features = ['Age', 'HTT_CAG_Repeat_Length', 'Chorea_Score', 'Brain_Volume_Loss',
            'Functional_Capacity', 'HTT_Gene_Expression_Level', 'Protein_Aggregation_Level']

# Build a severity score: higher CAG, Chorea, Brain loss, Protein = worse
# Higher Functional_Capacity = better (inverse)
df['severity_score'] = (
    (df['HTT_CAG_Repeat_Length'] - 35) / 45 +
    df['Chorea_Score'] / 10 +
    (df['Brain_Volume_Loss'] - 2) / 6.5 +
    df['Protein_Aggregation_Level'] / 5 +
    (100 - df['Functional_Capacity']) / 100
)

# Assign stages based on score percentiles
p33 = df['severity_score'].quantile(0.33)
p66 = df['severity_score'].quantile(0.66)

df['Stage'] = pd.cut(df['severity_score'],
                     bins=[-np.inf, p33, p66, np.inf],
                     labels=[0, 1, 2]).astype(int)

print("Class distribution:\n", df['Stage'].value_counts())

X = df[features]
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000))
])

pipeline.fit(X_train, y_train)

print("Accuracy:", pipeline.score(X_test, y_test))
joblib.dump(pipeline, 'final_model.pkl')
print("Model saved.")

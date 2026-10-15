import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
import joblib

def generate_water_data(n_samples=2000):
    np.random.seed(42)
    
    hardness = np.random.normal(196, 33, n_samples)
    solids = np.random.normal(22000, 8700, n_samples)
    chloramines = np.random.normal(7.12, 1.58, n_samples)
    sulfate = np.random.normal(333, 41, n_samples)
    conductivity = np.random.normal(421, 80, n_samples)
    organic_carbon = np.random.normal(14.2, 3.3, n_samples)
    trihalomethanes = np.random.normal(66.3, 16.1, n_samples)
    turbidity = np.random.normal(3.96, 0.78, n_samples)
    
    # Target Task B: pH value
    ph = 7.0 + 0.005 * hardness - 0.00001 * solids - 0.05 * organic_carbon + np.random.normal(0, 0.3, n_samples)
    ph = np.clip(ph, 0.0, 14.0)
    
    # Target Task A: Potability (Safe = 1, Unsafe = 0)
    potability_score = (
        (chloramines > 5.0) & (chloramines < 9.0) &
        (sulfate < 380) &
        (turbidity < 4.5) &
        (ph >= 6.5) & (ph <= 8.5)
    ).astype(int)
    
    # Introduce random noise / imbalance
    noise = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    potability = np.abs(potability_score - noise)
    
    df = pd.DataFrame({
        'hardness': hardness,
        'solids': solids,
        'chloramines': chloramines,
        'sulfate': sulfate,
        'conductivity': conductivity,
        'organic_carbon': organic_carbon,
        'trihalomethanes': trihalomethanes,
        'turbidity': turbidity,
        'ph': ph,
        'potability': potability
    })
    return df

print("Generating dataset and training pipelines...")
df = generate_water_data()

X = df.drop(columns=['ph', 'potability'])
y_class = df['potability']
y_reg = df['ph']

# Pipeline preprocessing
imputer = KNNImputer(n_neighbors=5)
scaler = RobustScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Train Task A (Classification - XGBoost)
clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
clf.fit(X_scaled, y_class)

# Train Task B (Regression - LightGBM)
reg = LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
reg.fit(X_scaled, y_reg)

# Save Pipeline Artifacts
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(clf, 'clf_model.joblib')
joblib.dump(reg, 'reg_model.joblib')

print("Models successfully trained and saved!")
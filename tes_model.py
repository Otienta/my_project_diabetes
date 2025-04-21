import joblib
import numpy as np

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Modèle et scaler chargés")
except Exception as e:
    print(f"Erreur de chargement : {e}")
    exit(1)

# Valeurs que vous avez utilisées (ajustez selon ce que vous avez entré)
test_data = np.array([[6, 180, 70, 30, 100, 35.0, 0.8, 50]])
test_data_scaled = scaler.transform(test_data)
prediction = model.predict(test_data_scaled)[0]
probability = model.predict_proba(test_data_scaled)[0][1]
result = 'Diabétique' if prediction == 1 else 'Non diabétique'
print(f"Résultat : {result}, Probabilité : {probability * 100:.2f}%")
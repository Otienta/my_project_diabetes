import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Charger le dataset
data = pd.read_csv('diabetes.csv')

# Afficher les premières lignes
print("Aperçu des données :")
print(data.head())

# Afficher les informations
print("\nInformations sur le dataset :")
print(data.info())

# Vérifier les valeurs manquantes explicites
print("\nValeurs manquantes explicites :")
print(data.isnull().sum())

# Vérifier les valeurs à 0
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nNombre de zéros dans les colonnes :")
for col in columns_to_check:
    print(f"{col}: {len(data[data[col] == 0])}")

# Répartition de la cible
print("\nRépartition de la cible (Outcome) :")
print(data['Outcome'].value_counts(normalize=True))

# Prétraitement : Remplacer les zéros par la médiane
for col in columns_to_check:
    data[col] = data[col].replace(0, data[col].median())

# Séparer features et cible
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluer le modèle
accuracy = model.score(X_test_scaled, y_test)
print(f"\nAccuracy sur le test set : {accuracy:.2f}")

# Sauvegarder le modèle et le scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modèle et scaler sauvegardés sous model.pkl et scaler.pkl")
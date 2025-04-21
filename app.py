from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'diabetes_prediction_key'

# Load the model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Users table (doctors)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    # Patients table
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER,
        patient_name TEXT NOT NULL,
        pregnancies REAL,
        glucose REAL,
        bloodpressure REAL,
        skinthickness REAL,
        insulin REAL,
        bmi REAL,
        pedigree REAL,
        age REAL,
        result TEXT,
        probability REAL,
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

init_db()

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('This username already exists.', 'danger')
        finally:
            conn.close()
    return render_template('signup.html')

# Login route
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['logged_in'] = True
            session['user_id'] = user[0]  # Store doctor's ID
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Incorrect username or password.', 'danger')
    return render_template('login.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            patient_name = request.form['patient_name']
            features = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['bloodpressure']),
                float(request.form['skinthickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['pedigree']),
                float(request.form['age'])
            ]
            if any(x < 0 for x in features):
                flash('Values cannot be negative.', 'danger')
                return render_template('predict.html')
            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            result = 'Diabetic' if prediction == 1 else 'Non-diabetic'
            
            # Store patient data
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute('''INSERT INTO patients (doctor_id, patient_name, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, pedigree, age, result, probability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (session['user_id'], patient_name, *features, result, probability * 100))
            conn.commit()
            conn.close()
            
            # Prepare data for display
            patient_data = {
                'Patient Name': patient_name,
                'Pregnancies': features[0],
                'Glucose': features[1],
                'Blood Pressure': features[2],
                'Skin Thickness': features[3],
                'Insulin': features[4],
                'BMI': features[5],
                'Diabetes Pedigree': features[6],
                'Age': features[7],
                'Result': result,
                'Diabetes Probability (%)': f"{probability * 100:.2f}"
            }
            flash('Data saved successfully!', 'success')
            return render_template('predict.html', patient_data=patient_data)
        except ValueError:
            flash('Please enter valid numeric values.', 'danger')
    
    return render_template('predict.html')

# Patients list route
@app.route('/patients', methods=['GET'])
def patients():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM patients WHERE doctor_id = ?', (session['user_id'],))
    patients = c.fetchall()
    conn.close()
    
    return render_template('patients.html', patients=patients)

# Edit patient route
@app.route('/edit_patient/<int:patient_id>', methods=['GET', 'POST'])
def edit_patient(patient_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    if request.method == 'POST':
        try:
            patient_name = request.form['patient_name']
            features = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['bloodpressure']),
                float(request.form['skinthickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['pedigree']),
                float(request.form['age'])
            ]
            if any(x < 0 for x in features):
                flash('Values cannot be negative.', 'danger')
                return redirect(url_for('edit_patient', patient_id=patient_id))
            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_array)[0][1]
            result = 'Diabetic' if prediction == 1 else 'Non-diabetic'
            
            c.execute('''UPDATE patients SET patient_name = ?, pregnancies = ?, glucose = ?, bloodpressure = ?, skinthickness = ?, insulin = ?, bmi = ?, pedigree = ?, age = ?, result = ?, probability = ?
                        WHERE id = ? AND doctor_id = ?''',
                     (patient_name, *features, result, probability * 100, patient_id, session['user_id']))
            conn.commit()
            flash('Data updated successfully!', 'success')
            return redirect(url_for('patients'))
        except ValueError:
            flash('Please enter valid numeric values.', 'danger')
    
    c.execute('SELECT * FROM patients WHERE id = ? AND doctor_id = ?', (patient_id, session['user_id']))
    patient = c.fetchone()
    conn.close()
    if not patient:
        flash('Patient not found or unauthorized.', 'danger')
        return redirect(url_for('patients'))
    
    return render_template('edit_patient.html', patient=patient)

# Delete patient route
@app.route('/delete_patient/<int:patient_id>')
def delete_patient(patient_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('DELETE FROM patients WHERE id = ? AND doctor_id = ?', (patient_id, session['user_id']))
    conn.commit()
    conn.close()
    flash('Patient deleted successfully.', 'success')
    return redirect(url_for('patients'))

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT NOT NULL,
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        bmi REAL,
        diabetes_pedigree REAL,
        age INTEGER,
        prediction TEXT,
        probability REAL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

init_db()  # Appeler au démarrage

from flask import Flask, request, jsonify
from flask_cors import CORS
import pennylane as qml
from pennylane import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

print("="*60)
print("LOADING QML MODEL")
print("="*60)

# Load the trained model
print("Loading model weights...")
if os.path.exists('models/qml_diabetes_weights_improved.npy'):
    weights = np.load('models/qml_diabetes_weights_improved.npy')
    print("✓ Weights loaded")
else:
    print("❌ ERROR: models/qml_diabetes_weights_improved.npy not found!")
    print("   Please run train_qml_model.py first")
    exit(1)

print("Loading decision threshold...")
if os.path.exists('models/decision_threshold.npy'):
    DECISION_THRESHOLD = float(np.load('models/decision_threshold.npy')[0])
    print(f"✓ Threshold loaded: {DECISION_THRESHOLD}")
else:
    DECISION_THRESHOLD = -0.15
    print(f"⚠️ Using default threshold: {DECISION_THRESHOLD}")

print("Loading scaler...")
if os.path.exists('models/scaler.pkl'):
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
else:
    print("❌ ERROR: models/scaler.pkl not found!")
    exit(1)

# Define quantum device and circuit - MUST MATCH TRAINING!
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def quantum_circuit(weights, x):
    """Quantum circuit - MATCHES training architecture exactly"""
    # Data encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    
    # 5 variational layers (IMPORTANT: matches training!)
    n_layers = 5
    
    for layer in range(n_layers):
        # Rotation gates with trainable parameters
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
            qml.RX(weights[layer, i, 2], wires=i)  # RX rotation
        
        # Entangling gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Final rotation layer
    for i in range(n_qubits):
        qml.RY(weights[n_layers, i, 0], wires=i)
    
    return qml.expval(qml.PauliZ(0))

print("✓ Quantum circuit ready")

def get_medicine_recommendations(patient_data, has_diabetes):
    """Generate medicine recommendations"""
    if not has_diabetes:
        return []
    
    medicines = []
    glucose = patient_data['glucose']
    bmi = patient_data['bmi']
    age = patient_data['age']
    
    # Metformin - First line treatment
    if glucose < 200 and bmi < 35:
        medicines.append({
            "name": "Metformin",
            "dosage": "500-850 mg",
            "frequency": "2-3 times daily with meals",
            "notes": "First-line medication. Reduces glucose production in liver.",
            "mechanism": "Biguanide - decreases hepatic glucose production"
        })
    
    # Glipizide for moderate to high glucose
    if glucose > 140:
        medicines.append({
            "name": "Glipizide (Glucotrol)",
            "dosage": "5-10 mg",
            "frequency": "Once daily, 30 minutes before breakfast",
            "notes": "Stimulates insulin secretion. Monitor for hypoglycemia.",
            "mechanism": "Sulfonylurea - stimulates pancreatic insulin release"
        })
    
    # SGLT2 inhibitors for weight management
    if bmi > 30:
        medicines.append({
            "name": "Empagliflozin (Jardiance)",
            "dosage": "10-25 mg",
            "frequency": "Once daily in the morning",
            "notes": "Helps with weight loss and cardiovascular protection.",
            "mechanism": "SGLT2 inhibitor - increases urinary glucose excretion"
        })
    
    # DPP-4 inhibitors for elderly
    if age > 60 or glucose > 200:
        medicines.append({
            "name": "Sitagliptin (Januvia)",
            "dosage": "100 mg",
            "frequency": "Once daily",
            "notes": "Low risk of hypoglycemia. Well-tolerated in elderly.",
            "mechanism": "DPP-4 inhibitor - increases incretin levels"
        })
    
    # GLP-1 agonist for severe cases
    if glucose > 250 or bmi > 35:
        medicines.append({
            "name": "Semaglutide (Ozempic)",
            "dosage": "0.5-1 mg",
            "frequency": "Once weekly injection",
            "notes": "Significant weight loss. Injectable medication.",
            "mechanism": "GLP-1 agonist - enhances insulin secretion"
        })
    
    return medicines

def get_lifestyle_recommendations(has_diabetes, patient_data):
    """Generate lifestyle recommendations"""
    recommendations = []
    
    if has_diabetes:
        recommendations.extend([
            "Monitor blood glucose regularly (fasting and 2 hours after meals)",
            "Target: Fasting 80-130 mg/dL, Post-meal < 180 mg/dL",
            "Follow low glycemic index diet, limit refined carbs",
            "Exercise 150 minutes/week (brisk walking, swimming)",
            "Lose 5-10% body weight if overweight",
            "Daily foot inspection and proper footwear",
            "Annual dilated eye examination",
            "HbA1c testing every 3 months"
        ])
        
        if patient_data['bmi'] > 30:
            recommendations.append("Priority: Weight loss - consult a dietitian")
        
        if patient_data['bloodPressure'] > 130:
            recommendations.append("Limit sodium to 2,300 mg/day")
            
    else:
        recommendations.extend([
            "Maintain healthy weight (BMI 18.5-24.9)",
            "Regular physical activity: 30 minutes daily",
            "Balanced diet: whole grains, vegetables, lean proteins",
            "Annual diabetes screening recommended",
            "Manage stress with relaxation techniques",
            "Get 7-9 hours of sleep per night"
        ])
    
    return recommendations

def get_risk_level(risk_score):
    """Determine risk level based on score"""
    if risk_score < 0.3:
        return {"level": "Low", "color": "green", "message": "Low risk of diabetes"}
    elif risk_score < 0.5:
        return {"level": "Moderate", "color": "yellow", "message": "Moderate risk - monitoring recommended"}
    elif risk_score < 0.7:
        return {"level": "High", "color": "orange", "message": "High risk - medical consultation advised"}
    else:
        return {"level": "Very High", "color": "red", "message": "Very high risk - immediate medical attention recommended"}

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        
        # Extract features in correct order
        features_order = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness',
                         'insulin', 'bmi', 'diabetesPedigreeFunction', 'age']
        
        # Validate input
        for feature in features_order:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
        
        # Prepare input
        x = np.array([[float(data[f]) for f in features_order]])
        x_scaled = scaler.transform(x)
        
        # Make prediction using quantum circuit
        output = float(quantum_circuit(weights, x_scaled[0]))
        
        # Convert to binary prediction using threshold
        has_diabetes = bool(output > DECISION_THRESHOLD)
        
        # Calculate probability (0 to 1 scale)
        probability = (output + 1) / 2
        
        # Calculate confidence
        if has_diabetes:
            confidence = (output - DECISION_THRESHOLD) / (1.0 - DECISION_THRESHOLD)
        else:
            confidence = (DECISION_THRESHOLD - output) / (DECISION_THRESHOLD - (-1.0))
        
        confidence = float(max(0.5, min(1.0, confidence)))
        
        # Calculate risk score
        risk_score = float(probability)
        risk_info = get_risk_level(risk_score)
        
        # Get recommendations
        medicines = get_medicine_recommendations(data, has_diabetes)
        lifestyle = get_lifestyle_recommendations(has_diabetes, data)
        
        # Prepare response
        response = {
            "hasDiabetes": has_diabetes,
            "confidence": confidence,
            "riskScore": risk_score,
            "probability": probability,
            "quantumOutput": output,
            "threshold": DECISION_THRESHOLD,
            "riskLevel": risk_info,
            "medicines": medicines,
            "lifestyle": lifestyle,
            "patientData": data,
            "modelInfo": {
                "version": "improved_v2",
                "layers": 5,
                "qubits": 8,
                "parameters": weights.size,
                "threshold": DECISION_THRESHOLD,
                "optimizedFor": "High sensitivity"
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy", 
        "model": "QML Diabetes Predictor",
        "version": "2.0",
        "threshold": DECISION_THRESHOLD,
        "layers": 5,
        "qubits": 8
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model information"""
    info = {
        "modelType": "Quantum Machine Learning",
        "framework": "PennyLane",
        "version": "2.0",
        "qubits": n_qubits,
        "layers": 5,
        "parameters": weights.size,
        "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        "decisionThreshold": DECISION_THRESHOLD,
        "architecture": "5 layers with RY+RZ+RX rotations",
        "improvements": [
            "Class-weighted loss (2x penalty for false negatives)",
            "Optimized decision threshold (-0.15)",
            "200 training epochs",
            "Adaptive learning rate (cosine annealing)",
            "RX rotation gates for better expressiveness"
        ],
        "expectedPerformance": {
            "accuracy": "70-72%",
            "sensitivity": "78-82%",
            "specificity": "63-68%"
        }
    }
    return jsonify(info), 200

if __name__ == '__main__':
    print("="*60)
    print("QML DIABETES PREDICTION API SERVER")
    print("="*60)
    print(f"✓ Model loaded with {n_qubits} qubits, 5 layers")
    print(f"✓ Decision threshold: {DECISION_THRESHOLD}")
    print(f"✓ Starting server on http://localhost:5000")
    print("="*60)
    print("\nAPI Endpoints:")
    print("  POST /predict      - Make predictions")
    print("  GET  /health       - Health check")
    print("  GET  /model-info   - Model information")
    print("\nPress CTRL+C to stop the server")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("="*60)
print(" QML DIABETES PREDICTOR - TRAINING")
print("="*60)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load the diabetes dataset
print("\n[1/9] Loading dataset...")
df = pd.read_csv('data/diabetes.csv')

# Prepare features and target
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

print(f" Dataset loaded: {len(X)} samples, {X.shape[1]} features")
print(f" Class distribution: {sum(y==0)} No Diabetes, {sum(y==1)} Diabetes")

# Split the data
print("\n[2/9] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# Normalize features
print("\n[3/9] Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to -1 and 1 for quantum processing
y_train_quantum = 2 * y_train - 1
y_test_quantum = 2 * y_test - 1

print("✓ Features normalized")

# Define quantum device
n_qubits = 8  # One qubit per feature
dev = qml.device("default.qubit", wires=n_qubits)

print(f"\n[4/9] Setting up improved quantum circuit...")
print(f"✓ Quantum device: {n_qubits} qubits")

#  More layers for better expressiveness
@qml.qnode(dev, interface="autograd")
def quantum_circuit(weights, x):
    """
    IMPROVED Variational quantum circuit with more layers
    """
    # Data encoding: Amplitude encoding with repetition
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    
    # Increased to 5 layers for better capacity
    n_layers = 5
    
    for layer in range(n_layers):
        # Rotation gates with trainable parameters
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
            qml.RX(weights[layer, i, 2], wires=i)  # ADDED: RX rotation
        
        # Entangling gates - ring topology
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Additional rotation layer
    for i in range(n_qubits):
        qml.RY(weights[n_layers, i, 0], wires=i)
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Initialize weights - IMPROVED: More parameters per layer
n_layers = 5
weight_shape = (n_layers + 1, n_qubits, 3)  # 3 rotations per qubit
weights = np.random.randn(*weight_shape, requires_grad=True) * 0.05  # Smaller init

print(f" Circuit parameters: {weights.size} trainable weights")
print(f" Architecture: {n_layers} layers with RY+RZ+RX rotations")

# IMPROVED: Class-weighted loss function to penalize false negatives
def cost_function(weights, X_batch, y_batch):
    """
    Weighted loss - penalizes false negatives (missing diabetes) more heavily
    """
    predictions = np.array([quantum_circuit(weights, x) for x in X_batch])
    pred_array = np.stack(predictions)
    
    # Calculate errors
    errors = pred_array - y_batch
    
    #  Weight the loss based on true label
    # Diabetes cases (y=1, y_quantum=1) get 2x weight to reduce false negatives
    weights_class = np.where(y_batch > 0, 2.0, 1.0)
    
    weighted_loss = np.sum(weights_class * errors ** 2) / len(y_batch)
    return weighted_loss

#  Adjustable threshold for prediction
DECISION_THRESHOLD = -0.15  # Lower threshold = more sensitive to diabetes

def predict(weights, X, threshold=DECISION_THRESHOLD):
    """
    Make predictions with adjustable threshold
    Negative threshold makes model more sensitive to diabetes
    """
    raw_predictions = []
    for x in X:
        output = quantum_circuit(weights, x)
        raw_predictions.append(float(output))
    
    raw_predictions = np.array(raw_predictions)
    #  Use custom threshold instead of 0
    binary_predictions = (raw_predictions > threshold).astype(int)
    
    return binary_predictions, raw_predictions

# Training parameters 
n_epochs = 200  
batch_size = 32
initial_learning_rate = 0.015  
min_learning_rate = 0.001

# Learning rate scheduler
def get_learning_rate(epoch, initial_lr, min_lr, total_epochs):
    """Cosine annealing learning rate"""
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))

# Training history
train_losses = []
train_accuracies = []
test_accuracies = []
learning_rates = []

print(f"\n[5/9]  quantum model...")
print(f" Epochs: {n_epochs} ")
print(f"Batch size: {batch_size}")
print(f" Initial learning rate: {initial_learning_rate}")
print(f" Decision threshold: {DECISION_THRESHOLD}")
print(f" Class weighting: 2x penalty for false negatives")
print("-"*60)

# Training loop with adaptive learning rate
for epoch in range(n_epochs):
    # IMPROVED: Adaptive learning rate
    current_lr = get_learning_rate(epoch, initial_learning_rate, min_learning_rate, n_epochs)
    opt = qml.AdamOptimizer(stepsize=current_lr)
    learning_rates.append(current_lr)
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train_quantum[indices]
    
    epoch_loss = 0
    n_batches = len(X_train_scaled) // batch_size
    
    # Mini-batch training
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Update weights
        weights, loss = opt.step_and_cost(
            lambda w: cost_function(w, X_batch, y_batch),
            weights
        )
        
        epoch_loss += float(loss)
    
    # Calculate average loss
    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)
    
    # Calculate accuracies every 10 epochs
    if (epoch + 1) % 10 == 0:
        train_pred, _ = predict(weights, X_train_scaled[:100])
        train_acc = accuracy_score(y_train[:100], train_pred)
        train_accuracies.append(train_acc)
        
        test_pred, _ = predict(weights, X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.5f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print("-"*60)
print(" Training complete!")

# Final evaluation with threshold optimization
print("\n[6/9] Evaluating model and finding optimal threshold...")

# Get raw predictions for ROC curve
_, y_pred_train_raw = predict(weights, X_train_scaled, threshold=-999)
y_pred_test_proba, y_pred_test_raw = predict(weights, X_test_scaled, threshold=-999)

# Convert raw predictions to probabilities
y_pred_train_proba = (y_pred_train_raw + 1) / 2
y_pred_test_proba = (y_pred_test_raw + 1) / 2

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_test_proba)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (Youden's index)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_prob = thresholds_roc[optimal_idx]
# Convert back to quantum output scale
optimal_threshold = 2 * optimal_threshold_prob - 1

print(f" ROC AUC Score: {roc_auc:.4f}")
print(f" Optimal threshold (Youden): {optimal_threshold:.4f}")
print(f" Using threshold: {DECISION_THRESHOLD:.4f} (prioritizes sensitivity)")

# Final predictions with our chosen threshold
y_pred_train, _ = predict(weights, X_train_scaled)
y_pred_test, _ = predict(weights, X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n✓ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"✓ Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_test, 
                            target_names=['No Diabetes', 'Diabetes'],
                            labels=[0, 1]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives:  {tn} ")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn} ")
print(f"  True Positives:  {tp})")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\nKey Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"  Specificity:          {specificity:.4f} ({specificity*100:.1f}%)")
print(f"  Precision:            {precision:.4f} ({precision*100:.1f}%)")

# Save the model with threshold
print("\n[7/9] Saving improved model...")
weights_to_save = np.array(weights)
np.save('models/qml_diabetes_weights_improved.npy', weights_to_save)
np.save('models/decision_threshold.npy', np.array([DECISION_THRESHOLD]))
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" weights saved: models/qml_diabetes_weights_improved.npy")
print(" Decision threshold saved: models/decision_threshold.npy")
print(" Scaler saved: models/scaler.pkl")

# Generate comprehensive visualizations
print("\n[8/9] Creating visualizations...")
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_losses, linewidth=2, color='#4F46E5')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy Comparison
ax2 = fig.add_subplot(gs[0, 1])
epochs_tracked = [i * 10 for i in range(len(train_accuracies))]
ax2.plot(epochs_tracked, train_accuracies, label='Train', 
         linewidth=2, marker='o', color='#10B981')
ax2.plot(epochs_tracked, test_accuracies, label='Test', 
         linewidth=2, marker='s', color='#EF4444')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Learning Rate Schedule
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(learning_rates, linewidth=2, color='#F59E0B')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Adaptive Learning Rate', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar_kws={'label': 'Count'}, ax=ax4, annot_kws={'size': 14})
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 5: ROC Curve
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(fpr, tpr, color='#8B5CF6', linewidth=3, label=f'ROC (AUC = {roc_auc:.3f})')
ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax5.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
            label=f'Optimal Point', zorder=5)
ax5.set_xlabel('False Positive Rate', fontsize=12)
ax5.set_ylabel('True Positive Rate', fontsize=12)
ax5.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: Metrics Comparison
ax6 = fig.add_subplot(gs[1, 2])
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision']
values = [test_accuracy, sensitivity, specificity, precision]
colors = ['#10B981', '#EF4444', '#3B82F6', '#F59E0B']
bars = ax6.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax6.set_ylim([0, 1])
ax6.grid(True, alpha=0.3, axis='y')
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 7: Prediction Distribution
ax7 = fig.add_subplot(gs[2, :])
ax7.hist(y_pred_test_proba[y_test == 0], bins=30, alpha=0.6, 
         label='No Diabetes', color='#10B981', edgecolor='black')
ax7.hist(y_pred_test_proba[y_test == 1], bins=30, alpha=0.6, 
         label='Diabetes', color='#EF4444', edgecolor='black')
ax7.axvline((DECISION_THRESHOLD + 1) / 2, color='purple', linestyle='--', 
            linewidth=3, label=f'Decision Threshold = {DECISION_THRESHOLD:.3f}')
ax7.set_xlabel('Predicted Probability', fontsize=12)
ax7.set_ylabel('Frequency', fontsize=12)
ax7.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax7.legend(fontsize=11)
ax7.grid(True, alpha=0.3)

plt.savefig('outputs/qml_improved_results.png', dpi=300, bbox_inches='tight')
print("✓ Comprehensive visualization saved: outputs/qml_improved_results.png")

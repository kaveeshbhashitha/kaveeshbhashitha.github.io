import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Modern ML models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional: For Transformer models
# from transformers import AutoModel, AutoTokenizer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Enhanced Data Preparation
def load_and_preprocess_data(filepath):
    """Load and preprocess the tea sensor dataset with more feature engineering"""
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    # Color features
    df['color_intensity'] = df[['color_r', 'color_g', 'color_b']].mean(axis=1)
    df['color_variation'] = df[['color_r', 'color_g', 'color_b']].std(axis=1)
    
    # Sensor features
    sensor_cols = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
    df['sensor_avg'] = df[sensor_cols].mean(axis=1)
    df['sensor_std'] = df[sensor_cols].std(axis=1)
    df['sensor_range'] = df[sensor_cols].max(axis=1) - df[sensor_cols].min(axis=1)
    
    # Interaction features
    df['color_sensor_interaction'] = df['color_intensity'] * df['sensor_avg']
    
    # Split features and target
    X = df.drop(['region'], axis=1)
    y = df['region']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

# 2. Modern Machine Learning Models
def train_ml_models(X_train, X_test, y_train, y_test):
    """Train and evaluate modern ML models"""
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',  # Faster training
            enable_categorical=True
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            boosting_type='gbdt'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred)
        }
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    
    return results

# 3. PyTorch Neural Network
class TeaClassifier(nn.Module):
    """Modern neural network architecture with dropout and batch normalization"""
    def __init__(self, input_size, num_classes):
        super(TeaClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def train_dl_model(X_train, y_train, X_val, y_val, input_size, num_classes):
    """Train the PyTorch model with early stopping"""
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TeaClassifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    patience = 5
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                early_stop = True
                break
        
        if early_stop:
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, best_val_acc

# 4. Main Workflow
def main():
    # Load data
    X, y, label_encoder = load_and_preprocess_data('tea_sensor_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ML models
    print("Training Machine Learning Models...")
    ml_results = train_ml_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Train DL model
    print("\nTraining Deep Learning Model...")
    input_size = X_train_scaled.shape[1]
    num_classes = len(np.unique(y))
    dl_model, dl_acc = train_dl_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        input_size, num_classes
    )
    
    print(f"\nDeep Learning Model Accuracy: {dl_acc:.4f}")
    
    # Save models and artifacts
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save the best ML model
    best_ml_name = max(ml_results.items(), key=lambda x: x[1]['accuracy'])[0]
    joblib.dump(ml_results[best_ml_name]['model'], f'best_ml_model.pkl')
    
    # Save PyTorch model
    torch.save(dl_model.state_dict(), 'dl_model.pth')
    
    # Return best performing model
    best_ml_acc = max([res['accuracy'] for res in ml_results.values()])
    if dl_acc > best_ml_acc:
        print("\nBest Model: Deep Learning")
        return dl_model
    else:
        print(f"\nBest Model: {best_ml_name}")
        return ml_results[best_ml_name]['model']

if __name__ == "__main__":
    best_model = main()
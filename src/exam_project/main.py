# src/exam_project/main.py
import torch
import pandas as pd
from train import create_dataloader, train_model, visualize_training
from evaluate import evaluate_model
from data import load_and_preprocess_data
from model import SectorClassifier
from api import preprocess_new_company, predict_sector


# Load and preprocess the dataset
file_path = "../../data/raw/sp500_companies.csv"  # Replace with the path to your dataset
column_transformer, X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)

# Create DataLoaders
batch_size = 32
train_loader = create_dataloader(X_train, y_train, batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size)
test_loader = create_dataloader(X_test, y_test, batch_size)

# Initialize model, criterion, and optimizer
input_size = X_train.shape[1]
num_classes = y_train.shape[1]
model = SectorClassifier(input_size, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
print("Starting training...")
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
visualize_training(train_losses, val_losses)

# Evaluate the model on the test set
print("Evaluating model...")
evaluate_model(model, test_loader)

# Save the model
model_path = "../../models/sector_classification_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Predict sector for a new company
new_company_raw = {
    'Exchange': 'NASDAQ',         # Example: 'NASDAQ'
    'Symbol': 'AAPL',             # Example: 'AAPL'
    'Industry': 'Technology',     # Example: 'Technology'
    'Currentprice': 150.0,        # Example: numerical value
    'Marketcap': 2500000000000,   # Example: numerical value
    'Ebitda': 80000000000,        # Example: numerical value
    'Revenuegrowth': 0.05,        # Example: numerical value
    'Fulltimeemployees': 154000,  # Example: numerical value
    'Weight': 0.01                # Example: numerical value
}

# Preprocess and predict
print("Predicting sector for a new company...")
new_company_transformed = preprocess_new_company(new_company_raw, column_transformer)
sector_index = predict_sector(model, new_company_transformed)
sector_name = pd.get_dummies(pd.read_csv(file_path)['Sector']).columns[sector_index]
print(f"Predicted Sector: {sector_name}")

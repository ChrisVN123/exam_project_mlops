# src/exam_project/api.py
import torch
import pandas as pd

def preprocess_new_company(raw_data, column_transformer):
    raw_df = pd.DataFrame([raw_data])
    transformed_data = column_transformer.transform(raw_df).toarray()
    return transformed_data

def predict_sector(model, input_data):
    model.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
    return predicted_index

# src/exam_project/evaluate.py
import torch

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(y_batch, dim=1)
            correct += (predicted == actual).sum().item()
            total += y_batch.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")
import torch
from torch import nn

class SectorClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SectorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x

if __name__ == "__main__":
    # Specify input size and number of classes
    input_size = 784  # Example: flattened 28x28 images
    num_classes = 10  # Example: classify into 10 categories

    # Initialize the model
    model = SectorClassifier(input_size=input_size, num_classes=num_classes)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Provide dummy input with the correct shape
    dummy_input = torch.randn(1, input_size)  # Flattened input
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

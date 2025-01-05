from pathlib import Path
import typer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)



def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Longbusinesssummary', 'City', 'State', 'Country', 'Shortname', 'Longname'])
    data = data.dropna()
    
    X = data.drop(columns=['Sector'])
    y = data['Sector']
    
    categorical_features = ['Exchange', 'Symbol', 'Industry']
    numerical_features = ['Currentprice', 'Marketcap', 'Ebitda', 'Revenuegrowth', 'Fulltimeemployees', 'Weight']
    
    column_transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

    X_transformed = column_transformer.fit_transform(X)
    y_encoded = pd.get_dummies(y).values

    X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return column_transformer, X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":
    typer.run(preprocess)

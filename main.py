import torch
import gc
from torch.utils.data import random_split
from preprocess import Preprocess
from model import GraphSAGE
from train import train_ill, test, test_first_10
from utils import create_pyg_data
from torch_geometric.data import DataLoader

# Load and preprocess data
preprocessor = Preprocess("archive/ALL")
preprocessor.load_data()
preprocessor.one_hot_encode_emotions()
processed_data = preprocessor.get_processed_data()


# Clean up memory
del preprocessor
gc.collect()


# Convert the dataframe to PyTorch Geometric data objects
pyg_list = create_pyg_data(processed_data)


# Clean up memory
del processed_data
gc.collect()


# Split the data into training and testing sets
dataset_size = len(pyg_list)
test_size = int(0.2 * dataset_size)
train_size = dataset_size - test_size
train_data, test_data = random_split(pyg_list, [train_size, test_size], generator=torch.Generator().manual_seed(42))
full_loader = DataLoader(pyg_list, batch_size=16, shuffle=True)

# Clean up memory
del pyg_list
gc.collect()


# Create PyTorch Geometric DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'lr': 0.001,
    'hidden_dim': 128,
    'num_layers': 4,
    'batch_size': 16,
    'num_epochs': 50,
    'dropout': 0.5
}


model = GraphSAGE(in_channels=15, hidden_channels=params['hidden_dim'], out_channels=1,
                                  num_layers=params['num_layers'], dropout=params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
criterion_emotion = torch.nn.CrossEntropyLoss()
criterion_illness = torch.nn.MSELoss()

# Training loop
for epoch in range(1, params['num_epochs'] + 1):
    train_loss_emotion, train_loss_illness = train_ill(model, optimizer, train_loader, device, criterion_emotion, criterion_illness)
    
    # Print training progress
    print(f'Epoch {epoch}, Train Loss Emotion: {train_loss_emotion:.4f}, Train Loss Illness: {train_loss_illness:.4f}')


# Testing the first 10 samples
test_first_10(model, test_loader, device)
test_accuracy, _ = test(model, test_loader, device)

print(f'Test Accuracy: {test_accuracy:.4f}')


# Full training loop
# for epoch in range(1, params['num_epochs'] + 1):
#     train_loss_emotion, train_loss_illness = train_ill(model, optimizer, full_loader, device, criterion_emotion, criterion_illness)

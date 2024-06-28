import torch
import gc
from torch.utils.data import random_split
from preprocess import Preprocess
from .model import GraphSAGE
from .train import train_ill, test, test_first_10, evaluate
from .utils import create_pyg_data
from torch_geometric.data import DataLoader
import numpy as np


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
val_size = int(0.1 * dataset_size)
final_train_size = dataset_size - val_size
final_train_data, val_data = random_split(pyg_list, [final_train_size, val_size], generator=torch.Generator().manual_seed(42))

train_size = dataset_size - test_size
train_data, test_data = random_split(pyg_list, [train_size, test_size], generator=torch.Generator().manual_seed(42))
full_loader = DataLoader(pyg_list, batch_size=16, shuffle=True)


# Clean up memory
del pyg_list
gc.collect()


# Create PyTorch Geometric DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
final_train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(train_data, batch_size=16, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'lr': 0.001,
    'hidden_dim': [64, 64, 128, 256],
    'num_layers': 4,
    'batch_size': 16,
    'num_epochs': 100,
    'dropout': 0.1
}


model = GraphSAGE(in_channels=15, hidden_channels=params['hidden_dim'], out_channels=1,
                                  num_layers=params['num_layers'], dropout=params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
criterion_emotion = torch.nn.CrossEntropyLoss()
criterion_illness = torch.nn.MSELoss()

# Training loop
# for epoch in range(1, params['num_epochs'] + 1):
#     train_loss_emotion, train_loss_illness = train_ill(model, optimizer, train_loader, device, criterion_emotion, criterion_illness)
    
#     # Print training progress
#     print(f'Epoch {epoch}, Train Loss Emotion: {train_loss_emotion:.4f}, Train Loss Illness: {train_loss_illness:.4f}')


# Testing the first 10 samples
# test_first_10(model, test_loader, device)
# test_accuracy, _ = test(model, test_loader, device)

# print(f'Test Accuracy: {test_accuracy:.4f}')

patience = 10
num_epochs = 80
best_val_loss = np.inf
epochs_no_improve = 0 
for epoch in range(1, num_epochs + 1):
    train_loss_emotion, train_loss_illness = train_ill(model, optimizer, final_train_loader, device, criterion_emotion, criterion_illness)
    val_loss_emotion, val_loss_illness, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, criterion_emotion, criterion_illness)
    val_loss = val_loss_emotion + val_loss_illness

    print(f'Epoch {epoch}, Train Loss Emotion: {train_loss_emotion:.4f}, Train Loss Illness: {train_loss_illness:.4f}')
    print(f'Epoch {epoch}, Val Loss Emotion: {val_loss_emotion:.4f}, Val Loss Illness: {val_loss_illness:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print('Stopped Early')
        break


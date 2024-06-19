import torch
from torch_geometric.data import Data
import numpy as np

def create_pyg_data(df):
    pyg_list = []

    for _, row in df.iterrows():
        x = torch.tensor(row['node_features'], dtype=torch.float)

        # Convert adjacency matrix non-zero indices to a single numpy array
        adjacency_matrix = np.array(row['adjacency_matrix'].nonzero())
        edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)


        # Use numerical labels
        y = torch.tensor([row['emotion_int']], dtype=torch.long)  
        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_list.append(data)

    return pyg_list

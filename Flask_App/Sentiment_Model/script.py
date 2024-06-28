import torch
from .model import GraphSAGE
from .preprocess import Preprocess
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F

def sentiment_model(file_path_recording):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessor = Preprocess()

    #File path for recording
    node_features, adjacency_matrix = preprocessor.process_single_file(file_path_recording)

    x = torch.tensor(node_features, dtype=torch.float)
    adjacency_matrix = np.array(adjacency_matrix.nonzero())
    edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

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
    state = torch.load('Sentiment_Model/best_model.pth')
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        emotion_probs, illness_score = model(data)
        probabilities = F.softmax(emotion_probs, dim=1)
        emotion_scores = torch.tensor([1, 0.5, 0.8, -1, 1, -0.8, 0])
        modified_scores = probabilities * emotion_scores
        aggregated_score = modified_scores.sum()

        probabilities_list = probabilities.tolist()[0]  
        modified_scores_list = modified_scores.tolist()[0] 
        aggregated_score_float = aggregated_score.item()
        print("Logits:", emotion_probs.tolist()[0])
        print("Probabilities:", probabilities_list)
        print("Weighted Probabilities:", modified_scores_list)
        print("Aggregated Score:", aggregated_score_float)
        formatted_score = f"{aggregated_score_float:.3f}"
        
        return formatted_score
    


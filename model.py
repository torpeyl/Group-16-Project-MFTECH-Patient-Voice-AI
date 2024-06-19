import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, num_emotions=7):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # Emotion prediction layers
        self.emotion_fc = torch.nn.Linear(hidden_channels, num_emotions)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GraphSAGE layers with BatchNorm1d (Prevent overfitting)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Emotion prediction
        emotion_logits = self.emotion_fc(x)
        emotion_probs = self.sigmoid(emotion_logits)

        # Calculate "illness" score based on emotion probabilities
        emotion_scores = torch.tensor([1, 0.5, 0.8, -1, 1, -0.8, 0], device=x.device)
        illness_scores = torch.matmul(emotion_probs, emotion_scores)

        return emotion_probs, illness_scores

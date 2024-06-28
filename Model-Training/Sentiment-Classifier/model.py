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
        self.convs.append(SAGEConv(in_channels, hidden_channels[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels[0]))

        # Hidden layers
        for i in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels[i-1], hidden_channels[i]))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels[i]))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels[-1], out_channels))

        # Emotion prediction layers
        self.emotion_fc = torch.nn.Linear(hidden_channels[-1], num_emotions)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GraphSAGE layers with BatchNorm1d and RELU
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

        # Calculate illness score based on emotion probabilities 
        emotion_scores = torch.tensor([1, 0.5, 0.8, -1, 1, -0.8, 0], device=x.device)
        illness_scores = torch.matmul(emotion_probs, emotion_scores)

        return emotion_probs, illness_scores


import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Embedding dimension
        embed_dim = 16
        
        # Embedding layer
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        # Average layer
        # self.avg = nn.AvgPool1d(kernel_size=1)
        # Linear layer
        self.linear = nn.Linear(embed_dim, 1)
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        # x : (B x T)
        x = self.embedding(x) # (B x T x d)
        # average over time dimension, dim 1
        x = torch.mean(x, dim=1) # (B x d)
        x = self.linear(x) # (B x 1)
        x = self.sigmoid(x) # (B x 1)
        return torch.round(x, decimals = 4)
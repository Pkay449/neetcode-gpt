import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        # first linear layer
        self.first_linear = nn.Linear(784, 512)
        # then we apply some non-linearity
        self.relu = nn.ReLU()
        # then we apply dropout
        self.dropout = nn.Dropout(p=0.2)
        # finally we have the output layer
        self.projection = nn.Linear(512, 10)
        # we apply sigmoid activation to make the output between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # Return the model's prediction to 4 decimal places
        # we have to run the forward pass for each of the layers
        # NOTE
        # Instead of calling the forward method of the layers, such as 
        # self.first_linear.forward(images), we can just call the layer
        # as a function, such as self.first_linear(images)
        # this is because the nn.Module class has a __call__ method that
        # calls the forward method of the class
        out = self.first_linear(images)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.projection(out)
        out = self.sigmoid(out)
        return torch.round(out, decimals = 4)
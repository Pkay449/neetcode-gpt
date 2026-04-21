import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        
        indices = torch.randint(0, len(raw_dataset.split()) - context_length, (batch_size,))

        # output raw_dataset at the indices
        X = [raw_dataset.split()[i:i+context_length] for i in indices]

        Y = [raw_dataset.split()[i+1:i+1+context_length] for i in indices]
        
        return X, Y

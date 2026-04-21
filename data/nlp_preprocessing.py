import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # initialize a vocabulary
        vocabulary = set()
        
        # full sentences
        input_text = positive + negative
        
        # add every word from `positive` and `negative` to `vocabulary`
        for sentence in input_text:
            for word in sentence.split():
                vocabulary.add(word)
                
        # convert the set to list and sort
        sorted_list = sorted(list(vocabulary))
        
        # create a mapping word -> integer
        word_to_int = {word: i + 1 for i, word in enumerate(sorted_list)}
        
        # initialize output tensors
        tensors = []
        # iterate over `positive` and `negative`, and encode
        # every sentence as a tensor
        for sentence in input_text:
            cur_sentence = []
            for word in sentence.split():
                cur_sentence.append(word_to_int[word])
            tensors.append(torch.tensor(cur_sentence))
        
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
            

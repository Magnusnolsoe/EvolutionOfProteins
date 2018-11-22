import numpy as np
import torch

class DataLoader(object):
    def __init__(self, path, batch_size=0, pad_sequences=True):
        self.path = path
        self.batch_size = batch_size
        self.pad_sequences = pad_sequences
        self.inputs = []
        self.targets = []
        self.input_lengths = []
        self.chunk_pointer = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        inputs_batch = []
        targets_batch =[]
        len_batch = []
        
        if (self.chunk_pointer >= len(self.inputs)):
            raise StopIteration
        end = min(len(self.inputs), self.chunk_pointer+self.batch_size)
        
        inputs_batch = self.inputs[self.chunk_pointer:end]
        targets_batch = self.targets[self.chunk_pointer:end]
        len_batch = self.input_lengths[self.chunk_pointer:end]
        
        
        self.chunk_pointer = self.chunk_pointer + self.batch_size
        
        if (self.pad_sequences):
            return self.pad_input_sequence(inputs_batch, targets_batch, len_batch)
        
        return torch.stack(inputs_batch), targets_batch, torch.stack(len_batch)
        
        
    def load_chunk(self):
        
        with open(self.path, 'r') as file:
            for line in file:
                inputs = line.split("\t")[0]
                outputs = line.split("\t")[1]
                inputs = inputs.split(",")
                inputs = list(map(int, inputs))
                inputs = torch.tensor(inputs, dtype=torch.long)
                outputs = outputs.split(",")
                outputs = [x.split(" ") for x in outputs]
                outputs = [list(map(float, x)) for x in outputs]
                outputs = torch.tensor(outputs, dtype=torch.float32)
                self.inputs.append(inputs)
                self.input_lengths.append(len(inputs))
                self.targets.append(outputs)
            
            file.close()   
            
        self.sort_chunk()
            
    def sort_chunk(self):
        sorted_data = sorted(zip(self.input_lengths, range(len(self.inputs)), self.inputs, self.targets), reverse = True)
        
        self.input_lengths = [x for x,_,_ in sorted_data]
        self.inputs = [x for _,x,_ in sorted_data]
        self.targets = [x for _,_,x in sorted_data]
    
    def pad_input_sequence(self, inputs, targets, lengths):
        
        max_length = len(inputs[0])
        padded_inputs = []
        for x in inputs:
            seq_len = len(x)
            padded_inputs.append(torch.nn.functional.pad(x, (0, max_length-seq_len), mode='constant', value=20))

        
        return torch.stack(padded_inputs), torch.tensor(lengths), targets
    

d = DataLoader('data/sample.txt', batch_size = 64)


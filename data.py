import numpy as np
import torch

class DataLoader(object):
    def __init__(self, path, batch_size=0, one_hot_encode=False, pad_sequences=False):
        self.path = path
        self.file = open(self.path, 'r')
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        self.pad_sequences = pad_sequences
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_size == 0:
            try:
                new_entry = next(self.file)
            except StopIteration:
                self.file.close()
                self.file = open(self.path, 'r')
                new_entry = next(self.file)
            return self.process_data(new_entry)
        else:
            inputs = []
            targets = []
            for i in range(self.batch_size):
                try:
                    new_entry = next(self.file)
                    x, t = self.process_data(next(self.file))
                    inputs.append(x)
                    targets.append(t)
                except StopIteration:
                    self.file.close()
                    self.file = open(self.path, 'r')
                    x, t = self.process_data(next(self.file))
                    inputs.append(x)
                    targets.append(t)
            
            if self.pad_sequences:                
                return self.pad_input_sequence(inputs, targets)                
                
            return inputs, targets
        
    def pad_input_sequence(self, inputs, targets):
        
        max_length = len(max(inputs, key=len))
        padded_inputs = []
        sequence_lengths = []
        for x in inputs:
            seq_len = len(x)
            padded_inputs.append(torch.nn.functional.pad(x, (0, max_length-seq_len), mode='constant', value=20))
            sequence_lengths.append(seq_len)
        
        xyz = sorted(zip(sequence_lengths, padded_inputs, targets), reverse=True)
        
        sequence_lengths = [x for x,_,_ in xyz]
        padded_inputs = [y for _,y,_ in xyz]
        targets = [z for _,_,z in xyz]
        
        return torch.stack(padded_inputs), torch.tensor(sequence_lengths), targets
    
    def process_data(self, line):
        inputs = line.split("\t")[0]
        outputs = line.split("\t")[1]

        inputs = inputs.split(",")
        inputs = list(map(int, inputs))
        inputs = torch.tensor(inputs, dtype=torch.long)
        
        if self.one_hot_encode:        
            one_hot_inputs = np.zeros((len(inputs), 20), dtype=int)
            one_hot_inputs[np.arange(len(inputs)), inputs] = 1
            inputs = one_hot_inputs
            
        outputs = outputs.split(",")
        outputs = [x.split(" ") for x in outputs]
        outputs = [list(map(float, x)) for x in outputs]
        outputs = torch.tensor(outputs, dtype=torch.float32)
        
        return inputs, outputs

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

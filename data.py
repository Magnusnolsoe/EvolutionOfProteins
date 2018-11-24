import os
import torch

from sklearn.model_selection import train_test_split

class DataIterator(object):
    def __init__(self, inputs, targets, sequence_lengths,
                 batch_size=0, pad_sequences=True):
        
        self.batch_size = batch_size
        self.pad_sequences = pad_sequences
        self.inputs = inputs
        self.targets = targets
        self.sequence_lengths = sequence_lengths
        self.data_pointer = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        inputs_batch = []
        targets_batch =[]
        seq_len_batch = []
        
        if (self.data_pointer >= len(self.inputs)):
            self.data_pointer = 0
            raise StopIteration
            
        end = min(len(self.inputs), self.data_pointer+self.batch_size)
        
        inputs_batch = self.inputs[self.data_pointer:end]
        targets_batch = self.targets[self.data_pointer:end]
        seq_len_batch = self.sequence_lengths[self.data_pointer:end]
        
        
        self.data_pointer += self.batch_size
        
        if (self.pad_sequences):
            return self.pad_input_sequence(inputs_batch, targets_batch, seq_len_batch)
        
        return torch.stack(inputs_batch), torch.stack(seq_len_batch), targets_batch
    
    
    def pad_input_sequence(self, inputs, targets, lengths):
        
        max_length = len(inputs[0])
        padded_inputs = []
        for x in inputs:
            seq_len = len(x)
            padded_inputs.append(torch.nn.functional.pad(x, (0, max_length-seq_len), mode='constant', value=20))

        
        return torch.stack(padded_inputs), torch.tensor(lengths), targets


class DataLoader(object):
    def __init__(self, dataset, data_dir = "data/"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.inputs = []
        self.targets = []
        self.sequence_lengths = []
        
    def load_data(self):
        path = os.path.join(self.data_dir, self.dataset)
        with open(path, 'r') as file:
            for line in file:
                inputs = line.split("\t")[0]
                outputs = line.split("\t")[1]
                inputs = inputs.split(",")
                inputs = list(map(int, inputs))
                inputs = torch.tensor(inputs, dtype=torch.long)
                outputs = outputs.split(",")
                outputs = [x.split(" ") for x in outputs]
                outputs = [list(map(float, x)) for x in outputs]
                outputs = torch.tensor(outputs, dtype=torch.float)
                self.inputs.append(inputs)
                self.sequence_lengths.append(len(inputs))
                self.targets.append(outputs)
            
            file.close()   

    def sort_data(self, inputs, targets, seq_lengths):
            sorted_data = sorted(zip(seq_lengths, range(len(inputs)), inputs, targets), reverse = True)
            
            i = [x for _,_,x,_ in sorted_data]
            t = [x for _,_,_,x in sorted_data]            
            seq_len = [x for x,_,_,_ in sorted_data]
            return i, t, seq_len
            
    def split(self, split_rate=0.33):
        X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(self.inputs, self.targets, self.sequence_lengths,
                                                                                 test_size=split_rate)
        return X_train, X_test, y_train, y_test, seq_train, seq_test


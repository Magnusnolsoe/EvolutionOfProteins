class DataLoader(object):
    def __init__(self, path, batch_size=0, one_hot_encode=False):
        self.path = path
        self.file = open(self.path, 'r')
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        
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
            container = []
            for i in range(self.batch_size):
                try:
                    new_entry = next(self.file)
                    container.append([self.process_data(next(self.file))])
                except StopIteration:
                    self.file.close()
                    self.file = open(self.path, 'r')
                    container.append([self.process_data(next(self.file))])
            return container
    
    def process_data(self, line):
        inputs = line.split("\t")[0]
        outputs = line.split("\t")[1]

        inputs = inputs.split(",")
        inputs = list(map(int, inputs))
        inputs = np.array(inputs)
        
        if self.one_hot_encode:        
            one_hot_inputs = np.zeros((len(inputs), 20), dtype=int)
            one_hot_inputs[np.arange(len(inputs)), inputs] = 1
            inputs = one_hot_inputs
            
        outputs = outputs.split(",")
        outputs = [x.split(" ") for x in outputs]
        outputs = [list(map(float, x)) for x in outputs]
        outputs = np.array(outputs)
        
        return inputs, outputs

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
import unittest
import torch
from utils import custom_cross_entropy, build_mask, pad_predictions

class testUtils(unittest.TestCase):
    def setUp(self):
        self.target1 = torch.tensor([[0.4, 0.6], [0.2,0.8],[0.1,0.9]], dtype=torch.float)
        self.target2 = torch.tensor([[0.2, 0.8],[0.3,0.7]], dtype=torch.float)
        self.y1=torch.tensor([[0.31,0.69],[0.23,0.77],[0.5,0.5]], dtype=torch.float)
        self.y2=torch.tensor([[0.22,0.78],[0.25,0.75]], dtype=torch.float)
        self.batcht = [self.target1,self.target2]
        self.batchy = [self.y1,self.y2]
        self.seqlengths = torch.tensor([3,2])
        
        self.target3 = torch.tensor([[0.4, 0.6], [0.2,0.8]], dtype=torch.float)
        self.batchEqualLen = [self.target2,self.target3]
        self.seqlengthsEqualLen = torch.tensor([2,2])
        
        self.device = torch.device('cpu')
        
    def test_crossEntropy(self):
        
        
        
        correctLoss = 0.5942626962
        
        self.batchy = pad_predictions(self.batchy, self.seqlengths)
        self.batcht = pad_predictions(self.batcht, self.seqlengths)
        
        self.assertAlmostEqual((custom_cross_entropy(self.batchy, self.batcht,self.seqlengths,self.device)).item(), correctLoss)
        
    def test_build_mask(self):
        seq_lengths = [3,2]
        correct_mask = [[1,1,1],[1,1,0]]
        self.assertAlmostEqual((build_mask(seq_lengths)).tolist(), correct_mask)
        
    def test_build_mask_no_padding(self):
        seq_lengths = [3,3]
        correct_mask = [[1,1,1],[1,1,1]]
        self.assertAlmostEqual((build_mask(seq_lengths)).tolist(), correct_mask)
    
    def test_pad_targets(self):
        
        padded1 = [[0.4, 0.6], [0.2,0.8],[0.1,0.9]]
        padded2 = [[0.2, 0.8],[0.3,0.7],[1.0,1.0]]
        correct_padded = torch.tensor([padded1,padded2])

        for profile, correct_profile in zip(pad_predictions(self.batcht, self.seqlengths),correct_padded):
            for position, correct_position in zip(profile, correct_profile):
                for acid,correct in zip(position,correct_position):
                    self.assertAlmostEqual(acid, correct)
                    
    def test_no_pad_targets(self):
        for profile, correct_profile in zip(pad_predictions(self.batchEqualLen, self.seqlengthsEqualLen),self.batchEqualLen):
            for position, correct_position in zip(profile, correct_profile):
                for acid,correct in zip(position,correct_position):
                    self.assertAlmostEqual(acid, correct)
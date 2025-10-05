import torch
import unittest

class TestTorchBasic(unittest.TestCase):
    def test_torch_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.sum(x).item() == 6.0

if __name__ == '__main__':
    unittest.main()
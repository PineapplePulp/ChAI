import sys
import os
import torch.nn as nn

# Add the scripts directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import chai

os.makedirs('models', exist_ok=True)

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)



class DummyTwo(nn.Module):
    def __init__(self, input_model):
        super(DummyTwo, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.newmod = nn.Sequential(
            input_model,
        )

dummy = Dummy()

dummy_two = DummyTwo(dummy)

dummy.chai_dump('models/dummy', 'dummy', with_json=True, verbose=True)
dummy_two.chai_dump('models/dummy_two', 'dummy_two', with_json=True, verbose=True)

import os
import torch

import env
from train import DQN
from utils.testing import run_system

if __name__ == "__main__":
    model = DQN()
    model.load_state_dict(torch.load(os.path.join(env.models_path, 'cnn_v1.pt')))
    run_system(model=model)

import os
import torch

import env
from models.models import CNNv2
from utils.testing import run_system

if __name__ == "__main__":
    model = CNNv2()
    model.load_state_dict(torch.load(os.path.join(env.models_path, 'cnnv2_3.pt')))
    print("Press Q to exit.")
    run_system(model=model)

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some CUDA ops may not support strict determinism depending on version.
            pass

import logging
import torch
import random
import numpy as np
import os
from tabulate import tabulate
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)



def seed_all(seed, cuda_deterministic=False):
    """
    设置所有的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The value of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - TorchVision (optional): TorchVision version.
        - OpenCV (optional): OpenCV version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision
        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    try:
        import cv2  # type: ignore # 不加这个注释vscode会有黄色波浪线警告。强迫症
        env_info.append(("OpenCV", cv2.__version__))
    except ModuleNotFoundError:
        pass

    return tabulate(env_info)


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)
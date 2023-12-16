import os
from typing import Optional
from dataclasses import dataclass
import logging

from .config_parser import Arguments

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(Arguments):
    seed: int = 42
    work_dir: str = "."
    max_epochs: int = 10
    max_iters: int = 0
    train_batch_size: int = 12
    eval_batch_size: int = 12
    max_num_checkpoints: Optional[int] = None
    checkpoint_period: int = 1
    log_period: int = 50
    eval_period: int = 1
    clip_grad_norm: float = 0.0
    enable_amp: bool = False
    grad_accum: int = 1
    metric_window_size: int = 20
    lr: float = 1e-4
    by_epoch: bool = True
    warmup_t: int = 0
    warmup_by_epoch: bool = False
    warmup_mode: str = "fix"
    warmup_init_lr: float = 0.0
    warmup_factor: float = 0.0
    unpack_batch_dict: bool = False

    def __post_init__(self):
        """尽量不要在这里创建新的属性，我们希望asdict只包含上面的属性"""
        self.work_dir = os.path.expanduser(self.work_dir)

        if self.max_iters > 0:
            logger.info("max_iters is given, it will override any value given in max_epochs.")

        if self.log_period % self.grad_accum != 0:
            logger.warning(f"log_period={self.log_period} is not divisible by grad_accum={self.grad_accum}, "
                           "which may cause inaccurate logging information.")
            
        if self.metric_window_size % self.grad_accum != 0:
            logger.warning(f"metric_window_size={self.metric_window_size} is not divisible by grad_accum={self.grad_accum}, "
                           "which may cause inaccurate metrics.")
            
    
    @property
    def train_by_epoch(self) -> bool:
        # 如果train_by_epoch，则需要在trainer中设置epoch_len和max_iters
        return self.max_epochs > 0

    @property
    def ckpt_dir(self) -> str:
        """The directory to save checkpoints. Overwrite this method to change the path."""
        return os.path.join(self.work_dir, "ckpt")
    
    @property
    def tb_log_dir(self) -> str:
        """The directory to save tensorboard files. Overwrite this method to change the path."""
        return os.path.join(self.work_dir, "tb_logs")


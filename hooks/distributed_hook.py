from .hookbase import HookBase
from ..distributed import is_distributed
from torch.distributed import barrier

class DistributedHook(HookBase):
    """Call :meth:`DistributedSampler.set_epoch` before each epoch."""
    def after_iter(self) -> None:
        if is_distributed():
            barrier()
            
    def before_iter(self) -> None:
        if is_distributed():
            barrier()

    def before_epoch(self) -> None:
        data_loader = self.trainer.train_data_loader
        if hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(self.trainer.cur_epoch)
        elif hasattr(data_loader.batch_sampler.sampler, "set_epoch"):
            # batch sampler in PyTorch warps the sampler as its attributes
            data_loader.batch_sampler.sampler.set_epoch(self.trainer.cur_epoch)
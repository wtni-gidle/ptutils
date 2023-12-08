"""The code of this module is modified from:

- https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
"""
import functools
from  datetime import timedelta
import logging
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup # type: ignore # 不加这个注释vscode会有黄色波浪线警告。强迫症


logger = logging.getLogger(__name__)
# todo

@functools.lru_cache()
def _get_global_gloo_group() -> ProcessGroup:
    """Return a process group based on gloo backend, containing all ranks.
    The result is cached.
    如果backend是nccl，则返回backend为gloo的全局组（新创建）
    否则返回全局组（这个否则应该即为gloo，不考虑第三种情况）
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD
    
def all_gather(data: Any, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Run :meth:`all_gather` on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        group (ProcessGroup): A torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: List of data gathered from each rank.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output

def gather(data: Any, dst: int = 0, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Run :meth:`gather` on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        dst (int): Destination rank.
        group (ProcessGroup): A torch process group. By default, will use a group which
            contains all ranks on ``gloo`` backend.

    Returns:
        list[data]: On ``dst``, a list of data gathered from each rank. Otherwise, an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        # 为什么默认使用gloo的全局组呢？我认为是因为pytorch推荐在gpu分布式下使用nccl，在cpu分布式下使用gloo。
        # 至于是否在cpu的情况下gloo比nccl更快或者更好？maybe
        # 所以这里应该是适用于还未将一些变量迁移到gpu的情况，企图以gloo为后端更快
        # facebook: use CPU group by default, to reduce GPU RAM usage.
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    if dist.get_rank(group) == dst:
        #! 再次注意，不要使用[None] * world_size这种写法
        output = [None for _ in range(world_size)]
        # 破案了，原来是你小子（参考gather_object官方文档）
        # 如果是nccl组，gather_object要求object的内部张量迁移到gpu上。
        # 然而又说到使用gpu张量调用这个函数没有得到很好的支持，可能效率很低
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []

# 没用到
def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.

    Args:
        input_dict (dict): All the values will be reduced.
        average (bool): Whether to do average or sum.

    Returns:
        dict: A dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def setup_print_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in master process.

    Args:
        is_master (bool): If the current process is the master process or not.
    """
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    builtins.print = print

def get_world_size() -> int:
    """Return the number of processes in the current process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Return the rank of the current process in the current process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_local_rank() -> int:
    """Return the local rank of the current process in the current process group."""
    # 用来设置device
    if not dist.is_available() or not dist.is_initialized():
        return 0
    # torchrun
    return int(os.environ["LOCAL_RANK"])

def is_distributed() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return False
    else:
        return True

def is_master() -> bool:
    """Return if the current process is the master process or not."""
    return get_rank() == 0

def init_distributed(timeout: int = 1200) -> Tuple[int]:
    local_rank = get_local_rank()
    world_size = get_world_size()
    rank = get_rank()

    torch.cuda.set_device(local_rank)
    print(f"| distributed init (rank {rank})", flush=True)
    dist.init_process_group(backend = 'nccl', timeout = timedelta(seconds=timeout))
    dist.barrier()
    setup_print_for_distributed(rank == 0)

    return rank, local_rank, world_size

# 没用到
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

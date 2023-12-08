import logging
import os
import time
import weakref
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .hooks import HookBase, CheckpointHook, DistributedHook, LoggerHook, LRUpdateHook
from .lr_scheduler import LRWarmupScheduler
from .logger import setup_logger
from .distributed import get_rank, get_world_size, is_master, gather, get_local_rank, is_distributed
from .utils import collect_env, symlink
from .history_buffer import HistoryBuffer
from .training_args import TrainingArguments


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        train_data_loader: DataLoader,
        args: Optional[TrainingArguments] = None
    ) -> None:
        if args is None:
            logger.info(f"No `TrainingArguments` passed, using default configuration")
            args = TrainingArguments()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = LRWarmupScheduler(lr_scheduler, args.by_epoch, len(train_data_loader), args.warmup_t,
                                              args.warmup_by_epoch, args.warmup_mode, args.warmup_init_lr,
                                              args.warmup_factor)

        self.train_data_loader = train_data_loader
        self.metric_storage = MetricStorage(self.args.metric_window_size)
        # 如果train_by_epoch，则需要设置epoch_len和max_iters
        # 这里把args中的max_epochs和max_iters迁移到trainer中
        if args.train_by_epoch:
            self.epoch_len = len(train_data_loader)
            self.max_spochs = args.max_epochs
            self.max_iters = args.max_epochs * self.epoch_len
        else:
            self.max_iters = args.max_iters
        
        self.cur_iter = 0  # [0, max_iters - 1]
        self.start_iter = 0  # [0, max_iters - 1]

        self._hooks: List[HookBase] = []
        self._data_iter = iter(train_data_loader)

        self._default_setup()

    # region property
    @property
    def lr(self) -> float:
        """The learning rate of the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    @property
    def inner_iter(self) -> int:
        """The iteration within the epoch, ranged in [0, epoch_len - 1]."""
        assert self.args.train_by_epoch, "inner_iter is only available when training by epoch."
        return self.cur_iter % self.epoch_len

    @property
    def cur_epoch(self) -> int:
        """The current epoch, ranged in [0, max_epochs - 1]."""
        assert self.args.train_by_epoch, "cur_epoch is only available when training by epoch."
        return self.cur_iter // self.epoch_len
    
    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model
    
    @property
    def hook_info(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.class_name + f" (priority {h.priority})" for h in self._hooks]
    
    # endregion

    def log(self, *args, **kwargs) -> None:
        """Update the metrics stored in :obj:`self.trainer.metric_storage`."""
        self.metric_storage.update(*args, **kwargs)

    def _default_setup(self) -> None:
        # setup the root logger of the `cpu` library to show
        # the log messages generated from this library
        #! 不清楚这里为什么不返回logger，需要测试一下
        # setup_logger("cpu", output_dir=self.args.work_dir, rank=get_rank())

        logger.info("Environment info:\n" + collect_env())

        # region register hooks
        default_hooks = [LRUpdateHook(), DistributedHook()]
        if is_master():
            default_hooks.extend([
                CheckpointHook(self.args.checkpoint_period, self.args.max_num_checkpoints),
                LoggerHook(self.args.log_period, tb_log_dir=self.args.tb_log_dir)
            ])
        self.register_hooks(default_hooks)
        logger.info(f"Registered default hooks: {self.hook_info}")
        # endregion

        # region AMP
        self._grad_scaler = GradScaler(enabled=self.args.enable_amp)
        if self.args.enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")
        # endregion

        # region directory
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(f"\n{split_line}\n"
                    f"Work directory: {self.args.work_dir}\n"
                    f"Checkpoint directory: {self.args.ckpt_dir}\n"
                    f"Tensorboard directory: {self.args.tb_log_dir}\n"
                    f"{split_line}")
        # endregion

    def register_hooks(self, hooks: List[HookBase]) -> None:
        """Register hooks to the trainer.

        Args:
            hooks (list[HookBase]): List of hooks to be registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: HookBase) -> None:
        """Register a hook to the trainer.

        For hooks with the same priority, they are executed in the order they are registered.

        Args:
            hook (HookBase): The hook to be registered.
        """
        assert isinstance(hook, HookBase)
        assert hook.priority >= 1 and hook.priority <= 10
        # To avoid circular reference, hooks and trainer cannot own each other. This normally
        # does not matter, but will cause memory leak if the involved objects contain __del__.
        # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        hook.trainer = weakref.proxy(self)
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if hook.priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def _call_hooks(self, stage: str) -> None:
        # 按顺序执行hook，比如stage为before_epoch
        for h in self._hooks:
            getattr(h, stage)()

    def _log_iter_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float,
                          iter_time: float) -> None:
        """
        Args:
            loss_dict (dict): Dict of scalar losses.
            data_time (float): Time taken by the dataloader iteration.
            iter_time (float): Time taken by one complete iteration.
        虽然loss_dict是一个可以有多个key的字典，但最好还是只有一个loss吧
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict.update(data_time=data_time, iter_time=iter_time)
        # gather metrics among all workers for logging
        all_metrics_dict: List[Dict] = gather(metrics_dict)

        if is_master():
            self.log(self.cur_iter, lr=self.lr, smooth=False)

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.log(self.cur_iter, data_time=data_time)

            # same as data_time
            iter_time = np.max([x.pop("iter_time") for x in all_metrics_dict])
            self.log(self.cur_iter, iter_time=iter_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.cur_iter}! "
                    f"loss_dict={metrics_dict}.")
            # 如果metrics_dict的loss有多个，则key不能是total_loss
            self.log(self.cur_iter, total_loss=losses_reduced)
            if len(metrics_dict) > 1:
                self.log(self.cur_iter, **metrics_dict)

    def train_one_iter(self) -> None:
        """Train one iteration.

        Subclass :class:`cpu.trainer.Trainer` and implement your own :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_data_loader)
            batch = next(self._data_iter)
        # region: 新增将batch迁移到gpu上，device默认为local_rank
        device = get_local_rank()
        if isinstance(batch, dict):
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            batch = [d.to(device) for d in batch]
        # endregion
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        # If self._enable_amp=False, autocast and GradScaler’s calls become no-ops.
        # This allows switching between default precision and mixed precision
        # without if-else statements.

        # 混合精度+梯度累加+梯度裁剪+分布式训练
        # 经过改动之后可能不适用于train_by_iter的训练
        i = self.inner_iter + 1
        scale_factor = self.args.grad_accum
        if i >= (self.epoch_len // self.args.grad_accum) * self.args.grad_accum + 1:
            scale_factor = self.epoch_len % self.args.grad_accum

        my_context = self.model.no_sync if is_distributed() and i % self.args.grad_accum != 0 else nullcontext
        with my_context():
            with autocast(enabled=self.args.enable_amp):
                if self.args.unpack_batch_dict:
                    loss_dict = self.model(**batch)
                else:
                    loss_dict = self.model(batch)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                # 此时loss_dict是字典
                else:
                    losses = sum(loss_dict.values())
                # grad_accum
                adjusted_loss = losses / scale_factor
            self._grad_scaler.scale(adjusted_loss).backward()

        ##########################
        # 3. Calculate gradients #
        ##########################
        if (i % self.args.grad_accum == 0) or (i == self.epoch_len):
            if self.args.clip_grad_norm > 0:
                self._grad_scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            ##############################
            # 4. Update model parameters #
            ##############################
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
            self.optimizer.zero_grad()

        self._log_iter_metrics(loss_dict, data_time, time.perf_counter() - iter_start_time)

    def train(self, resume_from_checkpoint: Optional[str] = None, auto_resume: bool = True) -> None:
        """Start training.

        If ``resume_from_checkpoint`` is specified, resume from the given checkpoint.
        Otherwise, auto resume from the latest checkpoint.

        Args:
            resume_from_checkpoint (str): Path to the checkpoint. Defaults to None.
            auto_resume (bool): Defaults to True.
        """
        # region 加载ckpt
        # 如果是第一次训练，resume_from_checkpoint为None，auto_resume是True，但是没有ckpt可以加载，有警告很正常
        # 如果是恢复训练，则可以实现自动加载ckpt
        if resume_from_checkpoint is not None:
            self.load_checkpoint(path=resume_from_checkpoint)
        else:
            self.load_checkpoint(auto_resume=auto_resume)
        # endregion

        logger.info(f"Start training from iteration {self.start_iter}")

        self._call_hooks("before_train")
        for self.cur_iter in range(self.start_iter, self.max_iters):
            if self.args.train_by_epoch and self.cur_iter % self.epoch_len == 0:
                self._call_hooks("before_epoch")
            self._call_hooks("before_iter")
            self.train_one_iter()
            # region 改动
            if self.args.train_by_epoch:
                i = self.inner_iter + 1
                if (i % self.args.grad_accum == 0) or (i == self.epoch_len):
                    self._call_hooks("after_iter")
            else:
                self._call_hooks("after_iter")
            # endregion
            if self.args.train_by_epoch and (self.cur_iter + 1) % self.epoch_len == 0:
                self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def save_checkpoint(self, file_name: str) -> None:
        """Save training state: ``epoch``, ``num_gpus``, ``model``, ``optimizer``, ``lr_scheduler``,
        ``metric_storage``, ``hooks`` (optional), ``grad_scaler`` (optional).

        Args:
            filename (str): The checkpoint will be saved as ``ckpt_dir/filename``. ``ckpt_dir/latest.pth``(link)
        """
        data = {
            "num_gpus": get_world_size(),
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
        }
        data.update(dict(epoch=self.cur_epoch) if self.args.train_by_epoch else dict(iter=self.cur_iter))
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states
        if self.args.enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = os.path.join(self.args.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag the latest checkpoint
        dst_file = os.path.join(self.args.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def load_checkpoint(self, path: Optional[str] = None, auto_resume: bool = False):
        """Load the given checkpoint or resume from the latest checkpoint.
        目前这个方法特定于于trainer恢复训练的情况

        Args:
            path (str): Path to the checkpoint to load.
            auto_resume (bool): If True, automatically resume from the latest checkpoint.
        """
        # region 根据path和auto_resume加载ckpt，path优先。auto_resume加载"latest.pth"
        if path is None and auto_resume:
            latest_ckpt = os.path.join(self.args.ckpt_dir, "latest.pth")
            if not os.path.exists(latest_ckpt):
                logger.warning("You specify auto_resume=True, but we fail to find "
                               f"{latest_ckpt} to auto resume from. (This warning )")
            else:
                logger.info(f"Found {latest_ckpt} to auto resume from.")
                path = latest_ckpt
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint: dict = torch.load(path, map_location="cpu")
        else:
            logger.info("Since `path` is not specified and `auto_resume` is False, skip loading checkpoint.")
            return
        # endregion

        # check if the number of GPUs is consistent with the checkpoint
        num_gpus = get_world_size()
        ckpt_num_gpus = checkpoint["num_gpus"]
        assert num_gpus == ckpt_num_gpus, (
            f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
            f"but currently only have {num_gpus} GPUs.")

        # 1. load epoch / iteration: 恢复epoch和iter
        if self.args.train_by_epoch:
            start_epoch = checkpoint["epoch"] + 1
            self.start_iter = start_epoch * self.epoch_len
        else:
            self.start_iter = checkpoint["iter"] + 1

        # 2. load model: 不严格加载，会输出missing和unexpected keys
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                           f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                           f"{incompatible.unexpected_keys}")

        # 3. load metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 4. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 5. load lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 6. load grad scaler
        consistent_amp = not (self.args.enable_amp ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self.args.enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 7. load hooks: 这里模仿model.load_state_dict方法，挺6的
        hook_states: Dict = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break




#* done
class MetricStorage(dict):
    """MetricStorage 类用于在训练过程中存储多个指标的值，支持平滑操作以及对平滑后的值进行访问，便于日志记录。

    该类的设计目的是为了与 TensorBoard 集成，用户在调用 update 方法时需要指定 smooth 参数，
    以确定哪些指标应该进行平滑记录。
    
    每个指标可以获得：局部平均值、最新值、全局平均值、全局总和

    示例::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss 将被平滑，但 lr 不会
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # 类似于 dict，可以通过字符串索引
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        # 窗口大小，用于平滑指标值的history buffer
        self._window_size = window_size
        # 字典，key为指标，value为history buffer
        self._history: Dict[str, HistoryBuffer] = self
        # 字典，key为指标，value为是否smooth
        self._smooth: Dict[str, bool] = {}
        # 字典，key为指标，value为最新一次（当前）为第i次迭代（从0开始）
        self._latest_iter: Dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """在特定迭代中添加多个指标的新标量值。

        Args:
            iter (int): 生成这些值的迭代次数。如果为 None，则使用从 0 开始的内置计数器。
            smooth (bool): 如果为 True，在调用 values_maybe_smooth 时返回这些指标的平滑值。
                否则，返回最新的值。不同调用 update 时，相同的指标必须具有相同的 smooth。
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                # 相同的指标必须具有相同的smooth
                assert self._smooth[key] == smooth
            else:
                # 第一次添加指标，默认上一次的iter序号为-1
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                # 若未指定iter，则自动+1
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """返回多个指标的平滑值或最新值。
        具体行为取决于更新指标时的 smooth。

        Returns:
            dict[str -> (int, float)]:
                从指标名称到其（最新迭代次数、局部平均值/最新值）对的映射。
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest)
            for key, his_buf in self._history.items()
        }

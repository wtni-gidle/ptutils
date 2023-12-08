from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, asdict
import os
import yaml
import logging
from typing import Union


logger = logging.getLogger(__name__)


class ConfigParser(ArgumentParser):
    """
    用于解析命令行参数，命令行参数负责除了train和model以外的参数
    Custom ArgumentParser for handling command-line arguments and loading values from a YAML configuration file.
    
    This parser extends the standard ArgumentParser to support loading configuration values from a YAML file.
    If a configuration file is provided, command-line arguments will override values from the configuration.
    """
    def __init__(self, *args, **kwargs):
        self.config_path = ArgumentParser(add_help = False)
        self.config_path.add_argument("--config", default = None, metavar = "FILE", 
                                      help = "Path to the YAML configuration file. If provided, "
                                      "other command-line arguments will override values from the configuration.")
        self.arguments = []  # List to store the names of added arguments
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        # 在添加参数时记录参数的 dest 值
        argument = super().add_argument(*args, **kwargs)
        self.arguments.append(argument.dest)

        return argument
    
    def parse_args(self):
        """
        解析命令行参数，并根据config汇总yaml文件的参数
        """
        known_args, remaining_args = self.config_path.parse_known_args()
        
        config_path = known_args.config

        if config_path is not None:
            with open(config_path, "r") as f:
                args_dict = yaml.safe_load(f)
            # Check if the configuration entries match the added arguments
            for key in args_dict:
                assert key in self.arguments, f"Unexpected configuration entry: {key}"
            self.set_defaults(**args_dict)

        return super().parse_args(remaining_args)



def save_args(args: Union[Namespace, dict], path: str, rank: int = -1) -> None:
    """
    Save `args` to a YAML file if in the specified rank or not in distributed training. Otherwise, do nothing.

    Args:
        args (Union[Namespace, dict]): The parsed arguments or a dictionary to be saved.
        path (str): A filepath ending with `.yaml`.
        rank (int): Process rank in distributed training. Defaults to -1.
    """
    # If not in the specified rank or in distributed training with rank -1, do nothing
    if rank != 0 and rank != -1:
        return
    # Convert Namespace to dictionary if args is a Namespace
    args_dict = vars(args) if isinstance(args, Namespace) else args

    assert path.endswith(".yaml")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)

    # Write the arguments to the specified YAML file
    with open(path, "w") as f:
        yaml.dump(args_dict, f)

    logger.info(f"Args are saved to {path}.")


    
@dataclass
class Arguments:
    @classmethod
    def from_yaml(cls, path: str):
        # If the input is a string, assume it's a file path to a YAML file
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return cls(**config_dict)
    
    def show_arguments(self) -> dict:
        return asdict(self)

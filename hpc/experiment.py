import dataclasses
import yaml


@dataclasses.dataclass
class ExpConfig:
    output_dir: str = None
    sheet_id: str = None
    credentials_path: str = None
    image: str = None
    checkpoints_dir: str = None
    models_dir: str = None
    datasets_dir: str = None
    train_sbatch_path: str = None
    train_config_path: str = None
    train_args: list = None
    eval_base_sbatch_path: str = None
    upload_script_path: str = None
    yaml_config_path: str = None
    partition: str = None
    account: str = None
    time_limit: str = "01:00:00"
    num_nodes: int = None
    num_nodes_eval: int = None
    account_eval: str = None
    partition_eval: str = None
    time_limit_eval: str = None
    eval_tasks: str = None
    account_upload: str = None
    partition_upload: str = None
    time_limit_upload: str = None
    max_restarts: str = None
    eval_config_path: str = None
    redo_evals: bool = False

    def __init__(
        self,
        config_path=None,
    ):
        if config_path is not None:
            self.from_yaml(config_path)

    def from_yaml(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.from_dict(config)

    def from_args(self, **kwargs):
        self.from_dict(kwargs)

    def from_dict(self, dict_):
        for key, value in dict_.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if current_value is not None:
                    print(
                        f"Overriding previous {key}  with {value} (previous value: {current_value})"
                    )
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

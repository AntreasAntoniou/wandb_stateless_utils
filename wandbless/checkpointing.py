import pathlib
from typing import Any, Dict, Optional, Sequence, Union

import torch
import wandb
from torch import nn


class StatelessCheckpointingWandb:
    def __init__(
        self,
        job_type: Optional[str] = None,
        dir: Union[str, pathlib.Path, None] = None,
        config: Union[Dict, str, None] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        reinit: bool = None,
        tags: Optional[Sequence] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        magic: Union[dict, str, bool] = None,
        config_exclude_keys=None,
        config_include_keys=None,
        anonymous: Optional[str] = None,
        mode: Optional[str] = None,
        allow_val_change: Optional[bool] = None,
        force: Optional[bool] = None,
        tensorboard=None,
        sync_tensorboard=None,
        monitor_gym=None,
        save_code=None,
        id=None,
        settings: Union[wandb.Settings, Dict[str, Any], None] = None,
    ):
        self.run = wandb.init(
            job_type=job_type,
            dir=dir,
            config=config,
            project=project,
            entity=entity,
            reinit=reinit,
            tags=tags,
            group=group,
            name=name,
            notes=notes,
            magic=magic,
            config_exclude_keys=config_exclude_keys,
            config_include_keys=config_include_keys,
            anonymous=anonymous,
            mode=mode,
            allow_val_change=allow_val_change,
            force=force,
            tensorboard=tensorboard,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=monitor_gym,
            save_code=save_code,
            id=id,
            settings=settings,
            resume="allow",
        )

    def save(
        self,
        model: Dict[str, Any],
        model_name: str,
        store_dir: Union[str, pathlib.Path],
    ):
        checkpoint_path = pathlib.Path(store_dir) / "checkpoint.pth"

        if not checkpoint_path.parent.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model, checkpoint_path)
        checkpoint_object = wandb.Artifact(
            type="model", name=f"exp-{self.run.id}.model-{model_name}"
        )
        checkpoint_object.add_file(checkpoint_path.as_posix())
        wandb.log_artifact(checkpoint_object)

    def restore(
        self,
        store_dir: Union[str, pathlib.Path],
        model_name: str,
        version: str,
    ):
        artifact = self.run.use_artifact(
            f"{self.run.entity}/{self.run.project}/exp-{id}.model-{model_name}:{version}",
            type="model",
        )
        artifact_dir = artifact.download(root=store_dir)

        latest_model = torch.load(pathlib.Path(artifact_dir) / "checkpoint.pth")

        return latest_model

    def restore_latest(
        self,
        store_dir: Union[str, pathlib.Path],
        model_name: str,
    ):

        return self.restore(store_dir, model_name, version="latest")

    def restore_epoch(
        self,
        store_dir: Union[str, pathlib.Path],
        model_name: str,
        epoch: int,
    ):

        return self.restore(store_dir, model_name, version=f"v{epoch}")

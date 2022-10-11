import pathlib
from typing import Dict, Union

import torch
import wandb
from torch import nn


def save(
    model: nn.ModuleDict,
    model_name: str,
    store_dir: Union[str, pathlib.Path],
):
    torch.save(model.state_dict(), store_dir)
    checkpoint_object = wandb.Artifact(
        type="model", name=f"exp-{wandb.run.id}.model-{model_name}"
    )
    checkpoint_object.add_file(store_dir)
    wandb.log_artifact(checkpoint_object)


def restore(
    project: str,
    entity: str,
    id: str,
    dir: Union[str, pathlib.Path],
    model_name: str,
    version: str,
):
    run = wandb.init()
    artifact = run.use_artifact(
        f"{entity}/{project}/exp-{id}.model-{model_name}:{version}", type="model"
    )
    artifact_dir = artifact.download(root=dir)

    latest_model = torch.load(pathlib.Path(artifact_dir) / "checkpoint.pth")

    return latest_model


def restore_latest(
    project: str,
    entity: str,
    id: str,
    dir: Union[str, pathlib.Path],
    model_name: str,
):

    return restore(project, entity, id, dir, model_name, version="latest")


def restore_epoch(
    project: str,
    entity: str,
    id: str,
    store_dir: Union[str, pathlib.Path],
    model_name: str,
    epoch: int,
):

    return restore(project, entity, id, store_dir, model_name, version=f"v{epoch}")

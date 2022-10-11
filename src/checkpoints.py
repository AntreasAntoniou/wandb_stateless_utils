import os
from dataclasses import dataclass

import dotenv
import wandb
from rich import print
from rich.traceback import install

dotenv.load_dotenv(override=True, verbose=True)
install()


@dataclass
class DummyConfig:
    i: int = 0
    j: int = 0
    z: int = 0


wandb.init(
    project=os.environ["wandb-extras-debug"],
    entity="machinelearningbrewery",
    config=DummyConfig().__dict__,
)

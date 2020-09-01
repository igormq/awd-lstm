import shutil
from functools import partial
from tempfile import mkdtemp
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from main import AWDLSTM
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import BRTD
from datasets.bptt import BPTTTensorDataset
from model import WDLSTM
from utils import repackage_hidden

import logging

class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))

class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics["val_loss"].item(),
            ppl=trainer.callback_metrics["val_ppl"].item())

def train_tune_checkpoint(
    config,
    checkpoint_dir=None,
    num_epochs=2,
    num_gpus=1):


    datasets, vocab = BRTD.create(config['data'])
    config['num_tokens'] = len(vocab)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[CheckpointCallback(),
                   TuneReportCallback()])
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = AWDLSTM._load_model_state(ckpt, config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = AWDLSTM(config)

    train_data = DataLoader(
        BPTTTensorDataset(datasets['train'], config['batch_size'], config['bptt']), batch_size=None, num_workers=1)

    val_data = DataLoader(
        BPTTTensorDataset(datasets['valid'], config['batch_size'], config['bptt']), batch_size=None, num_workers=1)

    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)

if __name__ == "__main__":
    num_samples=100
    num_epochs=2
    gpus_per_trial=1

    config = {
        "num_embedding": tune.choice([100, 200, 400, 800]),
        "num_hidden": tune.choice([500, 1150]),
        "num_layers": 3,
        "learning_rate": 30,
        "bptt": tune.choice([20, 50, 70, 100]),
        "optimizer": "sgd",
        "batch_size": 4,
        "output_dropout": tune.uniform(),
        "hidden_dropout": tune.uniform(),
        "input_dropout": tune.uniform(),
        "embedding_dropout": tune.uniform(),
        "weight_dropout": tune.uniform(),
        "alpha": tune.loguniform(1e-6, 3).func(None),
        "beta": tune.loguniform(1e-6, 3).func(None),
        "weight_decay": tune.loguniform(1e-9, 1e-3).func(None),
        "tie_weights": tune.choice([False, True]),
        "gradient_clip_val": tune.choice([0.25, 1, 10, 100]),
        "data": "/home/igor.quintanilha/repos/awd-lstm/data/brtd",
        "multi_step_lr_milestones": [1e10]
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="train_ppl",
        mode="min",
        perturbation_interval=4,
        hyperparam_mutations={
            "learning_rate": lambda: tune.loguniform(1e-4, 30).func(None),
            "batch_size": [4, 8]
        })

    reporter = CLIReporter(
        parameter_columns=["learning_rate", "batch_size", "weight_decay", "num_embedding"],
        metric_columns=["loss", "ppl", "training_iteration"])

    tune.run(
        partial(
            train_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pbt")
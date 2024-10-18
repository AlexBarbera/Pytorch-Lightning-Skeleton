import sys
import argparse
import models
import data
import lightning
import os


def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("--log-dir", type=str, help="Where to store training logs.", default="./logs")
    output.add_argument("--experiment-name", type=str, help="Name of run for logs.", default="lightning")
    return output.parse_args(sys.argv[1:])


def pipeline():
    args = parse_args()
    print(args)

    # TODO add params
    dataloader = data.get_dataloader()

    # TODO add params
    model = models.get_model()

    callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(
            os.path.join(args.log_dir, "checkpoints"),
            save_last=True,
            every_n_epochs=1
        ),
        lightning.pytorch.callbacks.ModelSummary(),
        lightning.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        lightning.pytorch.callbacks.TQDMProgressBar()
    ]

    logger = lightning.pytorch.loggers.TensorBoardLogger(args.log_dir, name=args.experiment_name)

    trainer = lightning.Trainer(callbacks=callbacks,
                                logger=logger,
                                accelerator="gpu" if torch.cuda.is_available() else "cpu")

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    pipeline()

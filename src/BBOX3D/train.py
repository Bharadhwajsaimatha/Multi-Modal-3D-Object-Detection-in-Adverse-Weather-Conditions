import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from BBOX3D.dataset.BBox3D_dm import BBox3DDataModule
from BBOX3D.model.BBox3DDetector import Lit3DDetector 
import torch

hydra.output_subdir = None

@hydra.main(config_path="config/", config_name="det3d.yaml")
def main(config: DictConfig):
    exp_name = f"{config.exp_prefix}_{config.dataset}_bs{config.batch_size}_lr{config.lr}"
    log_dir = os.path.join(config.logdir, exp_name)

    model = Lit3DDetector(lr=config.lr)

    datamodule = BBox3DDataModule(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="", version="")

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        monitor="val/IoU_3D",
        mode="max",
        save_top_k=1,
        filename="{epoch:03d}-{val/IoU_3D:.4f}"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=config.n_gpus,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        accelerator="ddp" if config.n_gpus > 1 else None,
        precision=16 if config.use_amp else 32
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

import os

import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.cli
import torch

import dataset
import cross_fold
import lightning_module
import lightning_fabric.utilities.cloud_io as pl_cloud_io
import wandb


class CustomSaveConfigCallback(pl.Callback):
    """Saves a LightningCLI config to the experimentlog_dir when training starts.

    This class including most of the documentation is completely copied/adapter from Pytorch Lightning's
    cli.SaveConfigCallback class. The issue with their implementation is that the config.yaml is saved in the trainer's
    log_dir which serves as the root directory for the logging. This is problematic because every training run uses this
    save log_dir which means the config.yaml is overwritten each time. Instead, we want to instead save the config.yaml
    in the training run's log directory, also known as the experiment's log_dir. My update to Lightning's class simply
    makes this update and changes relatively little code. I commented the code that was changed in self.setup(). There's
    a good chance that my change might break the code if a logger other than the WeightsAndBias logger is used or if
    multiple loggers are used simultaneously.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run

    """

    def __init__(
        self,
        parser: pl.cli.LightningArgumentParser,
        config: pl.cli.Namespace,
        config_filename: str = "lightning_config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        # The following lines are the ones I changed to make this a custom callback
        if trainer.fast_dev_run or trainer.logger is None or type(trainer.logger) is not pl.loggers.WandbLogger:
            # logging is disabled
            return

        log_dir = wandb.run.dir if trainer.is_global_zero else None
        log_dir = trainer.strategy.broadcast(log_dir)

        # end changes by Tim
        config_path = os.path.join(log_dir, self.config_filename)
        fs = pl_cloud_io.get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class CustomLightningCLI(pl.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        # checkpoint (filepath) for loading model weights
        parser.add_argument("--wandb_checkpoint", default=None, help="path to a Weights and Biases checkpoint to load"
                                                                     " before training, validation, or testing.")
        parser.add_argument("--use_cf_validation", default=False, help="whether to run cross fold validation.")
        parser.add_argument("--slurm", default=False, help="whether code is running on a SLURM cluster. When True, program will print out SLURM info so that it's logged.")

        # cross val set-up
        parser.add_class_arguments(cross_fold.CrossFoldUtil, "cross_fold")

        parser.link_arguments("model.do_ll", "data.do_ll")
        parser.link_arguments("model.do_seg", "data.do_seg")

        parser.link_arguments(source="model.do_seg", target="model.n_seg_classes",
                              compute_fn=lambda do_seg: 1 if do_seg else 0)
        parser.link_arguments(source="model.do_ll", target="model.n_landmarks",
                              compute_fn=lambda do_ll: 6 if do_ll else 0)

    def before_instantiate(self):
        if not self.config.model.do_ll:
            self.config.model.num_seg_classes = 0
        if not self.config.model.do_seg:
            self.config.model.num_landmarks = 0

    def before_any(self):
        """Code that runs before any subcommand (i.e., before_fit, before_validate, before_test, before_predict).
        This method is not built into LightningCLI, so it must manually be called in each of the subcommand methods."""
        self.load_wandb_weights()
        self.datamodule.device = f'cuda:{self.trainer.local_rank}'
        if self.config[f'{self.subcommand}.slurm']:
            self.print_slurm()

    def before_fit(self):
        self.before_any()
        if self.config[f'{self.subcommand}.use_cf_validation']:
            self.setup_cf_validation()

        if self.trainer.is_global_zero:
            self.trainer.logger.experiment.config['crop_probs'] = self.datamodule.crop_probs*100

    def before_validate(self):
        self.before_any()

        if self.config[f'{self.subcommand}.use_cf_validation']:
            self.setup_cf_validation()

    def before_test(self):
        self.before_any()

    def before_predict(self):
        self.before_any()

    def load_wandb_weights(self):
        ckpt = self.config[f'{self.subcommand}.wandb_checkpoint']

        if ckpt is not None:
            ckpt = f"model-{ckpt}"
            print(f"Checkpoint is {ckpt}")
            # load model
            wandb_logger = self.trainer.logger
            artifact = wandb_logger.experiment.use_artifact(ckpt, type="model")
            artifact_dir = artifact.download()
            artifact_path = os.path.join(artifact_dir, "model.ckpt")
            self.model.load_from_checkpoint(artifact_path)

    def setup_cf_validation(self):
        ct_util = cross_fold.CrossFoldUtil(**self.config[self.subcommand].cross_fold.as_dict())
        train_csvs, val_csvs, test_csvs = ct_util.get_train_val_test_lists()
        self.datamodule.data_csvs = {"train": train_csvs, "val": val_csvs, "test": test_csvs}
        print(f"Setup cross validation with fold: {ct_util.fold_idx}!")

    def after_predict(self):
        if len(self.model.skipped_preds) > 0:
            print("The following files were skipped, probably because of memory errors.")
            for img_filename in self.model.skipped_preds:
                print(img_filename)

    def print_slurm(self):
        envo = os.environ
        print(f"SLURM_JOB_ID:{envo['SLURM_JOB_ID']}\n"
              f"SLURM_JOB_NODELIST:{envo['SLURM_JOB_NODELIST']}\n")

def cli_main():
    # save only the k latest models
    #k = 3
    #model_ckpt = pl.callbacks.ModelCheckpoint(save_top_k=k, mode='max', monitor='time_log')

    torch.set_float32_matmul_precision('medium')
    cli = CustomLightningCLI(datamodule_class=dataset.SegLLDataModule,
                             model_class=lightning_module.UNetFastModule,
                             trainer_class=pl.Trainer,
                             save_config_callback=CustomSaveConfigCallback)


if __name__ == '__main__':
    cli_main()

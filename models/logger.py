import os
import time
import numpy as np
import torch
import torchvision
from PIL import Image
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0 and
                pl_module.global_step!=0):
            
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

class Val(Callback):
    def __init__(self, val_freq, save_img, metrics, dataset_config):
        super().__init__()
        self.val_freq = val_freq
        self.save_img = save_img
        self.metrics = metrics
        self.dataset_config = dataset_config

    @rank_zero_only
    def val(self, pl_module, batch_idx):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "val") and
                callable(pl_module.val)and
                pl_module.global_step!=0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            save_dir = os.path.join(pl_module.logger.save_dir, "val")
            with torch.no_grad():
                pl_module.val(self.save_img, save_dir, self.metrics, self.dataset_config)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.val_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.val(pl_module, batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, lightning_config,options):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.lightning_config = lightning_config
        self.options = options

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path, weights_only=True)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.options))
            OmegaConf.save(self.options,
                           os.path.join(self.cfgdir, "{}-train.yaml".format(self.now)))
            OmegaConf.save(self.lightning_config,
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
            

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

def logs_callbacks(config, ckptdir, logdir, cfgdir, resume, now, options):

    tb_logger = TensorBoardLogger(
        save_dir = logdir, 
        name="tensorboard",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath = ckptdir,
        filename = "{epoch:03}-{step:06}",
        verbose = config.modelcheckpoint.verbose,
        every_n_train_steps = config.modelcheckpoint.every_n_train_steps,
        save_last = config.modelcheckpoint.save_last,
        save_weights_only = config.modelcheckpoint.save_weights_only
    )

    img_logger = ImageLogger(
        batch_frequency = config.image_logger.batch_frequency,
        max_images = config.image_logger.max_images,
        clamp = config.image_logger.clamp
    )
     
    val = Val(
        val_freq = config.val.val_freq,
        save_img = config.val.save_img,
        metrics = config.val.metrics,
        dataset_config = config.val.dataset
    )

    setup_callback = SetupCallback(
        resume=  resume,
        now = now,
        logdir = logdir,
        ckptdir = ckptdir,
        cfgdir = cfgdir,
        lightning_config = config,
        options = options
    )

    learning_rate_logger = LearningRateMonitor(
        logging_interval = "step"
    )
    
    cuda_callback = CUDACallback()

    trainer_kwargs = dict()

    trainer_kwargs["callbacks"] = [setup_callback, learning_rate_logger, cuda_callback]

    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = checkpoint_callback
    else:
        trainer_kwargs["callbacks"].append(checkpoint_callback)

    if config.use_val:
        trainer_kwargs["callbacks"].append(val)

    if config.use_tb:
        trainer_kwargs["logger"] = tb_logger

    if config.use_image_logger:
        trainer_kwargs["callbacks"].append(img_logger)
    
    return  trainer_kwargs
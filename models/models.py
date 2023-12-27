import torch
from ldm.modules.diffusionmodules.util import timestep_embedding
import os
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataset_instantiate import instantiate_from_config as instantiate_dataset_from_config 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics.metrics_all import calculate_psnr_ssim, calculate_lpips, calculate_NIQE, calculate_fid_folder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import shutil
from omegaconf import OmegaConf


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if ((i+1)%3 == 0) and control is not None:
                h = h + control.pop(0)
            hs.append(h)
        h = self.middle_block(h, emb, context)
            

        if control is not None:
            h += control.pop(0)

        for i, module in enumerate(self.output_blocks):
            if  control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if ((i+2)%3 == 0) and control is not None:
                h = h + control.pop(0)

        h = h.type(x.dtype)
        return self.out(h)

class BFRffusion(LatentDiffusion):

    def __init__(self, control_stage_config, control_key,sd_locked_steps,CosineAnnealing_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked_steps = sd_locked_steps
        self.CosineAnnealing_steps = CosineAnnealing_steps
        self.top5_psnr_dict = {}


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = batch[self.first_stage_key]
        # x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        x = self.get_first_stage_encoding(encoder_posterior).detach()

        c = None

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = self.cond_stage_model(t)
        # cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat = c["c_concat"][0][:N]
        c  = None
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            # uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def val(self, save_img, save_dir, metrics, config, ddim_steps=50):
        if config.params.dataroot_gt is not None:
            filename = "gs-{:06}_e-{:06}".format(self.global_step, self.current_epoch)
            save_dir_img = os.path.join(save_dir, filename)
            os.makedirs(save_dir_img, exist_ok=True)
            dataset = instantiate_dataset_from_config(config)
            dataloader = DataLoader(dataset, 
                                batch_size=config.batch_size,
                                num_workers = config.num_workers,
                                )
            ddim_sampler = DDIMSampler(self)
            H = W = 512
            shape = (4, H // 8, W // 8)

            for data in tqdm(dataloader):
                lq = data['lq']
                gt = data['gt']
                lq_path = data['lq_path']
                gt_path = data['gt_path']
                lq = lq.to(self.device)
                cond = {"c_concat": [lq], "c_crossattn": [[""] * config.batch_size]}
                samples, _ = ddim_sampler.sample(ddim_steps,config.batch_size,shape, cond, verbose=False)
                hq_imgs =self.decode_first_stage(samples)
                hq_imgs = torch.clamp((hq_imgs + 1.0) / 2.0, min=0.0, max=1.0)

                for i, img_name in enumerate(lq_path):
                    basename = os.path.splitext(os.path.basename(img_name))[0]
                    hq_img = 255. * rearrange(hq_imgs[i].cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(hq_img.astype(np.uint8)).save(
                            os.path.join(save_dir_img, basename+'.png'))
            writer = SummaryWriter(os.path.join(os.path.dirname(save_dir),'tensorboard','version_0'))
            with open(f'{save_dir}/metrics.txt','a') as f:
                f.write(f'----------{filename}----------\n')
                if metrics.psnr_ssim:
                    psnr,ssim = calculate_psnr_ssim(config.params.dataroot_gt, save_dir_img)
                    f.write(f'psnr={psnr},ssim={ssim}\n')
                    writer.add_scalar("metrics/psnr", psnr, self.global_step)
                    writer.add_scalar("metrics/ssim", ssim, self.global_step)
                if metrics.lpips:
                    lpips = calculate_lpips(config.params.dataroot_gt, save_dir_img)
                    f.write(f'lpips={lpips}\n')
                    writer.add_scalar("metrics/lpips",lpips, self.global_step)
                if metrics.niqe:
                    niqe = calculate_NIQE(save_dir_img)
                    f.write(f'niqe={niqe}\n')
                    writer.add_scalar("metrics/niqe",niqe, self.global_step)
                if metrics.fid:
                    fid = calculate_fid_folder(save_dir_img)
                    f.write(f'niqe={fid}\n')
                    writer.add_scalar("metrics/fid",fid, self.global_step)

                
            # save top5 PSNR model
            if psnr is None:
                if len(self.top5_psnr_dict) < 5:
                    self.top5_psnr_dict[filename] = psnr
                    checkpoint_path = os.path.join(self.logger.save_dir, 'checkpoints', f'checkpoint_{filename}.ckpt')
                    print(f"Saving TOP5 checkpoint_{filename} with PSNR {psnr} to {checkpoint_path}...")
                    torch.save(self.state_dict(), checkpoint_path)
                else:
                    min_psnr = min(self.top5_psnr_dict.values())
                    min_psnr_key = min(self.top5_psnr_dict, key=self.top5_psnr_dict.get)
                    if psnr > min_psnr:
                        self.top5_psnr_dict.pop(min_psnr_key)
                        self.top5_psnr_dict[filename] = psnr
                        checkpoint_path = os.path.join(self.logger.save_dir, 'checkpoints', f'checkpoint_{filename}.ckpt')
                        print(f"Saving TOP5 checkpoint_{filename} with PSNR {psnr} to {checkpoint_path}...")
                        torch.save(self.state_dict(), checkpoint_path)
                        checkpoint_path = os.path.join(self.logger.save_dir, 'checkpoints', f'checkpoint_{min_psnr_key}.ckpt')
                        if os.path.exists(checkpoint_path):
                            print(f"Deleting checkpoint_{min_psnr_key} with PSNR {min_psnr} to {checkpoint_path}...")
                            os.remove(checkpoint_path)

                # delete img if not save_img
                if not save_img and os.path.exists(save_dir_img):
                    shutil.rmtree(save_dir_img)


    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if self.global_step >= self.sd_locked_steps:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cond_stage_trainable:
            params += list(self.cond_stage_model.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        CosineAnnealing_steps = self.CosineAnnealing_steps
        def lr_lambda(current_step):
            if current_step >= CosineAnnealing_steps:
                progress = (current_step - CosineAnnealing_steps) / (self.trainer.max_steps - CosineAnnealing_steps)
                return  0.5 * (1.0 + math.cos(progress * math.pi))
            else:
                return 1.0
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
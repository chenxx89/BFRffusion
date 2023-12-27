import argparse, os, sys, datetime
import cv2
import einops
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange
from annotator.util import HWC3
from models.models import load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import math
from tqdm import trange
import copy
import multiprocessing as mp
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_img(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = HWC3(image)
    image = torch.from_numpy(image.copy()).float() / 255.0
    image = image.unsqueeze(0)
    image = einops.rearrange(image, 'b h w c -> b c h w').clone()

    return image


def main(i, gpuid, num_gpus, opt):
    
    #  config
    seed_everything(opt.seed)
    config = OmegaConf.load(opt.config)
    device = torch.device(f"cuda:{gpuid}") if torch.cuda.is_available() else torch.device("cpu")
    inputdir = config.data.params.dataroot_lq
    input_list = os.listdir(inputdir)
    input_list.sort()
    num_per_parts = len(input_list) //num_gpus
    if i == num_gpus - 1:
        input_list = input_list[i*num_per_parts:]
    else:
        input_list = input_list[i*num_per_parts:(i+1)*num_per_parts]
    output_list = copy.deepcopy(input_list)
    niters = math.ceil(len(input_list) / opt.batch_size)
    config.model.params.cond_stage_config.params.device = f"cuda:{gpuid}"
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(load_state_dict(opt.checkpoint_path),strict= True)
    ddim_sampler = DDIMSampler(model)
    H = W = opt.input_size
    shape = (4, H // 8, W // 8)

    with torch.no_grad():
        for n in trange(niters, desc="Sampling"):
            lq_img_list = []
            for item in input_list[n*opt.batch_size: (n+1)*opt.batch_size]:
                lq_image = load_img(os.path.join(inputdir, item))
                lq_image = lq_image.to(device)
                lq_img_list.append(lq_image)  
            lq_imgs = torch.cat(lq_img_list, dim=0)
            model.cond_stage_model.to(device)
            cond = {"c_concat": [lq_imgs], "c_crossattn": []}
            samples, _ = ddim_sampler.sample(opt.ddim_steps, len(lq_img_list),
                                            shape, cond, verbose=False)
            
            hq_imgs =model.decode_first_stage(samples)

            hq_imgs = torch.clamp((hq_imgs + 1.0) / 2.0, min=0.0, max=1.0)

            for i in range(lq_imgs.size(0)):
                        img_name = output_list.pop(0)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        hq_img = 255. * rearrange(hq_imgs[i].cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(hq_img.astype(np.uint8)).save(
                            os.path.join(opt.outdir, basename+'.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",type=str,default="Test",help="postfix for logdir")
    parser.add_argument("--logdir",type=str,nargs="?",help="dir to write results to",default="results")
    parser.add_argument("--checkpoint_path",type=str,default="experiments/weights/checkpoint_BFRffusion_FFHQ.ckpt",help="dir of ckpt to load")
    parser.add_argument("--ddim_steps",type=int,default=50,help="number of ddpm sampling steps")
    parser.add_argument("--batch_size",type=int,default=10,help="how many samples to produce for each given prompt. A.k.a batch size")
    parser.add_argument("--config",type=str,default="options/test.yaml",help="path to config which constructs model")
    parser.add_argument("--seed",type=int,default=42,help="the seed (for reproducible sampling)")
    parser.add_argument("--input_size",type=int,default=512,help="input size")
    parser.add_argument("--gpu_ids", type=list, default=[0,1,2,3], help='nuns of gpus of all')
    opt = parser.parse_args()

    sys.path.append(os.getcwd())
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    nowname = now + "_" + os.path.splitext(os.path.basename(opt.checkpoint_path))[0]
    opt.outdir = os.path.join(opt.logdir, opt.name, nowname)
    os.makedirs(opt.outdir, exist_ok=True)
    config = OmegaConf.load(opt.config)

    mp = mp.get_context('spawn') 
    process_list = []
    for i,gpu_id in enumerate(opt.gpu_ids):
        process = mp.Process(target=main, args=(i, gpu_id, len(opt.gpu_ids), opt))
        process.start()
        process_list.append(process)
    for i in process_list:
        process.join()   
    print("Done!")

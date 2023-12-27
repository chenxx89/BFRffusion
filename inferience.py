import os
import torch
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser
import cv2
from pytorch_lightning import seed_everything
from models.models import load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import glob
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils import imwrite

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(opt):
    # ------------------------ input & output ------------------------
    if opt.input.endswith('/'):
        opt.input = opt.input[:-1]
    if os.path.isfile(opt.input):
        img_list = [opt.input]
    else:
        img_list = sorted(glob.glob(os.path.join(opt.input, '*')))

    os.makedirs(opt.output, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if opt.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=opt.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None


    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # initialize face helper
    face_helper = FaceRestoreHelper(
        opt.sr_scale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device,
        model_rootpath='weights')

    seed_everything(opt.seed)
    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(load_state_dict(opt.ckpt),strict= True)
    ddim_sampler = DDIMSampler(model)
    H = W = opt.image_size
    shape = (4, H // 8, W // 8)

    # ------------------------ restore ------------------------
    for img_path in img_list:
        face_helper.clean_all()

        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if opt.aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=opt.only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration
        for cropped_face in face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                cond = {"c_concat": [cropped_face_t], "c_crossattn": []}
                samples, _ = ddim_sampler.sample(opt.ddim_steps, 1,
                                                shape, cond, verbose=False)
                
                hq_imgs =model.decode_first_stage(samples)

                # convert to image
                restored_face = tensor2img(hq_imgs.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        if not opt.aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=opt.sr_scale)[0]
            else:
                bg_img = None

            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        else:
            restored_img = None

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(opt.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if opt.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{opt.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(opt.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(opt.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if opt.ext == 'auto':
                extension = ext[1:]
            else:
                extension = opt.ext

            if opt.suffix is not None:
                save_restore_path = os.path.join(opt.output, 'restored_imgs', f'{basename}_{opt.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(opt.output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{opt.output}] folder.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument("--ckpt", type=str, default='experiments/weights/checkpoint_BFRffusion_FFHQ.ckpt', help='dir of ckpt to load')
    parser.add_argument("--config",type=str,default="options/test.yaml",help="path to config which constructs model")
    parser.add_argument("--ddim_steps",type=int,default=50,help="number of ddpm sampling steps")
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument("--sr_scale", type=float, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='Background upsampler.')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument("--image_size", type=int, default=512, help='Image size as the model input.')
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()

    main(opt)

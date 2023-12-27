import argparse, os, sys, datetime, glob
from pytorch_lightning.trainer import Trainer
from models.models import load_state_dict
import datetime
import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from ldm.util import instantiate_from_config
from models.logger import logs_callbacks

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="FFHQ",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['options/train.yaml'],
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="experiments",
        help="directory for logging dat shit",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)


    # # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    if opt.resume:
        trainer_opt.resume_from_checkpoint = opt.resume_from_checkpoint
        print(f"Resuming from {trainer_opt.resume_from_checkpoint}...")
    else:
        if config.model.checkpoint_path:
            model.load_state_dict(load_state_dict(config.model.checkpoint_path, location='cpu'),strict= False)

    # callbacks
    trainer_kwargs = logs_callbacks(lightning_config, ckptdir, logdir, cfgdir, opt.resume, now, config)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs,)
    trainer.logdir = logdir  

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("--------------- Data ---------------")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    model.learning_rate = config.model.base_learning_rate
    print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run
    trainer.fit(model, data)

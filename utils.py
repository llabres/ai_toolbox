import os
import random
import argparse
import datetime

import yaml
import torch

from transformers import get_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='AI toolbox framework')

    # Path to config file to use as default
    parser.add_argument('--config-path', type=str, help='Path to yaml config file')

    # Model and Dataset
    parser.add_argument('-m', '--model', type=str, help='Model name')
    parser.add_argument('--model-checkpoint', type=str, help='Path to model weights.')

    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('-ebs', '--eval-batch-size', type=int, help='DataLoader batch size for evaluation.')

    # Iterations
    parser.add_argument('--n_iterations', type=int, help='Number of train iterations.')
    parser.add_argument('--save_every', type=int, help='Number of iterations to save a checkpoint.')

    # Epochs
    parser.add_argument('--n_epochs', type=int, help='Number of train epochs. Only used if number of iterations "--n_iterations" has not been set.')

    # Optional
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
    parser.add_argument('--eval-start', action='store_true', help='Whether to evaluate the model before training or not.', default=None)
    parser.add_argument('--only-keep-best', action='store_true', help='Whether to only keep the best model, instead of all checkpoints.', default=None)
    
    # wandb
    parser.add_argument('--project-name', type=str, help='Name of the project in wandb.')
    parser.add_argument('--wandb', action='store_true', help='Whether to enable wandb logging.', default=False)
    
    # multi-gpu (only used in train_parallel.py)
    parser.add_argument('--num-nodes', type=int, help='Number of available nodes/hosts')
    parser.add_argument('--node-id', type=int, help='Unique ID to identify the current node/host')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs in each node')

    # Resume previous experiment, if used, every other argument is ignored
    parser.add_argument('--resume', type=str, help='Path to Experiment Checkpoint', default=None)

    # Pass any other argument
    _, unknown = parser.parse_known_args()
    for i, arg in enumerate(unknown):
        if arg.startswith("--"):
            if i+1 == len(unknown) or unknown[i+1].startswith("--"):
                parser.add_argument(arg, action='store_true')
            else:
                if "." in unknown[i+1]:
                    parser.add_argument(arg, type=float)
                else:
                    try:
                        int(unknown[i+1])
                        parser.add_argument(arg, type=int)
                    except:
                        parser.add_argument(arg, type=str)
                
    return parser.parse_args()


def load_config(args): 
    if args.resume:
        config = yaml.safe_load(open(os.path.join(args.resume, 'config.yml'), "r"))
        config['model_checkpoint'] = os.path.join(args.resume, 'model.ckpt')
        config['optimizer_checkpoint'] = os.path.join(args.resume, 'optimizer.ckpt')
        config['lr_scheduler_checkpoint'] = os.path.join(args.resume, 'lr_scheduler.ckpt')
        config['dataset_checkpoint'] = os.path.join(args.resume, 'dataset.ckpt')

        return config
    
    config = yaml.safe_load(open(args.config_path, "r")) if args.config_path else {}
    config['model_checkpoint'] = os.path.join('models', args.model.replace('-', '').replace('base', '').replace('large', ''), f"{args.model.lower()}")

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    config |= args
    config['use_images'] = True if config['model'] in ['MP-Pix2Struct', 'MP-Pix2Struct-base', 'MP-Pix2Struct-large'] else False
    config['use_ocr'] = True if config['model'] in ['MP-VT5', 'MP-VT5-base', 'MP-VT5-large'] else False
    config['gradient_accumulation_steps'] = config.get('gradient_accumulation_steps', 1)

    config['eval'] = True if 'eval_batch_size' in config.keys() else False
    config['n_epochs'] = config['n_epochs'] if 'n_epochs' in config else config['n_iterations']//config['save_every']

    config['experiment_name'] = f"{config['model']}_{config['dataset']}_{datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
    config['wandb_id'] = None
    config['current_epoch'] = 0

    config['device'] = config.get('device', 'cuda')
    
    return config

def build_model(config):
    if config['model'] in ['MP-VT5', 'MP-VT5-base', 'MP-VT5-large']:
        from transformers import T5Config
        from models.MPVT5.mp_vt5 import MPVT5

        model_config = T5Config.from_pretrained(config['model_checkpoint']).to_dict()
        for key in config.keys():
            if key in model_config:
                model_config[key] = config[key]

        model_config = T5Config.from_dict(model_config)
        model = MPVT5.from_pretrained(config['model_checkpoint'], config=model_config)
    
    elif config['model'] == 'MP-Pix2Struct':
        from transformers import Pix2StructConfig
        from models.MPPix2Struct.mp_pix2struct import MPPix2Struct

        model_config = Pix2StructConfig.from_pretrained(config['model_checkpoint'])
        for key in config.keys():
            if key in model_config:
                model_config[key] = config[key]
            if key in model_config['vision_config']:
                model_config['vision_config'][key] = config[key]
        
        model_config = Pix2StructConfig.from_dict(model_config)
        model = MPPix2Struct.from_pretrained(config['model_checkpoint'], config=model_config)

    else:
        raise NotImplementedError(f"Model {config['model']} not implemented.")            
    return model

def build_dataset(config, split):
    if split != 'train':
        max_pages = config['max_pages']
        config['max_pages'] = config.get('eval_max_pages', max_pages)

    if config['dataset'] == 'MP-DocVQA':
        from my_datasets.mp_docvqa import build_mp_docvqa
        data_dir = os.path.join(config['data_dir'], 'MP-DocVQA')
        config['gt_answers'] = False if split == 'train' else True
        dataset = build_mp_docvqa(data_dir, split, config)
    
    elif config['dataset'] == 'Docmatix':
        from my_datasets.docmatix import build_docmatix
        data_dir = os.path.join(config['data_dir'], 'Docmatix')
        dataset = build_docmatix(data_dir, split, config)
        dataset = dataset.shuffle(buffer_size=1000, seed=random.randint(0, 1000))

    elif config['dataset'] == 'cauldron':
        from my_datasets.the_cauldron import build_cauldron
        data_dir = os.path.join(config['data_dir'], 'the_cauldron')
        dataset = build_cauldron(data_dir, split, config)
        dataset = dataset.shuffle(buffer_size=1000, seed=random.randint(0, 1000))

    if 'dataset_checkpoint' in config.keys():
        dataset.load_state_dict(torch.load(config['dataset_checkpoint']))

    if split != 'train':
        config['max_pages'] = max_pages
    
    return dataset

def build_optimizer(config, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
    n_training_steps = config['n_epochs'] * config['iter_per_epoch']

    if 'warmup_steps' in config.keys():
        num_warmup_steps = config['warmup_steps']
    elif 'warmup_ratio' in config.keys():
        num_warmup_steps = int(config['warmup_ratio'] * n_training_steps)
    else:
        num_warmup_steps = 0

    lr_scheduler = get_scheduler(
        name=config['scheduler_name'], optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_training_steps)
    
    if 'lr_scheduler_checkpoint' in config.keys():
        lr_scheduler.load_state_dict(torch.load(config['lr_scheduler_checkpoint']))
    if 'optimizer_checkpoint' in config.keys():
        optimizer.load_state_dict(torch.load(config['optimizer_checkpoint']))
    
    return optimizer, lr_scheduler

def build_logger(config):
    from logger import Logger
    return Logger(config)
        
def save_checkpoint(config, model, dataset, optimizer, lr_scheduler, logger, epoch, is_best):
    if is_best or not config['only_keep_best']:
        experiment_dir = os.path.join(config['save_dir'], 'checkpoints', config['experiment_name'])
        os.makedirs(experiment_dir, exist_ok=True)
        
        config['current_epoch'] = epoch + 1
        config['wandb_id'] = logger.wandb.id if logger.use_wandb else None
        device = config.pop('device')
        save_yaml(os.path.join(experiment_dir, 'config.yml'), config)

        model.save_pretrained(os.path.join(experiment_dir, f"model.ckpt" if config['only_keep_best'] else f"model_{epoch}.ckpt"))
        model.tokenizer.save_pretrained(os.path.join(experiment_dir, f"model.ckpt" if config['only_keep_best'] else f"model_{epoch}.ckpt"))
        torch.save(optimizer.state_dict(), os.path.join(experiment_dir, "optimizer.ckpt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(experiment_dir, "lr_scheduler.ckpt"))
        torch.save(dataset.state_dict(), os.path.join(experiment_dir, "dataset.ckpt"))
        config['device'] = device
       
    
def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
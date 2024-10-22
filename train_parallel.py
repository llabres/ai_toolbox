import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node

from tqdm import tqdm

from eval import evaluate_parallel
from utils import parse_args, load_config, build_model, build_dataset, build_optimizer, build_logger, save_checkpoint

def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, logger, epoch, global_rank):
    model.train()

    pbar = tqdm(range(config['iter_per_epoch'])) if global_rank == 0 else range(config['iter_per_epoch'])
    i = 0
    losses = {'Train/Batch Loss': 0}
    while i < config['iter_per_epoch']:
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(config['device']) for k, v in batch.items() if v is not None}
            outputs = model.forward(**batch)
            loss = outputs.loss
            loss = loss / config['gradient_accumulation_steps']
            losses['Train/Batch Loss'] += loss.item()
            loss.backward()

            if ((batch_idx + 1) % config['gradient_accumulation_steps'] == 0) or (batch_idx + 1 == config['iter_per_epoch']):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
              
                if config['wandb']:
                    logger.wandb.log(losses | {'lr': optimizer.param_groups[0]['lr']}, step=i+epoch*config['iter_per_epoch'])
                
                if global_rank == 0:
                    pbar.set_description(f"Loss: {losses['Train/Batch Loss']:.4f}")
                    pbar.update(1)

                i += 1
                losses = {'Train/Batch Loss': 0}

                if i == config['iter_per_epoch']:
                    break


def train(local_rank, config):
    WORLD_SIZE = config['num_gpus'] * config['num_nodes']
    global_rank = config['node_id'] * config['num_gpus'] + local_rank 
    dist.init_process_group( 
        backend='nccl',  
        world_size=WORLD_SIZE, 
        rank=global_rank 
    )
    config['wandb'] = config['wandb'] and global_rank == 0
    config['device'] = torch.device("cuda:" + str(local_rank))

    model = build_model(config)
    model.to(config['device'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    logger = build_logger(config)
    logger.log_model_parameters(model)
    dist.barrier()

    train_dataset = build_dataset(config, 'train')
    train_dataset = split_dataset_by_node(train_dataset, world_size=WORLD_SIZE, rank=global_rank)
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=model.module.collator, num_workers=2, pin_memory=True)
    config['iter_per_epoch'] = config['save_every']

    optimizer, lr_scheduler = build_optimizer(config, model)

    is_best = True
    if config['eval']:
        eval_dataset = build_dataset(config, 'val')
        eval_dataset = split_dataset_by_node(eval_dataset, world_size=WORLD_SIZE, rank=global_rank)
        eval_data_loader = DataLoader(eval_dataset, batch_size=config['eval_batch_size'], collate_fn=model.module.collator, num_workers=2, pin_memory=True)
        if config['eval_start']:
            dist.barrier()
            model.module.set_eval_mode(max_pages=config['eval_max_pages'])
            is_best = evaluate_parallel(config, model, eval_data_loader, logger, global_rank)
            model.module.set_train_mode(max_pages=config['max_pages'])
    
    for epoch in range(config['current_epoch'], config['n_epochs']):
        
        train_one_epoch(config, model, train_data_loader, optimizer, lr_scheduler, logger, epoch, global_rank)
        
        if config['eval']:
            dist.barrier()
            model.module.set_eval_mode(max_pages=config['eval_max_pages'])
            is_best = evaluate_parallel(config, model, eval_data_loader, logger, global_rank)
            model.module.set_train_mode(max_pages=config['max_pages'])

        dist.barrier()
        if global_rank == 0:
            save_checkpoint(config, model.module, train_dataset, optimizer, lr_scheduler, logger, epoch, is_best)



if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '9956'
    torch.multiprocessing.spawn(train, nprocs=config['num_gpus'], args=(config,))
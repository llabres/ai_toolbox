from tqdm import tqdm

from eval import evaluate
from utils import parse_args, load_config, build_model, build_dataset, build_optimizer, build_logger, save_checkpoint

from torch.utils.data import DataLoader

def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, logger, epoch):
    model.train()

    pbar = tqdm(range(config['iter_per_epoch']))
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
                
                i += 1
                pbar.set_description(f"Loss: {losses['Train/Batch Loss']:.4f}")
                pbar.update(1)

                losses = {'Train/Batch Loss': 0}

                if i == config['iter_per_epoch']:
                    break

def train(config):

    model = build_model(config)
    model.to(config['device'])

    logger = build_logger(config)
    logger.log_model_parameters(model)

    train_dataset = build_dataset(config, 'train')
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=model.collator, num_workers=0, pin_memory=True)
    config['iter_per_epoch'] = config['save_every']

    optimizer, lr_scheduler = build_optimizer(config, model)

    is_best = True
    if config['eval']:
        eval_dataset = build_dataset(config, 'val')
        eval_data_loader = DataLoader(eval_dataset, batch_size=config['eval_batch_size'], collate_fn=model.collator, num_workers=0, pin_memory=True)
        if config['eval_start']:
            model.set_pages(config['eval_max_pages'])
            is_best = evaluate(config, model, eval_data_loader, logger)
            model.set_pages(config['max_pages'])
    
    for epoch in range(config['current_epoch'], config['n_epochs']):
        
        train_one_epoch(config, model, train_data_loader, optimizer, lr_scheduler, logger, epoch)
        
        if config['eval']:
            model.set_pages(config['eval_max_pages'])
            is_best = evaluate(config, model, eval_data_loader, logger)
            model.set_pages(config['max_pages'])

        save_checkpoint(config, model, train_dataset, optimizer, lr_scheduler, logger, epoch, is_best)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    train(config)

        
        


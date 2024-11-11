from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.data_loader import prepare_data_loader

from tqdm import tqdm

from eval import evaluate
from utils import parse_args, load_config, build_model, build_dataset, build_optimizer, build_logger, save_checkpoint

from torch.utils.data import DataLoader

def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, logger, epoch, accelerator):
    model.train()

    pbar = tqdm(range(config['iter_per_epoch']), disable=not accelerator.is_main_process)
    i = 0
    while i < config['iter_per_epoch']:
        for batch_idx, batch in enumerate(data_loader):
            with accelerator.accumulate(model):
                outputs = model.forward(**batch)
                loss = outputs.loss
            
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if config['wandb']:
                    accelerator.log({'Train/Batch Loss': loss.item()} | {'lr': optimizer.param_groups[0]['lr']}, step=i+epoch*config['iter_per_epoch'])
                    
                i += 1
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

                if i == config['iter_per_epoch']:
                    break

def train(config, accelerator):

    model = build_model(config)
    model.to(config['device'])

    logger = build_logger(config)
    logger.log_model_parameters(model)

    config['Model Params'] = logger.total_params
    config['Model Trainable Params'] = logger.trainable_params

    if config['wandb']:
        accelerator.init_trackers(
            project_name=config['project_name'],
            config=config,
            init_kwargs={
                'wandb': {
                    'name': config['experiment_name'],
                    'tags': [config['model'], config['dataset'], logger.machine],
                    'id': config['wandb_id'],
                    'dir': config['save_dir'],
                    'resume': 'allow'
                }
            }
        )

    train_dataset = build_dataset(config, 'train')
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=model.collator, num_workers=4, pin_memory=True)
    config['iter_per_epoch'] = config['save_every']

    optimizer, lr_scheduler = build_optimizer(config, model)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    train_data_loader = prepare_data_loader(train_data_loader, split_batches=True)

    is_best = True
    if config['eval']:
        eval_dataset = build_dataset(config, 'val')
        eval_data_loader = DataLoader(eval_dataset, batch_size=config['eval_batch_size'], collate_fn=model.module.collator, num_workers=4, pin_memory=True)
        eval_data_loader = prepare_data_loader(eval_data_loader, split_batches=True)
        if config['eval_start']:
            model.module.set_eval_mode(max_pages=config['eval_max_pages'])
            is_best = evaluate(config, model, eval_data_loader, logger, accelerator)
            model.module.set_train_mode(max_pages=config['max_pages'])
    
    for epoch in range(config['current_epoch'], config['n_epochs']):
        
        train_one_epoch(config, model, train_data_loader, optimizer, lr_scheduler, logger, epoch, accelerator)
        
        if config['eval']:
            model.module.set_eval_mode(max_pages=config['eval_max_pages'])
            is_best = evaluate(config, model, eval_data_loader, logger, accelerator)
            model.module.set_train_mode(max_pages=config['max_pages'])

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb")
            run_id = wandb_tracker.run.id if config['wandb'] else None
            save_checkpoint(config, accelerator.unwrap_model(model), train_dataset, optimizer, lr_scheduler, logger, epoch, is_best, wandb_id=run_id)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=config['mixed_precision'],
                              gradient_accumulation_steps=config['gradient_accumulation_steps'],
                              step_scheduler_with_optimizer=False,
                              log_with=('wandb' if config['wandb'] else None),
                              kwargs_handlers=[ddp_kwargs])
    
    config['batch_size'] = config['batch_size']*accelerator.num_processes
    if config['eval']:
        config['eval_batch_size'] = config['eval_batch_size']*accelerator.num_processes

    config['device'] = accelerator.device 
    
    train(config, accelerator)

    accelerator.end_training()
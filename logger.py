import socket

class Logger:
    def __init__(self, config):
        self.use_wandb = config['wandb']
        self.best_metrics = None

        machine = socket.gethostname()
        if not machine.startswith('cvc') and not machine.startswith('cuda'):
            machine = 'marenostrum5'

        if self.use_wandb:
            import wandb as wb
            tags = [config['model'], config['dataset'], machine]
            self.wandb = wb.init(project=config['project_name'], name=config['experiment_name'], dir=config['save_dir'], tags=tags, config=config, id=config['wandb_id'], resume="allow")
    
    def log_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.use_wandb:
            self.wandb.config.update({
                'Model Params': int(total_params / 1e6),  # In millions
                'Model Trainable Params': int(trainable_params / 1e6)  # In millions
            })

    def update_best(self, metrics):
        if not self.best_metrics:
            self.best_metrics = metrics
            return True

        if metrics['ANLS'] >= self.best_metrics['ANLS']:
            self.best_metrics = metrics
            return True

        return False


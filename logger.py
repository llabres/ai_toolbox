import os
import json
import socket

class Logger:
    def __init__(self, config):
        self.use_wandb = config['wandb']
        self.best_metrics = None

        machine = socket.gethostname()
        if not machine.startswith('cvc') and not machine.startswith('cuda'):
            machine = 'marenostrum5'

        self.machine = machine
    
    def log_model_parameters(self, model):
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def update_best(self, metrics):
        if not self.best_metrics:
            self.best_metrics = metrics
            return True

        if metrics['ANLS'] >= self.best_metrics['ANLS']:
            self.best_metrics = metrics
            return True

        return False

    def save_answers(self, gt_answers, preds, config):
        experiment_dir = os.path.join(config['save_dir'], 'checkpoints', config['experiment_name'])
        os.makedirs(experiment_dir, exist_ok=True)

        with open(os.path.join(experiment_dir, 'answers.json'), 'a') as f:
            for i in range(len(gt_answers)):
                data = {
                    'gt_answers': gt_answers[i],
                    'preds': preds[i],
                }
                f.write(json.dumps(data) + '\n')



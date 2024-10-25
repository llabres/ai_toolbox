import torch
import editdistance

from tqdm import tqdm

from utils import parse_args, load_config, build_model, build_dataset, build_logger

from transformers import GenerationConfig, StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria

from accelerate.utils import gather_object

def get_metrics(gt_labels, pred_labels):
    batch_accuracy = []
    batch_anls = []


    gt = [[preprocess_str(gt_elm) for gt_elm in gt_answers] for gt_answers in gt_labels]
    pred = [preprocess_str(pred) for pred in pred_labels]

    for i in range(len(gt_labels)):
        batch_accuracy.append(calculate_accuracy(gt[i], pred[i]))
        batch_anls.append(calculate_anls(gt[i], pred[i]))

    return {'Accuracy': batch_accuracy, 'ANLS': batch_anls}

def preprocess_str(string):
    string = string.lower()
    return string.strip()

def calculate_accuracy(gt, pred):
        for gt_elm in gt:
            if gt_elm == pred:
                return 1
        return 0

def calculate_anls(gt, pred):
    if len(pred) == 0:
        return 0

    answers_similarity = [1 - editdistance.eval(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
    max_similarity = max(answers_similarity)

    anls = max_similarity if max_similarity >= 0.5 else 0
    return anls

def evaluate(config, model, eval_data_loader, logger, accelerator):
    model.eval()
    pbar = tqdm(eval_data_loader, disable=not accelerator.is_local_main_process)
    
    total_samples = 0
    metrics = {'Accuracy': [], 'ANLS': []}
    
    with torch.no_grad():
        for batch in pbar:
            bs = batch['labels'].shape[0] 
            batch.pop('labels')
            batch.pop('decoder_attention_mask')
            gt_answers = batch.pop('gt_answers')
            gt_answer_page = batch.pop('gt_answer_page') if 'gt_answer_page' in batch.keys() else None
            batch = {k: v.to(config['device']) for k, v in batch.items()}
            outputs = model.module.generate(**batch, output_scores=True, output_attentions=False, return_dict_in_generate=True, max_new_tokens=20)

            preds = model.module.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)

            batch_metrics = get_metrics(gt_answers, preds) 
          
            pbar.set_description(" - ".join([f"{k}: {sum(v)/bs:.4f}" for k, v in batch_metrics.items()]))
            
            for k, v in batch_metrics.items():
                metrics[k].extend(v)

            total_samples += bs

            if config['save_answers']:
                logger.save_answers(gt_answers, preds, config)

            if config['debug'] and total_samples >= 10:
                break


    for k, v in metrics.items():
        metrics[k] = gather_object(v)
    total_samples = sum(gather_object([total_samples]))

    total_metrics = {k: sum(v)/total_samples for k, v in metrics.items()}

    if config['wandb']:
        accelerator.log({f'Eval/{k}': v for k, v in total_metrics.items()})

    return logger.update_best(total_metrics)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args, eval_only=True)
    
    config['max_pages'] = config['eval_max_pages']

    model = build_model(config)
    model.set_pages(config['eval_max_pages'])
    model.to(config['device'])

    logger = build_logger(config)
    logger.log_model_parameters(model)

    eval_dataset = build_dataset(config, 'val')

    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['eval_batch_size'], collate_fn=model.collator, num_workers=2, pin_memory=True)

    evaluate(config, model, eval_data_loader, logger)

        



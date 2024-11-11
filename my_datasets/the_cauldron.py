import os
import random
from datasets import load_dataset, interleave_datasets
from .mp_docvqa import format_data as mp_format_data
from .pfl_docvqa import format_data as pfl_format_data

def format_data(sample):        
    question_idx = random.randint(0, len(sample['texts']) - 1)
    sample['question'] = sample['texts'][question_idx]['user']
    sample['label'] = sample['texts'][question_idx]['assistant']

    images = []
    for image in sample['images']:
        image_size = image.size
        if image_size[0] > 800 or image_size[1] > 800:
            scale = 800 / max(image_size)
            image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
            image = image.resize(image_size)
        images.append(image)
    sample['images'] = [images]

    return sample



def build_cauldron(data_dir, split, config):
    assert split == 'train', 'Only train split is available for The Cauldron dataset'

    datasets = []
    for dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, dir)) and dir not in ["cordv2", "rendered_text"]:
            dataset = load_dataset(os.path.join(data_dir, dir), streaming=True, split='train')
            datasets.append(dataset)

    dataset = interleave_datasets(datasets, stopping_strategy="all_exhausted")
    dataset = dataset.map(format_data, remove_columns=['texts'])
    datasets = [dataset]
    
    docmatix = load_dataset(os.path.join(config['data_dir'], 'Docmatix', 'data'), streaming=True, split='train')
    docmatix = docmatix.map(format_data, remove_columns=['texts'])
    datasets.append(docmatix)

    data_files = {"train": "train-*.parquet"}
    mp_docvqa = load_dataset(os.path.join(config['data_dir'], 'MP-DocVQA', 'data'), data_files=data_files, streaming=True, split='train')
    mp_docvqa = mp_docvqa.map(mp_format_data, fn_kwargs={'max_pages': config['max_pages'], 'use_images': True, 'use_ocr': False, 'gt_answers': False}, remove_columns=['questions', 'doc_pages', 'images_id', 'ocr_tokens', 'ocr_boxes'])
    datasets.append(mp_docvqa)

    pfl_docvqa = load_dataset(os.path.join(config['data_dir'], 'PFL-DocVQA', 'data'), data_files=data_files, streaming=True, split='train')
    pfl_docvqa = pfl_docvqa.map(pfl_format_data, fn_kwargs={'use_images': True, 'use_ocr': False}, remove_columns=['questions', 'images_id', 'key_value_pairs', 'page', 'ocr_tokens', 'ocr_boxes'])
    datasets.append(pfl_docvqa)

    dataset = interleave_datasets(datasets, probabilities=[0.5, 0.2, 0.1, 0.2], stopping_strategy="all_exhausted")
    
    return dataset
    
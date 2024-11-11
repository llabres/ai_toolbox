import os
import random
from datasets import load_dataset

def format_data(sample, max_pages):
    question_idx = random.randint(0, len(sample['texts']) - 1)
    sample['question'] = sample['texts'][question_idx]['user']
    sample['label'] = sample['texts'][question_idx]['assistant']
    images = []
    for image in sample['images']:
        image_size = image.size
        scale = 1024 / max(image_size)
        image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
        image = image.resize(image_size)
        images.append(image)
    sample['images'] = [images[:max_pages]]

    return sample

def filter_data_ocr(sample):
    if sample['ocr_tokens'] is None:
        return False
    elif len(sample['ocr_tokens']) == 0:
        return False
    return True

def build_docmatix(data_dir, split, config):
    assert split == 'train', 'Only train split is available for Docmatix dataset'
    data_dir = os.path.join(data_dir, 'data' if config['use_images'] else 'ocr')
    dataset = load_dataset(data_dir, split=split, streaming=True)
    
    if config['use_ocr']:
        dataset = dataset.filter(filter_data_ocr)
    
    dataset = dataset.map(format_data, remove_columns=['texts'], fn_kwargs={'max_pages': config['max_pages']})

    return dataset
import os
import random
from datasets import load_dataset, interleave_datasets  

def format_data(sample):
    question_idx = random.randint(0, len(sample['texts']) - 1)
    sample['question'] = sample['texts'][question_idx]['user']
    sample['label'] = sample['texts'][question_idx]['assistant']

    return sample


def build_cauldron(data_dir, split, config):
    assert split == 'train', 'Only train split is available for The Cauldron dataset'

    datasets = []
    for dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, dir)) and dir != "cordv2":
            datasets.append(load_dataset(os.path.join(data_dir, dir), streaming=True, split='train'))

    dataset = interleave_datasets(datasets, stopping_strategy="all_exhausted")

    dataset = dataset.map(format_data, remove_columns=['texts'])

    return dataset
    
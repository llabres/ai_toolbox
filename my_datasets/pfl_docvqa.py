import os
import io
import random
from PIL import Image
from datasets import load_dataset

def format_data(sample, use_images, use_ocr):
    idx = random.randint(0, len(sample['questions'])-1)

    images = []
    ocr_tokens = []
    ocr_boxes = []
    
    if use_images:
        for page in range(len(sample['images'])):
            image = Image.open(io.BytesIO(sample['images'][page]))
            image_size = image.size
            scale = 1024 / max(image_size)
            image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
            image = image.resize(image_size)
            images.append(image)

        
        sample['images'] = [images]

    if use_ocr:
        for page in range(len(sample['ocr_tokens'])):
            ocr_tokens.append(sample['ocr_tokens'][page])
            ocr_boxes.append(sample['ocr_boxes'][page])
        
        sample['ocr_tokens'] = ocr_tokens
        sample['ocr_boxes'] = ocr_boxes
                    
    sample['question'] = sample['questions'][idx]['question']
    sample['label'] = random.choice(sample['questions'][idx]['answers'])
    sample['key_value_pairs'] = None


    return sample

def build_pfl_docvqa(data_dir, split, config):
    data_files = {"train": "train-*.parquet"}
    dataset = load_dataset(os.path.join(data_dir, 'data'), data_files=data_files, split=split, streaming=True)

    remove_columns = ['questions', 'images_id', 'key_value_pairs', 'page']
    remove_columns += ['images'] if not config['use_images'] else []
    remove_columns += ['ocr_tokens', 'ocr_boxes'] if not config['use_ocr'] else []
    dataset = dataset.map(format_data, fn_kwargs={'use_images': config['use_images'], 'use_ocr': config['use_ocr']}, remove_columns=remove_columns)

    return dataset
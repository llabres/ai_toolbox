import os
import io
import random
from PIL import Image
from datasets import load_dataset


def format_data(sample, max_pages, use_images, use_ocr, gt_answers):
    idx = random.randint(0, len(sample['questions'])-1)
    answer_page = sample['questions'][idx]['answer_page_idx']
    n_pages = len(sample['images'])

    images = []
    ocr_tokens = []
    ocr_boxes = []
    
    if n_pages <= max_pages:
        first_page, last_page = 0, n_pages

    else:
        first_page_lower_bound = max(0, answer_page-max_pages+1)
        first_page_upper_bound = answer_page
        first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
        last_page = first_page + max_pages

        if last_page > n_pages:
            last_page = n_pages
            first_page = last_page-max_pages

    assert answer_page in range(first_page, last_page)
    
    if use_images:
        for page in range(first_page, last_page):
            images.append(Image.open(io.BytesIO(sample['images'][page])))
        
        sample['images'] = images

    if use_ocr:
        for page in range(first_page, last_page):
            ocr_tokens.append(sample['ocr_tokens'][page])
            ocr_boxes.append(sample['ocr_boxes'][page])
        
        sample['ocr_tokens'] = ocr_tokens
        sample['ocr_boxes'] = ocr_boxes
                    
    sample['question'] = sample['questions'][idx]['question']
    sample['label'] = random.choice(sample['questions'][idx]['answers'])


    sample['answer_page_idx'] = answer_page-first_page
    
    if gt_answers:
        sample['gt_answers'] = sample['questions'][idx]['answers']
        sample['gt_answer_page'] = answer_page-first_page

    return sample
    

def build_mp_docvqa(data_dir, split, config):
    data_files = {"train": "train-*.parquet", "val": "val-*.parquet", "test": "test-*.parquet"}
    dataset = load_dataset(os.path.join(data_dir, 'data'), data_files=data_files, split=split, streaming=True)
    remove_columns = ['questions', 'doc_pages', 'images_id']
    remove_columns += ['images'] if not config['use_images'] else []
    remove_columns += ['ocr_tokens', 'ocr_boxes'] if not config['use_ocr'] else []
    dataset = dataset.map(format_data, fn_kwargs={'max_pages': config['max_pages'], 'use_images': config['use_images'], 'use_ocr': config['use_ocr'], 'gt_answers': config['gt_answers']}, remove_columns=remove_columns)

    return dataset

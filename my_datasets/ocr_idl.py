import io
import os
import random

from PIL import Image
from datasets import load_dataset

questions = [
    "Count the amount of times '<span>' appears in the documents",
    "How many times does '<span>' appear in the documents?",
    "Return the number of times '<span>' is in the documents",
    "How many '<span>' are in the documents?",
    "Count the number of '<span>' in the documents",
    "What is the count of '<span>' in the documents?",
    "Calculate how often '<span>' appears in the documents.",
    "Determine the frequency of '<span>' in the documents.",
    "Find the occurrence count of '<span>' in the documents.",
    "What is the total number of '<span>' in the documents?",
    "How frequently does '<span>' occur in the documents?",
    "Count occurrences of '<span>' across the documents.",
    "Get the count of '<span>' in the documents.",
    "Check how many instances of '<span>' are in the documents.",
    "Provide the total occurrences of '<span>' in the documents.",
    "Identify how many times '<span>' appears in the documents.",
    "Count all instances of '<span>' within the documents.",
    "What is the total count of '<span>' found in the documents?",
    "How many appearances of '<span>' exist in the documents?",
    "Count how many '<span>' can be found in the documents.",
    "What is the number of occurrences for '<span>' in the documents?",
    "Tally the number of times '<span>' appears in the documents.",
    "Assess how many '<span>' are present across the documents.",
    "Find the total count of '<span>' occurrences in the documents.",
    "Calculate the total times '<span>' shows up in the documents."
]


def format_data(samples):
    # from dict of list to list of dict
    samples = [dict(zip(samples,t)) for t in zip(*samples.values())]
    samples = [sample for sample in samples if len(sample['ocr_tokens'][0]) > 3] 
    new_samples = {
        'question': [],
        'label': [],
        'images': []
    }
    for sample1, sample2 in zip(samples[::2], samples[1::2]):
        rand = random.random()
        if rand < 0.33:
            # Pick a random document
            random_doc = random.choice(samples)
            # Pick a random span from the document
            span_start = random.randint(0, len(random_doc['ocr_tokens'][0]) - 3)
            random_span = random_doc['ocr_tokens'][0][span_start:span_start + random.randint(1, 3)]

        elif rand < 0.66:
            # Pick a random span from the first document
            span_start = random.randint(0, len(sample1['ocr_tokens'][0]) - 3)
            random_span = sample1['ocr_tokens'][0][span_start:span_start + random.randint(1, 3)]

        else:
            # Pick a random span from the second document
            span_start = random.randint(0, len(sample2['ocr_tokens'][0]) - 3)
            random_span = sample2['ocr_tokens'][0][span_start:span_start + random.randint(1, 3)]
    
        new_samples['question'].append(random.choice(questions).replace('<span>', ' '.join(random_span)))
        new_samples['label'].append(str(' '.join(sample1['ocr_tokens'][0]).count(' '.join(random_span)) + ' '.join(sample2['ocr_tokens'][0]).count(' '.join(random_span))))
       
        image1 = Image.open(io.BytesIO(sample1['images'][0]))
        image2 = Image.open(io.BytesIO(sample2['images'][0]))
        if sample1['images_id'][0].split('_')[0] == sample2['images_id'][0].split('_')[0]:
            
            new_samples['images'].append([[image1, image2]]) # A two page document
            
        else:
            new_samples['images'].append([[image1], [image2]]) # Two single page documents

    
    return new_samples


def build_ocr_idl(data_dir, split):
    assert split == 'train', 'Only train split is available for OCR-IDL dataset'
    data_files = {"train": "train-*.parquet"}
    dataset = load_dataset(os.path.join(data_dir, 'data'), data_files=data_files, split=split, streaming=True)

    dataset = dataset.map(format_data, batched=True, remove_columns=['ocr_tokens', 'ocr_boxes', 'images_id', 'doc_pages', 'metadata'])

    return dataset
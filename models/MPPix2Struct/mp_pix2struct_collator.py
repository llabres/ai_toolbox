import torch

def process_patches(patches, max_patches):

    patches = patches[torch.std(patches[:, 2:], dim=1) > 0.1] # Remove patches with low variance    
    
    if patches.shape[0] > max_patches: # if there are more than max_patches remove extra patches at random 
        patches = patches[torch.randperm(patches.shape[0])[:max_patches]]

    elif patches.shape[0] < max_patches: # add padding
        patches = torch.cat([patches, torch.zeros((max_patches - patches.shape[0], patches.shape[1]))], dim=0)

    return patches


class MPPix2StructCollator:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = self.tokenizer.model_max_length
        self.padding = padding
        self.max_pages = config.vision_config.max_pages
        self.max_patches = config.vision_config.max_patches
        self.image_resolution = config.vision_config.image_resolution
        self.question_in_image = config.question_in_image
        self.n_global_tokens = config.vision_config.n_global_tokens
        

    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        
        batch_size = len(batch['images'])

        images = []
        for batch_idx in range(batch_size):
            pages_patches = []
            num_pages = len(batch['images'][batch_idx][:self.max_pages])
            
            try: # On Docmatix there have been problems when processing the image
                patches = self.image_processor(batch['images'][batch_idx][:self.max_pages], text=batch['question'][batch_idx] if self.question_in_image else "", return_tensors='pt', max_patches=self.image_resolution)['flattened_patches']
                if self.max_patches != self.image_resolution:
                    for page_idx in range(num_pages):                
                        pages_patches.append(process_patches(patches[page_idx], self.max_patches))
                else:
                    pages_patches = [p for p in patches]
            except:
                num_pages -= 1
                               
            if num_pages < self.max_pages:
                pages_patches += [torch.zeros_like(pages_patches[0])]*(self.max_pages-num_pages)
        
            images.extend(pages_patches)

        images = torch.stack(images, dim=0).view(batch_size, self.max_pages, self.max_patches, -1)

        if not self.question_in_image:
            question_ids = self.tokenizer(batch['question'], return_tensors="pt", padding=self.padding, truncation=True, max_length=self.max_length)
            question_attention_mask = question_ids.attention_mask
            question_ids = question_ids.input_ids
        else:
            question_ids = None
            question_attention_mask = None
        
        # Create attention mask for the image and add global tokens
        attention_mask = torch.cat([(images[:, :, :, 2] != 0), torch.ones(images.size(0), images.size(1), self.n_global_tokens)], dim=-1).to(torch.long)

        labels = self.image_processor.tokenizer(batch['label'], return_tensors="pt", padding=self.padding, truncation=True, max_length=128)
        decoder_attention_mask = labels.attention_mask
        labels = labels.input_ids


        return dict(
            flattened_patches=images,
            attention_mask=attention_mask,
            question_ids=question_ids,
            question_attention_mask=question_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            gt_answers=batch.get('gt_answers', None),
        )
        

            


        
            


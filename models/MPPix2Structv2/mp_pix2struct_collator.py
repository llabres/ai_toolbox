import torch

# images: list of lists (batch_size, num_documents, num_pages)


class MPPix2StructCollator:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patches_per_block = config.vision_config.patches_per_block
        self.patch_size = config.vision_config.patch_size
        self.random_patch_removal = config.random_patch_removal

        self.num_global_tokens = config.vision_config.num_global_tokens
        self.num_memory_tokens = config.vision_config.num_memory_tokens

    def _remove_patches(self, patches):
        patches = patches[(torch.std(patches[:, 4:][:, 0::3], dim=1) > 0.1) + (torch.std(patches[:, 4:][:, 1::3], dim=1) > 0.1) + (torch.std(patches[:, 4:][:, 2::3], dim=1) > 0.1)] # Remove patches with low variance in all three channels
        patches = patches[patches[:, 0] != 0] # Remove padding patches if any

        if self.random_patch_removal > 0:
            # remove random_patch_removal% of patches at random
            num_patches = patches.shape[0]
            num_patches_to_remove = int(self.random_patch_removal*num_patches)
            patches = patches[torch.randperm(num_patches)[:-num_patches_to_remove]]

        return patches

    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        batch_size = len(batch['images'])
        batch_patches = []
        max_blocks = 0
        for batch_idx in range(batch_size):
            documents = batch['images'][batch_idx]
            documents_patches = []
            for document_idx, document in enumerate(documents):
                num_pages = len(document) 
                image_resolution = document[0].size # All pages of a document are assumed to be the same size
                max_patches = (image_resolution[0]//self.patch_size)*(image_resolution[0]//self.patch_size)
                document_patches = self.image_processor(document, return_tensors='pt', max_patches=max_patches)['flattened_patches'] # (num_pages, max_patches, hidden_size + 2)
                page_idx = torch.arange(1, num_pages + 1).repeat(max_patches, 1).T.unsqueeze(-1) # (num_pages, max_patches, 1)
                document_idx = torch.full((num_pages, max_patches), document_idx + 1).unsqueeze(-1) # (num_pages, max_patches, 1)
                document_patches = torch.cat([document_patches[:, :, :2], page_idx, document_idx, document_patches[:, :, 2:]], dim=-1) # (num_pages, max_patches, hidden_size + 4)
                document_patches = self._remove_patches(document_patches.flatten(0, 1))
                documents_patches.append(document_patches)
            
            # if document_patches is not divisible by patches_per_block, add padding
            documents_patches = torch.cat(documents_patches, dim=0)
            num_patches = documents_patches.shape[0]
            if num_patches % self.patches_per_block != 0:
                padding = torch.zeros((self.patches_per_block - num_patches % self.patches_per_block, documents_patches.shape[-1]))
                documents_patches = torch.cat([documents_patches, padding], dim=0)
            
            documents_patches = documents_patches.view(-1, self.patches_per_block, documents_patches.shape[-1])
            batch_patches.append(documents_patches)
            max_blocks = max(max_blocks, documents_patches.shape[0])
        
        # Add padding to batch_patches
        for batch_idx in range(batch_size):
            if batch_patches[batch_idx].shape[0] < max_blocks:
                padding = torch.zeros((max_blocks - batch_patches[batch_idx].shape[0], self.patches_per_block, batch_patches[batch_idx].shape[-1]))
                batch_patches[batch_idx] = torch.cat([batch_patches[batch_idx], padding], dim=0)

        batch_patches = torch.stack(batch_patches, dim=0).view(-1, self.patches_per_block, documents_patches.shape[-1]) # (batch_size*max_blocks, patches_per_block, hidden_size + 4)

        question_ids = self.tokenizer(batch['question'], return_tensors="pt", padding='longest', truncation=True, max_length=self.tokenizer.model_max_length) 
        question_attention_mask = question_ids.attention_mask
        question_ids = question_ids.input_ids # (batch_size, max_length)
        
        global_attention_mask = torch.cat([question_attention_mask, torch.ones((batch_size, self.num_memory_tokens), dtype=torch.long), torch.ones((batch_size, self.num_global_tokens), dtype=torch.long)], dim=-1)
        attention_mask = (batch_patches[:, :, 0] != 0).long() # (batch_size*max_blocks, patches_per_block)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size*max_blocks, self.num_global_tokens), dtype=torch.long)], dim=-1) # (batch_size*max_blocks, patches_per_block + num_global_tokens)

        labels = self.image_processor.tokenizer(batch['label'], return_tensors="pt", padding='longest', truncation=True, max_length=128)
        decoder_attention_mask = labels.attention_mask
        labels = labels.input_ids

        return {
            'question_input_ids': question_ids,
            'global_attention_mask': global_attention_mask,
            'attention_mask': attention_mask,
            'flattened_patches': batch_patches,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'gt_answers': batch.get('gt_answers', None),
        }



            




        

            


        
            


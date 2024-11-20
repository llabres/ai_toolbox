import torch

class MPVT5Collator:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.block_length = config.block_size
        self.patch_size = config.patch_size

        self.num_global_tokens = config.num_global_tokens

    def _remove_patches(self, patches):
        patches = patches[(torch.std(patches[:, 4:][:, 0::3], dim=1) > 0.1) + (torch.std(patches[:, 4:][:, 1::3], dim=1) > 0.1) + (torch.std(patches[:, 4:][:, 2::3], dim=1) > 0.1)] # Remove patches with low variance in all three channels
        patches = patches[patches[:, 0] != 0] # Remove padding patches if any

        if self.random_patch_removal > 0:
            # remove random_patch_removal% of patches at random
            num_patches = patches.shape[0]
            num_patches_to_remove = int(self.random_patch_removal*num_patches)
            patches = patches[torch.randperm(num_patches)[:-num_patches_to_remove]]

        return patches


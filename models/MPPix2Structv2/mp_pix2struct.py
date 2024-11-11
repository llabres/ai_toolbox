from typing import Dict, List, Optional, Tuple, Union

import copy

import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import Pix2StructConfig, Pix2StructProcessor, Pix2StructVisionConfig, T5Config
from transformers.models.pix2struct.modeling_pix2struct import Pix2StructPreTrainedModel, Pix2StructLayerNorm, Pix2StructVisionLayer, Pix2StructForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
from transformers.utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput

from .mp_pix2struct_collator import MPPix2StructCollator

@dataclass
class MPPix2StructBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    global_hidden_states: torch.Tensor = None


class MPPix2StructVisionEmbeddings(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        
        self.config = config
        self.patch_embeddings = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.row_embeddings = self._sinusoidal_positional_embedding(config.num_rows, config.hidden_size//4)
        self.col_embeddings = self._sinusoidal_positional_embedding(config.num_cols, config.hidden_size//4)
        self.page_embeddings = self._sinusoidal_positional_embedding(config.num_pages, config.hidden_size//4)
        self.doc_embeddings = self._sinusoidal_positional_embedding(config.num_docs, config.hidden_size//4)

        self.text_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.global_tokens_embeddings = nn.Embedding(config.num_global_tokens, config.hidden_size)
        self.memory_embeddings = nn.Embedding(config.num_memory_tokens, config.hidden_size)
    
    @torch.no_grad()
    def _sinusoidal_positional_embedding(self, token_sequence_size: int, token_embedding_dim: int) -> torch.Tensor:
        # Source: https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6
        if token_embedding_dim % 2 != 0:
            raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

        T = token_sequence_size
        d = token_embedding_dim 

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(10000.0, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

        embeddings = embeddings.to(self.patch_embeddings.weight.device)

        return embeddings
    
    def four_d_positional_embeddings(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()
        page_indices = flattened_patches[:, :, 2].long()
        doc_indices = flattened_patches[:, :, 3].long()
        
        row_embeddings = self._sinusoidal_positional_embedding(row_indices.max()+1, self.config.hidden_size//4)[row_indices]
        col_embeddings = self._sinusoidal_positional_embedding(col_indices.max()+1, self.config.hidden_size//4)[col_indices]
        page_embeddings = self._sinusoidal_positional_embedding(page_indices.max()+1, self.config.hidden_size//4)[page_indices]
        doc_embeddings = self._sinusoidal_positional_embedding(doc_indices.max()+1, self.config.hidden_size//4)[doc_indices]
        
        # Instead of concatenate -> interleave them
        #four_d_embeddings = torch.cat([row_embeddings, col_embeddings, page_embeddings, doc_embeddings], dim=-1)
        four_d_embeddings = torch.stack([row_embeddings, col_embeddings, page_embeddings, doc_embeddings], dim=-1) # (batch_size, num_patches, hidden_size//4, 4)
        four_d_embeddings = four_d_embeddings.view(four_d_embeddings.size(0), four_d_embeddings.size(1), -1) # (batch_size, num_patches, hidden_size)

        return four_d_embeddings
    
    def forward(self, flattened_patches: torch.Tensor, question_input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = flattened_patches.size(0) // question_input_ids.size(0)

        four_d_embeddings = self.four_d_positional_embeddings(flattened_patches)
        patch_embeddings = self.patch_embeddings(flattened_patches[:, :, 4:]) # (batch_size*num_blocks, num_patches, hidden_size)
        patch_embeddings = patch_embeddings + four_d_embeddings
        
        text_embeddings = self.text_embeddings(question_input_ids) # (batch_size, num_patches, hidden_size)
        memory_embeddings = self.memory_embeddings(
            torch.arange(self.config.num_memory_tokens, device=question_input_ids.device)).unsqueeze(0).expand(question_input_ids.size(0), -1, -1) # (batch_size, num_memory_tokens, hidden_size)
        global_tokens_embeddings = self.global_tokens_embeddings(
            torch.arange(self.config.num_global_tokens, device=question_input_ids.device)).unsqueeze(0).expand(question_input_ids.size(0), -1, -1) # (batch_size, num_global_tokens, hidden_size)

        global_embeddings = torch.cat([text_embeddings, memory_embeddings, global_tokens_embeddings], dim=1) # (batch_size, num_question_tokens+num_memory_tokens+num_global_tokens, hidden_size)
        
        patch_embeddings = torch.cat([patch_embeddings, global_tokens_embeddings.repeat(num_blocks, 1, 1)], dim=1) # (batch_size*num_blocks, num_patches+num_global_tokens, hidden_size)

        return patch_embeddings, global_embeddings

class MPPix2StructVisionGlobalLayer(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        
        self.global_layer = T5Block(config.global_layer_config)
        self.num_global_tokens = config.num_global_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = global_hidden_states.size(0)
        global_tokens_embeddings = hidden_states[:, -self.num_global_tokens:, :].reshape(batch_size, -1, hidden_states.size(-1)) # (batch_size, num_blocks*num_global_tokens, hidden_size)
        global_tokens_attention_mask = attention_mask[:, -self.num_global_tokens:].reshape(batch_size, -1) # (batch_size, num_blocks*num_global_tokens)

        global_hidden_states = self.global_layer(global_hidden_states,
                                                 attention_mask=None, #global_attention_mask,
                                                 encoder_hidden_states=global_tokens_embeddings,
                                                 encoder_attention_mask=global_tokens_attention_mask,
                                                 output_attentions=output_attentions)[0]
        
        return hidden_states, global_hidden_states


class MPPix2StructVisionLocalLayer(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.local_layer = Pix2StructVisionLayer(config)
        self.num_global_tokens = config.num_global_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_blocks = hidden_states.size(0) // global_hidden_states.size(0)
        hidden_states = hidden_states[:, :-self.num_global_tokens, :] # (batch_size*num_blocks, num_patches, hidden_size)
        
        global_tokens_embeddings = global_hidden_states[:, -self.num_global_tokens:, :].repeat(num_blocks, 1, 1) # (batch_size*num_blocks, num_global_tokens, hidden_size)
        hidden_states = torch.cat([hidden_states, global_tokens_embeddings], dim=1) # (batch_size*num_blocks, num_patches+num_global_tokens, hidden_size)
        hidden_states = self.local_layer(hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions)[0]

        return hidden_states, global_hidden_states


class MPPix2StructVisionEncoder(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.config = config
        
        self.layer = nn.ModuleList()
        for _ in range(config.num_hidden_layers):     
            self.layer.append(MPPix2StructVisionGlobalLayer(config))
            self.layer.append(MPPix2StructVisionLocalLayer(config))
        self.layer.append(MPPix2StructVisionGlobalLayer(config))

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, MPPix2StructBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i//2] if head_mask is not None and i % 2 != 0 else None
           
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    global_hidden_states,
                    attention_mask,
                    global_attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, global_hidden_states, attention_mask, global_attention_mask, layer_head_mask, output_attentions)
            
            hidden_states = layer_outputs[0]
            global_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        outputs =  MPPix2StructBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            global_hidden_states=global_hidden_states)

        return outputs

class MPPix2StructVisionModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructVisionConfig
    main_input_name = "question_input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pix2StructVisionLayer"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = MPPix2StructVisionEmbeddings(config)
        self.encoder = MPPix2StructVisionEncoder(config)

        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        question_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MPPix2StructBaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, Pix2StructVisionModel

        >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 2048, 768]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        patch_embeddings, global_embeddings = self.embeddings(flattened_patches, question_input_ids)

        encoder_outputs = self.encoder(
            patch_embeddings,
            global_hidden_states=global_embeddings,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        outputs = MPPix2StructBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_hidden_states=encoder_outputs.global_hidden_states,
        )

        return outputs
    

class MPPix2Struct(Pix2StructForConditionalGeneration):
    _tied_weights_keys = ["encoder.embeddings.text_embeddings.weight", "decoder.lm_head.weight"]
    main_input_name = "question_input_ids"
    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)

        self.processor = Pix2StructProcessor.from_pretrained(config._name_or_path)
        self.tokenizer = self.processor.tokenizer

        self.collator = MPPix2StructCollator(self.tokenizer, self.processor, config)

        config.vision_config.vocab_size = config.text_config.vocab_size
        
        config.vision_config.global_layer_config = T5Config.from_dict(config.vision_config.global_layer_config)
        self.encoder = MPPix2StructVisionModel(config.vision_config)

        self.config = config
    
    def set_eval_mode(self, **kwargs):
        self.collator.random_patch_removal = 0.0
    
    def set_train_mode(self, **kwargs):
        self.collator.random_patch_removal = self.config.random_patch_removal

    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                flattened_patches=flattened_patches,
                question_input_ids=question_input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, MPPix2StructBaseModelOutput):
            encoder_outputs = MPPix2StructBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1],
                attentions=encoder_outputs[2],
                global_hidden_states=encoder_outputs[3],
            )

        hidden_states = encoder_outputs.last_hidden_state
        global_hidden_states = encoder_outputs.global_hidden_states

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = (
                decoder_attention_mask
                if decoder_attention_mask is not None
                else decoder_input_ids.ne(self.config.pad_token_id).float()
            )
            # Always attend to the first token
            decoder_attention_mask[:, 0] = 1

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=global_hidden_states,
            encoder_attention_mask=None, #global_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "flattened_patches": flattened_patches,
            "question_input_ids": question_input_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }



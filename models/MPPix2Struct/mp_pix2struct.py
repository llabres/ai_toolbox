from typing import Dict, List, Optional, Tuple, Union

import copy

import torch
import torch.nn as nn

from transformers import Pix2StructConfig, Pix2StructProcessor, Pix2StructVisionConfig
from transformers.models.pix2struct.modeling_pix2struct import Pix2StructPreTrainedModel, Pix2StructLayerNorm, Pix2StructVisionLayer, Pix2StructForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, Seq2SeqModelOutput, Seq2SeqLMOutput

from .mp_pix2struct_collator import MPPix2StructCollator
from models.modules import sinusoidal_positional_embedding


class MPPix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.global_token_embedder = nn.Embedding(config.n_global_tokens, config.hidden_size)
        self.question_embedder = nn.Embedding(config.vocab_size, config.hidden_size)

        self.config = config
        self.max_pages = config.max_pages

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor, question_ids: Optional[torch.Tensor] = None)  -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        question_embeddings = self.question_embedder(question_ids) if question_ids is not None else None
        
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, :, 0].long()
        col_indices = flattened_patches[:, :, :, 1].long()

        flattened_patches = flattened_patches[:, :, :, 2:]

        global_token_indices = torch.arange(self.config.n_global_tokens, device=flattened_patches.device).unsqueeze(0).unsqueeze(0).expand(flattened_patches.shape[0], flattened_patches.shape[1], -1)

        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)

        # `batch_size`, `max_pages`, `n_global_tokens`, `hidden_size`
        global_token_embeddings = self.global_token_embedder(global_token_indices)

        page_position = sinusoidal_positional_embedding(self.max_pages, self.config.hidden_size).to(embeddings.device)
        page_position = page_position.unsqueeze(0).repeat(embeddings.size(0), 1, 1).unsqueeze(2).repeat(1, 1, self.config.n_global_tokens, 1)
        global_token_embeddings = global_token_embeddings + page_position

        # sum all embeddings together
        embeddings = embeddings + row_embeddings + col_embeddings
        embeddings = torch.cat([embeddings, global_token_embeddings], dim=2)

        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(-1, embeddings.shape[2], embeddings.shape[3])

        return embeddings, question_embeddings

class MPPix2StructVisionLayer_global_first(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.local_layer = Pix2StructVisionLayer(config)
        
        global_layer_config = copy.deepcopy(config)
        global_layer_config.d_ff = config.global_d_ff
        global_layer_config.num_heads = config.global_num_heads
        global_layer_config.d_kv = config.global_d_kv
        self.global_layer = Pix2StructVisionLayer(global_layer_config)
        self.n_global_tokens = config.n_global_tokens
        self.max_pages = config.max_pages

    def forward(
        self,
        hidden_states: torch.Tensor,
        question_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        global_hidden_states = hidden_states[:, -self.n_global_tokens:, :].reshape(-1, self.max_pages*self.n_global_tokens, hidden_states.size(-1))
        global_attention_mask = attention_mask[:, -self.n_global_tokens:].reshape(-1, 1, 1, hidden_states.size(1))
        
        if question_hidden_states is not None:
            global_hidden_states = torch.cat([question_hidden_states, global_hidden_states], dim=1)
            global_attention_mask = torch.cat([question_attention_mask, global_attention_mask], dim=-1)
        
        global_layer_outputs = self.global_layer(global_hidden_states, global_attention_mask,  output_attentions=output_attentions)
        
        if question_hidden_states is not None:
            question_hidden_states = global_layer_outputs[0][:, :question_hidden_states.size(1), :]
            global_hidden_states = global_layer_outputs[0][:, question_hidden_states.size(1):, :]

        else:
            global_hidden_states = global_layer_outputs[0]

        hidden_states[:, -self.n_global_tokens:, :] = global_hidden_states.view(hidden_states.size(0), self.n_global_tokens, hidden_states.size(-1))


        hidden_states = self.local_layer(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)

        return hidden_states, question_hidden_states

class MPPix2StructVisionLayer_local_first(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.local_layer = Pix2StructVisionLayer(config)
        
        global_layer_config = copy.deepcopy(config)
        global_layer_config.d_ff = config.global_d_ff
        global_layer_config.num_heads = config.global_num_heads
        global_layer_config.d_kv = config.global_d_kv
        self.global_layer = Pix2StructVisionLayer(global_layer_config)
        self.n_global_tokens = config.n_global_tokens
        self.max_pages = config.max_pages

    def forward(
        self,
        hidden_states: torch.Tensor,
        question_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
            
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            hidden_states = self.local_layer(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
            
            hidden_states = hidden_states[0]
            
            global_hidden_states = hidden_states[:, -self.n_global_tokens:, :].reshape(-1, self.max_pages*self.n_global_tokens, hidden_states.size(-1))
            global_attention_mask = attention_mask[:, -self.n_global_tokens:].reshape(-1, self.max_pages*self.n_global_tokens)

            if question_hidden_states is not None:
                global_hidden_states = torch.cat([question_hidden_states, global_hidden_states], dim=1)
                global_attention_mask = torch.cat([question_attention_mask, global_attention_mask], dim=-1)
            
            global_layer_outputs = self.global_layer(global_hidden_states, global_attention_mask,  output_attentions=output_attentions)

            if question_hidden_states is not None:
                question_hidden_states = global_layer_outputs[0][:, :question_hidden_states.size(1), :]
                global_hidden_states = global_layer_outputs[0][:, question_hidden_states.size(1):, :]
            else:
                global_hidden_states = global_layer_outputs[0]
            
            hidden_states[:, -self.n_global_tokens:, :] = global_hidden_states.reshape(hidden_states.size(0), self.n_global_tokens, hidden_states.size(-1))

            return hidden_states, question_hidden_states

class MPPix2StructVisionEncoder(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.config = config
        self.n_global_tokens = config.n_global_tokens
        
        if config.global_first:
            self.layer = nn.ModuleList([MPPix2StructVisionLayer_global_first(config) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([MPPix2StructVisionLayer_local_first(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        question_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    question_hidden_states,
                    attention_mask,
                    question_attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, question_hidden_states, attention_mask, question_attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            question_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        outputs =  BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        outputs['question_hidden_states'] = question_hidden_states
        outputs['question_attention_mask'] = question_attention_mask

        return outputs

class MPPix2StructVisionModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructVisionConfig
    main_input_name = "flattened_patches"
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
        question_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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

        embedding_output, question_hidden_states = self.embeddings(flattened_patches, question_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            question_hidden_states=question_hidden_states,
            attention_mask=attention_mask,
            question_attention_mask=question_attention_mask,
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

        outputs = BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        outputs['question_hidden_states'] = encoder_outputs['question_hidden_states']
        outputs['question_attention_mask'] = encoder_outputs['question_attention_mask']

        return outputs
    

class MPPix2Struct(Pix2StructForConditionalGeneration):
    _tied_weights_keys = ["decoder.lm_head.weight"]
    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)

        self.processor = Pix2StructProcessor.from_pretrained(config._name_or_path)
        self.tokenizer = self.processor.tokenizer

        self.collator = MPPix2StructCollator(self.tokenizer, self.processor, config)

        config.vision_config.vocab_size = config.text_config.vocab_size
        self.encoder = MPPix2StructVisionModel(config.vision_config)

        self.page_prediction = None
        self.box_prediction = None

        self.config = config

        self.max_pages = config.vision_config.max_pages
        self.n_global_tokens = config.vision_config.n_global_tokens
        self.use_all_tokens = config.use_all_tokens
    
    def set_pages(self, max_pages: int):
        self.max_pages = max_pages
        self.collator.max_pages = max_pages
        self.encoder.embeddings.max_pages = max_pages
        for block in self.encoder.encoder.layer:
            block.max_pages = max_pages
    
    def set_max_patches(self, max_patches: int):
        self.collator.max_patches = max_patches

    
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        question_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,
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
                question_ids=question_ids,
                attention_mask=attention_mask,
                question_attention_mask=question_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        question_hidden_states = encoder_outputs['question_hidden_states']
        question_attention_mask = encoder_outputs['question_attention_mask']
        
        if not self.use_all_tokens:
            global_hidden_states = hidden_states[:, -self.n_global_tokens:, :].reshape(-1, self.max_pages*self.n_global_tokens, hidden_states.size(-1))
            global_attention_mask = attention_mask[:, -self.n_global_tokens:].reshape(-1, self.max_pages*self.n_global_tokens)
            hidden_states = global_hidden_states
            attention_mask = global_attention_mask
        else:
            hidden_states = hidden_states.view(-1, self.max_pages*hidden_states.size(-2), hidden_states.size(-1))
            attention_mask = attention_mask.view(-1, self.max_pages*attention_mask.size(-1))

        if question_hidden_states is not None:
            hidden_states = torch.cat([question_hidden_states, hidden_states], dim=1)
            attention_mask = torch.cat([question_attention_mask, attention_mask], dim=-1)

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
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
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
        question_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,
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
            "question_ids": question_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "question_attention_mask": question_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }



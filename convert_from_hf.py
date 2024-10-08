def convert_t5_to_vt5(hf_path, vt5_path):
    pass

def convert_t5_to_mpvt5(hf_path, mpvt5_path):
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
    
    tokenizer = T5Tokenizer.from_pretrained(hf_path, legacy=False)
    t5 = T5ForConditionalGeneration.from_pretrained(hf_path)
    t5_config = t5.config
    t5_weights = t5.state_dict()
    del t5

    t5_config.architectures = ["MPVT5"]
    t5_config.model_type = "mp-vt5"
    t5_config.n_positions = 1024

    t5_config = t5_config.to_dict()

    t5_config["use_all_tokens"] = False
    t5_config["box_prediction"] = False
    t5_config["page_prediction"] = False
    t5_config["pixel_only"] = False
    t5_config["pretraining"] = False
    t5_config["continuous_spatial_embeddings"] = False
    t5_config["feature_extractor_name"] = "google/pix2struct-base"
    t5_config["global_d_ff"] = 1024
    t5_config["global_d_kv"] = 64
    t5_config["global_d_model"] = 256
    t5_config["global_num_heads"] = 4
    t5_config["image_resolution"] = 512
    t5_config["max_patches"] = 196
    t5_config["max_2d_position_embeddings"] = 1024
    t5_config["max_pages"] = 2
    t5_config["n_page_tokens"] = 32
    t5_config["padding"] = "longest"
    t5_config["hidden_dropout_prob"] = 0.1

    t5_config = T5Config.from_dict(t5_config)

    new_weights = t5_weights.copy()
    for key in t5_weights.keys():
        if key.startswith('encoder.block'):
            new_weights[key.replace('layer.', 'page_layer.layer.')] = new_weights.pop(key)
    
    from models.MPVT5.mp_vt5 import MPVT5

    mpvt5 = MPVT5(t5_config)
    mpvt5.load_state_dict(new_weights, strict=False)

    mpvt5.tokenizer.add_special_tokens({'additional_special_tokens': [f'<page_token_{i}>' for i in range(32)]})
    mpvt5.resize_token_embeddings(len(mpvt5.tokenizer))

    mpvt5.tokenizer.model_max_length = 1024
    mpvt5.save_pretrained(mpvt5_path)
    mpvt5.tokenizer.save_pretrained(mpvt5_path)


def convert_pix2struct_to_mppix2struct(hf_path, mppix2struct_path):
    from transformers import Pix2StructForConditionalGeneration, Pix2StructConfig

    pix2struct = Pix2StructForConditionalGeneration.from_pretrained(hf_path)
    pix2struct_config = pix2struct.config
    pix2struct_weights = pix2struct.state_dict()
    del pix2struct

    pix2struct_config.architectures = ["MPPix2Struct"]
    pix2struct_config.model_type = "mp-pix2struct"

    vision_config = pix2struct_config.vision_config
    vision_config = vision_config.to_dict()
    vision_config["global_d_ff"] = 1024
    vision_config["global_d_kv"] = 64
    vision_config["global_d_model"] = 256
    vision_config["global_num_heads"] = 4
    vision_config["image_resolution"] = 4096
    vision_config["max_patches"] = 1024
    vision_config["max_pages"] = 2
    vision_config["n_global_tokens"] = 32
    vision_config["global_first"] = False

    pix2struct_config.vision_config = Pix2StructConfig.from_dict(vision_config)
    pix2struct_config = pix2struct_config.to_dict()
    pix2struct_config["use_all_tokens"] = False
    pix2struct_config["question_in_image"] = True

    pix2struct_config = Pix2StructConfig.from_dict(pix2struct_config)


    new_weights = pix2struct_weights.copy()
    for key in pix2struct_weights.keys():
        if key.startswith("encoder.encoder.layer."):
            num_layer = int(key.split(".")[3])
            
            new_weights[key.replace(f"encoder.encoder.layer.{num_layer}", f"encoder.encoder.layer.{num_layer}.local_layer")] = new_weights.pop(key)

    from models.MPPix2Struct.mp_pix2struct import MPPix2Struct

    model = MPPix2Struct(pix2struct_config)
    model.load_state_dict(new_weights, strict=False)

    model.save_pretrained(mppix2struct_path)
    model.tokenizer.save_pretrained(mppix2struct_path)



convert_pix2struct_to_mppix2struct("google/pix2struct-docvqa-large", "models/MPPix2Struct/mp-pix2struct-large")



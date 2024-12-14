from huggingface_hub import hf_hub_download
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig
)

import torch

def get_attn_only_2L_transformer(from_pretrained: bool = False, device = None):
    cfg = HookedTransformerConfig(
        d_model = 768,
        d_head = 64,
        n_heads = 12,
        n_layers = 2,
        n_ctx = 2048,
        d_vocab = 50278,
        attention_dir = "causal",
        attn_only = True,
        tokenizer_name = "EleutherAI/gpt-neox-20b",
        seed = 398,
        use_attn_result = True,
        normalization_type = None,
        positional_embedding_type = "shortformer"
    )
    
    model = HookedTransformer(cfg)
    
    if from_pretrained:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        repo_id = "callummcdougall/attn_only_2L_half"
        filename = "attn_only_2L_half.pth"
        
        weights_path = hf_hub_download(
            repo_id,
            filename
        )
        
        pretrained_weights = torch.load(weights_path, map_location = device)
        model.load_state_dict(pretrained_weights)
        
    return model

if __name__ == "__main__":
    model = get_attn_only_2L_transformer()
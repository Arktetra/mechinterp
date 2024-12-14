from transformer_lens import HookedTransformer, ActivationCache
from mechinterp.transformer_utils import get_attn_only_2L_transformer

def current_attn_detector(
    model: HookedTransformer, cache: ActivationCache
) -> dict[int, list[int]]:
    """
    Returns a list of heads which are judged to be current-token heads.
    """
    attn_heads = {}
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_pattern = cache["pattern", layer][head]
            score = attn_pattern.diagonal().mean()
            
            if score > 0.4:
                if layer not in attn_heads.keys():
                    attn_heads.update({layer: [head]})
                else:
                    attn_heads[layer].append(head)
    
    return attn_heads

def prev_attn_detector(
    model: HookedTransformer, cache: ActivationCache
) -> dict[int, list[int]]:
    """
    Returns a list of heads which are judged to be previous-token heads.
    """
    attn_heads = {}
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_pattern = cache["pattern", layer][head]
            score = attn_pattern.diagonal(offset = -1).mean()
            
            if score > 0.4:
                if layer not in attn_heads.keys():
                    attn_heads.update({layer: [head]})
                else:
                    attn_heads[layer].append(head)
    
    return attn_heads

def first_attn_detector(
    model: HookedTransformer, cache: ActivationCache
) -> dict[int, list[int]]:
    """
    Returns a list of heads which are judged to be first-token heads.
    """
    attn_heads = {}
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_pattern = cache["pattern", layer][head]
            score = attn_pattern[:, 0].mean()
            
            if score > 0.4:
                if layer not in attn_heads.keys():
                    attn_heads.update({layer: [head]})
                else:
                    attn_heads[layer].append(head)
    
    return attn_heads
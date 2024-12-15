from mechinterp.plotly_utils import imshow
from transformer_lens import HookedTransformer, ActivationCache
from typing import Optional

def plot_attn_pattern(
    cache: ActivationCache, 
    layer: int, 
    head_idx: int,
    tokens: Optional[list[str]] = None,
    return_type: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    attn_pattern = cache["pattern", layer][head_idx]
    
    if tokens is not None:
        x_tokens = [token + f" ({i})" for i, token in enumerate(tokens)]
        y_tokens = x_tokens
    else:
        x_tokens, y_tokens = None, None
    
    fig = imshow(
        attn_pattern,
        x = x_tokens,
        y = y_tokens,
        **kwargs
    )
    
    fig.update_layout(
        font_family = "Times New Roman",
        title_font_family = "Times New Roman",
    )
    
    if return_type is None:
        fig.show()
    else:
        return fig.to_html(full_html = False)
    
def current_attn_detector(
    model: HookedTransformer, cache: ActivationCache
) -> dict[int, list[int]]:
    """
    Returns a dictionary of heads which are judged to be current-token heads.
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
    Returns a dictionary of heads which are judged to be previous-token heads.
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
    Returns a dictionary of heads which are judged to be first-token heads.
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

def induction_attn_detector(
    model: HookedTransformer, cache: ActivationCache
) -> dict[int, list[int]]:
    """
    Returns a dictionary of heads which are judged to be induction heads.
    """
    attn_heads = {}
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_pattern = cache["pattern", layer][head]
            seq_len = (attn_pattern.shape[-1] - 1) // 2
            score = attn_pattern.diagonal(-seq_len + 1).mean()
            
            if score > 0.4:
                if layer not in attn_heads.keys():
                    attn_heads.update({layer: [head]})
                else:
                    attn_heads[layer].append(head)
    
    return attn_heads
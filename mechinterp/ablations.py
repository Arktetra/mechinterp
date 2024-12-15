import torch
import transformer_lens.utils as utils

from functools import partial
from jaxtyping import Float, Int
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import Optional

from mechinterp.utils import cross_entropy_loss
from mechinterp.plotly_utils import imshow

def zero_ablation_hook(
    v: Float[torch.Tensor, "batch seq n_heads d_head"], 
    hook: HookPoint, 
    head_idx_to_ablate: int
) -> Float[torch.Tensor, "batch seq n_heads d_head"]:
    """
    Hook function for performing zero ablation.
    """
    v[:, :, head_idx_to_ablate, :] = 0.0
    return v


def get_ablation_scores_induction(
    model: HookedTransformer, tokens: Int[torch.Tensor, "batch seq"]
) -> Float[torch.Tensor, "n_layers n_heads"]:
    """
    Returns ablation scores for induction.
    """
    ablation_scores: Float[torch.Tensor, "n_layers n_heads"] = torch.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )

    # Calculate a baseline loss
    model.reset_hooks()
    logits: Float[torch.Tensor, "batch seq d_vocab"] = model(tokens, return_type="logits")
    seq_len = (tokens.shape[1] - 1) // 2
    loss_no_ablation: torch.Tensor = cross_entropy_loss(
        logits[:, -seq_len:], tokens[:, -seq_len:]
    )
    
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            temp_hook_fn = partial(zero_ablation_hook, head_idx_to_ablate = head)
            ablated_logits: Float[torch.Tensor, "batch seq d_vocab"] = model.run_with_hooks(
                tokens,
                fwd_hooks = [
                    (utils.get_act_name("v", layer), temp_hook_fn)
                ],
                return_type = "logits"
            )
            loss = cross_entropy_loss(ablated_logits[:, -seq_len:], tokens[:, -seq_len:])
            ablation_scores[layer, head] = loss - loss_no_ablation 
            
        del ablated_logits 
        del loss 
        torch.cuda.empty_cache()
        
    del loss_no_ablation
    del logits 
    torch.cuda.empty_cache()
    
    return ablation_scores

def plot_ablation_scores(
    ablation_scores: Float[torch.Tensor, "n_layers n_heads"],
    return_type: Optional[str] = None
) -> Optional[str]:
    fig = imshow(
        ablation_scores,
        labels = {"x": "Head", "y": "Layer", "color": "Loss diff"},
        title = "Loss Difference After Ablating Heads",
        text_auto = ".2f",
        width = 600, height = 600
    )
    
    if return_type == "html":
        return fig.to_html(full_html = False)
    else:
        fig.show()
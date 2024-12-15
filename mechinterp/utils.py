from transformer_lens import (
    ActivationCache,
    HookedTransformer
)
from jaxtyping import Float, Int

import torch 
import torch.nn.functional as F

def get_log_probs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """
    Returns log of the probabilities for each token in tokens.
    """
    log_probs = logits.log_softmax(dim = -1)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim = -1, index = tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    
    return log_probs_for_tokens 

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> torch.Tensor:
    """
    Returns a batch of repeated of tokens of `2 * seq_len`. The tokens upto
    `seq_len` are repeated after it.
    """
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    tokens = torch.randint(0, model.cfg.d_vocab, (batch, seq_len))
    repeated_tokens = torch.cat([prefix, tokens, tokens], dim = 1)
    return repeated_tokens

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> tuple[torch.Tensor, torch.Tensor, ActivationCache]:
    """
    Generates repeated tokens, then runs and caches the model on it.
    
    Returns a tuple of `repeated_tokens, repeated_logits, repeated_cache`.
    """
    repeated_tokens = generate_repeated_tokens(model, seq_len, batch).to(model.cfg.device)
    repeated_logits, repeated_cache = model.run_with_cache(
        repeated_tokens,
        remove_batch_dim = True
    )
    return repeated_tokens, repeated_logits, repeated_cache

def cross_entropy_loss(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    tokens: Float[torch.Tensor, "batch seq"]
) -> torch.Tensor:
    """
    Returns cross entropy loss for the given logits and tokens.
    """
    log_probs: Float[torch.Tensor, "batch seq d_vocab"] = F.log_softmax(logits, dim = -1)
    pred_log_probs: Float[torch.Tensor, "batch seq"] = torch.gather(
        logits[:, :-1], -1, tokens[:, 1:]
    )[..., 0]
    return -pred_log_probs.mean()

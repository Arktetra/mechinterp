from mechinterp import heads 
from mechinterp.transformer_utils import get_attn_only_2L_transformer
from mechinterp.utils import run_and_cache_model_repeated_tokens

class TestHeadDetectors:
    model = get_attn_only_2L_transformer(from_pretrained = True)
    
    def test_current_attn_detector(self):
        text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
        logits, cache = self.model.run_with_cache(text, remove_batch_dim = True)
        attn_heads = heads.current_attn_detector(self.model, cache)
        assert attn_heads == {0: [9]} 
    
    def test_prev_attn_detector(self):
        text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
        logits, cache = self.model.run_with_cache(text, remove_batch_dim = True)
        attn_heads = heads.prev_attn_detector(self.model, cache)
        assert attn_heads == {0: [7]}
    
    def test_first_attn_detector(self):
        text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
        logits, cache = self.model.run_with_cache(text, remove_batch_dim = True)
        attn_heads = heads.first_attn_detector(self.model, cache)
        assert attn_heads == {0: [3], 1: [4, 10]}
        
    def test_induction_attn_detector(self):
        repeated_tokens, repeated_logits, repeated_cache = run_and_cache_model_repeated_tokens(
            self.model, seq_len = 50
        )
        attn_heads = heads.induction_attn_detector(self.model, repeated_cache)
        assert attn_heads == {1: [4, 10]}
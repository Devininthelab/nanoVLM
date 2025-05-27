import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))  # 756; gamma in RMSNorm
        self.eps = cfg.lm_rms_eps

    def forward(self, x):
        # bs, seq_len, dim 
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head 576 // 9 = 64
        self.base = cfg.lm_re_base # lm_re_base=100000
        self.max_seq_len = cfg.lm_max_position_embeddings # 8192
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings # 8192
        self.attention_scaling = cfg.lm_attn_scaling # 1.0

    @torch.no_grad()
    def forward(self, position_ids):
        batch_size, seq_len = position_ids.shape
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        # Dynamic scaling for longer sequences
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq
        # Got 10000 * (2i/dim)
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float() # [batch_size * seq_len]
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0) # m * theta; same theta for two dimensions, flat_position_ids at the input is m: the position of the token in the sentence
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        # bs, seq_len, dim//2 ex: 2, 10, 32
        
        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1) # bs, seq_len, dim
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin

# Rotates half the hidden dims of the input by swapping and negating dimensions.
def rotate_half(x):
    # Given x = [x0, x1, x2, x3], this will return [-x1, x0, -x3, x2]
    x1, x2 = x.chunk(2, dim=-1) # [bs, seq_len, dim//2], [bs, seq_len, dim//2]
    return torch.cat((-x2, x1), dim=-1) # negate x2 and concatenate with x1

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q, k, cos, sin, unsqueeze_dim=1):
    """ given x = [a, b]
    Rope(x) = 
            |cos(theta) -sin(theta) |   |a|
            |sin(theta)   cos(theta)| * |b|
        =     |a| * cos(theta) - |b| * sin(theta)
              |b|                |a|
    where theta = position * frequency

    that is why we got rotate half  
    """
    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads # 9
        self.n_kv_heads = cfg.lm_n_kv_heads # 3
        self.embd_dim = cfg.lm_hidden_dim # 576
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        # Number of queries = number of heads
        # Number of kv = number of kv_groups
        # In this case, we have 9 heads ( 9 queries), and 3 kv groups (3 keys and 3 values)
        # Meaning that for each kv group, we will perform 3 queries on it
        self.n_kv_groups = self.n_heads // self.n_kv_heads # 3
        self.head_dim = self.embd_dim // self.n_heads # 576//9 = 64

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False) # Linear from (embed_dim, to 9 heads * 64 head dim = embed_dim)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False) # Linear from embed_dim, to 3 kv heads * 64 dim = 192
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False) # Linear from embed_dim, to 3 kv heads * 64 dim = 192
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim) -> B, n_heads, T, head_dim
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim) -> B, n_kv_heads, T, head_dim
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim) -> B, n_kv_heads, T, head_dim
        
        # Use precomputed positional embeddings
        q, k = apply_rotary_pos_embd(q, k, cos, sin)

        k = k.repeat_interleave(self.n_kv_groups, dim=1) # repeat k for each kv group
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Process attention mask if provided
        if attention_mask is not None:
            # Create a 4D attention mask [batch_size, 1, 1, seq_length], In this format, 1 = attend, 0 = mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            padding_mask = (attention_mask == 0).transpose(-1, -2) # Use this for the manual path
            # Convert to attention mask where 0 keeps values and -inf masks
            attention_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min 
            # [[1, 1, 0, 0], [1, 1, 1, 0]] -> [[0, 0, -inf, -inf], [0, 0, 0, -inf]]

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # LM attention is causal (masked)
            )
        else:
            attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))
            if attention_mask is not None:
                attn = attn + attention_mask 

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v
            
            if attention_mask is not None:
                y = y.masked_fill(padding_mask, 0.0) # Zero out the padded positions in the output
            """
            I, love, you, <pad>
            attn_scores_raw =
                [[1, 2, 3, 4],   # token "I"
                [5, 6, 7, 8],    # token "love"
                [9, 10, 11, 12], # token "you"
                [13,14,15,16]]   # token "<pad>"
            causal_mask =
                [[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]]

            attn_scores_after_causal =
                [[  1, -inf, -inf, -inf],
                [  5,    6, -inf, -inf],
                [  9,   10,   11, -inf],
                [ 13,   14,   15,   16]]

            attention_mask = [1, 1, 1, 0] -> last token is padding
            attention_mask_float = (1 - attention_mask) * -inf
                    = [0, 0, 0, -inf]

            attn_scores = attn_scores + attention_mask_float
            attn_scores_after_both =
                [[  1, -inf, -inf, -inf],        # line 1 + [0, 0, 0, -inf]
                [  5,    6, -inf, -inf],
                [  9,   10,   11, -inf],
                [ 13,   14,   15, -inf]]         # line 4 + [0, 0, 0, -inf]

            Then take softmax
            attn_scores_after_softmax =
                [[1, 0, 0, 0],   # token "I"
                [0.2, 0.8, 0, 0],    # token "love"
                [0.1, 0.2, 0.7, 0],    # token "you"
                [0.3, 0.1, 0.6, 0]]    # token "<pad>"
            """
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim # 576
        self.inter_dim = cfg.lm_inter_dim # 1536

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        # x is [batch_size, seq_len, embd_dim]
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x, cos, sin, attention_mask=None):
        res = x # Residual connection
        x = self.norm1(x)
        x = self.attn(x, cos, sin, attention_mask) # bs, seq_len, dim
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens # False # Whether to use token embeddings or not
        self.lm_tie_weights = cfg.lm_tie_weights # True # Whether to tie the token embedding and output head weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim) # lm_vocab_size=49152, lm_hidden_dim=576
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks) # lm_n_blocks=30
        ])
        self.norm = RMSNorm(cfg) # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x, attention_mask=None):
        if self.lm_use_tokens: # If we are using tokens, we expect x to be token indices; or else use hf tokenzizers + embeddings
            x = self.token_embedding(x) # Only embed the inputs when using tokens
        
        B , T, _ = x.size()
        
        # Note: You could also cache these input embeddings if you want to avoid recomputing them
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1) # Create position ids [0, 1, 2, ..., seq_len-1]
        cos, sin = self.rotary_embd(position_ids) # Get rotary position embeddings

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
        x = self.norm(x)

        if self.lm_use_tokens:
            x = self.head(x) # Compute logits if we are using tokens, otherwise stay in the embedding space

        return x

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=20):
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            
        generated = inputs.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass through the model
            outputs = self.forward(generated)
            last_output = outputs[:, -1, :]

            if self.lm_use_tokens:
                # Now the model outputs logits
                next_token = torch.argmax(last_output, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
            else:
                # Now the model outputs embeddings
                next_token_embedding = last_output.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
                generated = torch.cat((generated, next_token_embedding), dim=1)
            
            #Note: You could enable the generation to break earlier than max_new_tokens when it detects a eos token, but this does not work in batched generation (output tensors need to have the same size)
    
        return generated

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")
        
        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    
                    # Special handling for token embeddings if vocab sizes differ
                    if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                        has_extended_embeddings = True
                        print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                        
                        # Copy existing embeddings to the beginning of our larger embedding matrix
                        sd[our_key][:tensor.shape[0]].copy_(tensor)
                        
                        # Initialize the new embeddings using the same approach as the original model
                        std = 0.02  # Common value, but you might want to adjust based on model
                        init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                        
                        print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                        sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                    elif tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                if 'lm_head.weight' in f.keys():
                    lm_head = f.get_tensor('lm_head.weight')
                    if lm_head.shape[0] != sd['head.weight'].shape[0]:
                        print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                        # Copy existing weights
                        sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                        # Initialize new weights
                        std = 0.02
                        init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                        # Load updated weights
                        model.load_state_dict(sd)
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model

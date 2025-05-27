# Modality Projection from Vision to Language
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2) # 768 * (2**2) = 3072
        self.output_dim = cfg.lm_hidden_dim # 576
        self.scale_factor = cfg.mp_pixel_shuffle_factor # 2

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size() # of output from ViT: bs, num_patches = 196, embed_dim 
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root # = 14
        x = x.view(bsz, height, width, embed_dim) # Reshape to (bsz, height, width, embed_dim)
        h_out = height // self.scale_factor # 14 // 2 = 7
        w_out = width // self.scale_factor # 14 // 2 = 7
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim) # (bsz, 7, 2, 7, 2, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (bsz, 7, 7, 2, 2, embed_dim)
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2) # (bsz, 49, 3072)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x

    
from models.utils import check_multiple_choice_with_regex, top_k_top_p_filtering
import torch
import torch.nn.functional as softmax
import torch.nn as nn
from transformers import SiglipVisionConfig
from huggingface_hub import hf_hub_download
import safetensors
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.config import VLMConfig
from models.language_model import RotaryEmbedding, RMSNorm
import math

rms = RMSNorm(VLMConfig)
tensors = torch.ones((1, 5, 576)) * 5 # bs, seq_len, dim

print(rms(tensors))

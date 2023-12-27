import torch.nn as nn
import torch
from ldm.modules.diffusionmodules.util import timestep_embedding
import torch.nn.functional as F
from ldm.modules.attention import CrossAttention
    
class TTPM(nn.Module):
    def __init__(self,
                 model_channels= 320,
                 dropout=0.0,
                 time_embed_dim=1280,
                 prompt_channels=77,
                 prompt_dim = 1024,
                 hidden_size = 512,
                 num_heads = 8,
                 device = 'cuda',
                ):
        
        super().__init__()
        self.model_channels = model_channels
        self.dropout = nn.Dropout(dropout)
        self.time_embed_dim = time_embed_dim
        self.prompt_channels = prompt_channels
        self.prompt_dim = prompt_dim

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size),
        )

        self.learnable_param = nn.Parameter(torch.rand(prompt_channels,prompt_dim))
        self.transform_in = nn.Sequential(nn.Linear(prompt_dim, hidden_size),
                                           nn.Tanh(), 
                                           nn.Linear(hidden_size, hidden_size)) 
        
        num_head_channels = hidden_size // num_heads

        self.atten = CrossAttention(hidden_size,hidden_size,num_heads,num_head_channels)

        self.transform_out = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                           nn.Tanh(), 
                                           nn.Linear(hidden_size, prompt_dim))

        
    def forward(self, t):
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb = self.dropout(F.normalize(emb))
        emb = emb.unsqueeze(1).expand(-1, self.prompt_channels, -1)

        prompt = self.learnable_param 
        prompt = self.transform_in(prompt)
        prompt = F.normalize(prompt)
        batchsize = emb.shape[0]
        prompt = prompt.unsqueeze(0).repeat(batchsize,1,1)

        prompt = self.atten(prompt,emb) + prompt

        prompt = self.transform_out(prompt)
        prompt = F.normalize(prompt)
        
        return prompt   
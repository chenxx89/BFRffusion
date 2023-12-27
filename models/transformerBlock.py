## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, Upsample


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def reshape(x, emb_out):
    emb_list = []
    for emb in emb_out:
        while len(emb.shape) < len(x.shape):
            emb = emb[..., None]
        emb_list.append(emb)
    return emb_list

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*3, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.pwconw = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2, x3 = self.project_in(x).chunk(3, dim=1)
        x2, x3 = self.dwconv(torch.cat((x2,x3),dim=1)).chunk(2, dim=1)
        x = F.gelu(x1)*x3 + F.gelu(x2)*x3 + self.pwconw(x3)
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 6 * dim, bias=True)
        )

    def forward(self, x, emb):

        emb_out = self.adaLN_modulation(emb).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = reshape(x, emb_out)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


##########################################################################
class conv3x3(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(conv3x3, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class MFEM(nn.Module):
    def __init__(self, 
        in_channels=4, 
        control_channels = 320,
        time_embed_dim = 1280,
        heads = [1,2,4,8],
        conv_resample=True,
        dims=2,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
    ):

        super().__init__()
        self.control_channels = control_channels
        self.dims = dims
        self.time_embed = nn.Sequential(
            linear(control_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, control_channels, 3, padding=1))
        )
        self.input_blocks = TimestepEmbedSequential(conv_nd(dims, in_channels, control_channels, 3, padding=1))


        # self.conv3x3 = nn.Conv2d(in_channels, control_channels,3, padding=1)      # 4,64,64 ->320,64,64
        self.resblock = ResBlock(control_channels, time_embed_dim, dropout=0, out_channels=control_channels)
        
        self.encoder1 = TransformerBlock(dim=control_channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down1 = Downsample(control_channels, conv_resample, dims, control_channels*2) ## From 320,64,64 to 640,32,32
        
        self.encoder2 = TransformerBlock(dim=int(control_channels*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down2 = Downsample(control_channels*2, conv_resample, dims, control_channels*4) ## From 640,32,32 -> 1280,16,16
        
        self.encoder3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down3 = Downsample(control_channels*4, conv_resample, dims, control_channels*4) ## From 1280,16,16 -> 1280,8,8
            
        self.mid1 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.mid2 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.mid3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 

        self.up3 = Upsample(control_channels*4, conv_resample, dims, control_channels*4) ## From 1280,8,8 -> 1280,16,16
        self.reduce_chan_level3 = nn.Conv2d(int(control_channels*8), int(control_channels*4), kernel_size=1, bias=bias)
        self.decoder3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)

        self.up2 = Upsample(control_channels*4, conv_resample, dims, control_channels*2) ## From 1280,16,16 -> 640,32,32
        self.reduce_chan_level2 = nn.Conv2d(int(control_channels*4), int(control_channels*2), kernel_size=1, bias=bias)
        self.decoder2 = TransformerBlock(dim=int(control_channels*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)
        
        self.up1 = Upsample(control_channels*2, conv_resample, dims, control_channels)  ## From 640,32,32 -> 320,64,64
        self.reduce_chan_level1 = nn.Conv2d(int(control_channels*2), int(control_channels), kernel_size=1, bias=bias)
        self.decoder1 = TransformerBlock(dim=int(control_channels), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)
        
        self.zero_convs_module = nn.ModuleList([TimestepEmbedSequential(conv_nd(self.dims, control_channels, control_channels, 1, padding=0)), 
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*2, control_channels*2, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*2, control_channels*2, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels, control_channels, 1, padding=0))])

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.control_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hint = self.input_hint_block(hint, emb, context)
        x = self.input_blocks(x, emb, context) + hint
 

        # x = self.conv3x3(x)
        x = self.resblock(x, emb)
        en1 = self.encoder1(x, emb)    
        dn1 = self.down1(en1)

        en2 = self.encoder2(dn1, emb)
        dn2 = self.down2(en2)

        en3 = self.encoder3(dn2, emb) 
        dn3 = self.down3(en3)
        
        mid1 = self.mid1(dn3, emb)
        mid2 = self.mid2(mid1, emb)
        mid3 = self.mid3(mid2, emb)

        up3 = self.up3(mid3)
        up3 = torch.cat([up3, en3], 1)
        up3 = self.reduce_chan_level3(up3)
        de3 = self.decoder3(up3, emb) 

        up2 = self.up2(de3)
        up2 = torch.cat([up2, en2], 1)
        up2 = self.reduce_chan_level2(up2)
        de2 = self.decoder2(up2, emb) 

        up1 = self.up1(de2)
        up1 = torch.cat([up1, en1], 1)
        up1 = self.reduce_chan_level1(up1)
        de1 = self.decoder1(up1, emb)

        out = [en1,en2,en3,mid1,mid2,mid3,de3,de2,de1]

        assert len(out) == len(self.zero_convs_module)

        for i,module in enumerate(self.zero_convs_module):
            out[i] = module(out[i],emb, context)
        return out
    








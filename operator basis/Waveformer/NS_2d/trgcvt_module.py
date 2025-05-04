
#%%
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import module
from module import ConvAttention1, ConvAttention2, PreNorm, PreNorm2, FeedForward
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler as lr_scheduler
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# device= torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
#%%

class Transformer1(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention1(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
     

    def forward(self, x):
         for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
         return x



class Transformer2(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.):
        
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ConvAttention2(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout),
                                              FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    # def forward(self, x,y):
    #      for attn, ff in self.layers:
    #          y = attn(x,y) + y
    #          y = ff(y) + y
    #      return y
        
        # self.layers = nn.ModuleList([ConvAttention2(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout)])
        
        
    def forward(self, x, y):
        for attn, ff in self.layers:
            self.norm = nn.LayerNorm(self.dim).to(device)
            y = self.norm(attn(x,y) + y)
            y = self.norm(ff(y) + y)
        return y




class CvTencoderdecoder1(nn.Module):
    def __init__(self, image_size, in_channels1, in_channels2,out_dim, dim = 28, kernels=[6, 3, 2], strides=[2, 2, 2],
                 heads=[2, 4, 4] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()




        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.image_size = image_size
        
        ##### Stage 1 #######
        # self.stage1enc_conv_embed = nn.Sequential(
        #     nn.Conv2d(in_channels1, dim, kernels[0], strides[0], 2),
        #     Rearrange('b c h w -> b (h w) c', h = (self.image_size), w = (self.image_size)),
        #     nn.LayerNorm(dim)
        # )
        
        # self.stage1dec_conv_embed = nn.Sequential(
        #     nn.Conv2d(in_channels2, dim, kernels[0], strides[0], 2),
        #     Rearrange('b c h w -> b (h w) c', h = (self.image_size), w = (self.image_size)),
        #     nn.LayerNorm(dim)
        # )
        
        self.stage1enc_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(self.image_size),depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
            # Rearrange('b (h w) c -> b c h w', h = (self.image_size)//2, w = (self.image_size)//2)
        )
        
        self.stage1dec_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(self.image_size),depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
            # Rearrange('b (h w) c -> b c h w', h = (self.image_size)//2, w = (self.image_size)//2)
        )
        ##### Stage 2 #######
        # in_channels = dim
        # scale = heads[1]//heads[0]
        # dim = scale*dim
        # self.stage2enc_conv_embed = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
        #     Rearrange('b c h w -> b (h w) c', h = (self.image_size+2)//4, w = (image_size+2)//4),
        #     nn.LayerNorm(dim)
        # )
        
        # self.stage2dec_conv_embed = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
        #     Rearrange('b c h w -> b (h w) c', h = (image_size+2)//4, w = (image_size+2)//4),
        #     nn.LayerNorm(dim)
        # )
        
        # self.stage2enc_transformer = nn.Sequential(
        #     Transformer1(dim=dim, img_size=(image_size+2)//4, depth=depth[1], heads=heads[1], dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout)
        # )
        
        # self.stage2dec_transformer = nn.Sequential(
        #     Transformer1(dim=dim, img_size=(image_size+2)//4, depth=depth[1], heads=heads[1], dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout)
        # )
        
        
        
        
        
        ##### Stage 3 #######
        in_channels = self.dim
        # scale = heads[1]//heads[0]
        # dim = scale*dim
        # dim = 32
        self.stage3_transformer = Transformer2(dim=self.dim, img_size=(image_size), depth=depth[1], heads=4, dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
        
        # self.stage4_transformer = Transformer2(dim=self.dim, img_size=(image_size), depth=depth[1], heads=4, dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout)
        
        
        
        # nn.Sequential(
        #     Transformer2(dim=dim, img_size=image_size//8, depth=depth[1], heads=4, dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout),
        #     Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
        # )

        # self.stage4_conv_embed = nn.ConvTranspose2d(dim, 32, kernels[1], strides[1], padding=1)
        
        # self.stage5_conv_embed = nn.ConvTranspose2d(dim, out_dim, kernels[0], strides[0], padding=2)



    def forward(self, img1, img2):
        
        img1 = rearrange(img1, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)
        img2 = rearrange(img2, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)

        # xs = self.stage1enc_conv_embed(img1)
        xs1 = self.stage1enc_transformer(img1)
        # xs = self.stage2enc_conv_embed(xs)
        # xs1= self.stage2enc_transformer(xs)
        # xs = self.stage1dec_conv_embed(img2)
        xs2 = self.stage1dec_transformer(img2)
        # xs = self.stage2dec_conv_embed(xs)
        # xs2 = self.stage2dec_transformer(xs)
        xs3 = self.stage3_transformer(xs1,xs2)
        # xs3 = self.stage4_transformer(xs1,xs3)
        # xs3 = rearrange(xs3,'b (h w) c -> b c h w', h = (self.image_size)//2, w = (self.image_size)//2)
        xsr = rearrange(xs3,'b (h w) c -> b c h w', h = (self.image_size), w = (self.image_size))
        # xsr = self.stage4_conv_embed(xs3)
        # xsr = self.stage5_conv_embed(xsr)
       
        return xsr



class CvTencoderdecoder2(nn.Module):
    def __init__(self, image_size, in_channels1, in_channels2,out_dim, dim = 28, kernels=[6, 4, 2], strides=[2, 2, 2],
                 heads=[2, 4, 4] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()




        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.image_size = image_size
        ##### Stage 1 #######
        self.stage1enc_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels1, dim, kernels[0], strides[0], 2),
            Rearrange('b c h w -> b (h w) c', h = (self.image_size)//2, w = (self.image_size)//2),
            nn.LayerNorm(dim)
        )
        
        self.stage1dec_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels2, dim, kernels[0], strides[0], 2),
            Rearrange('b c h w -> b (h w) c', h = (self.image_size)//2, w = (self.image_size)//2),
            nn.LayerNorm(dim)
        )
        
        self.stage1enc_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(self.image_size)//2,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = (self.image_size)//2, w = (self.image_size)//2)
        )
        
        self.stage1dec_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(self.image_size)//2,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = (self.image_size)//2, w = (self.image_size)//2)
        )
        ##### Stage 2 #######
        in_channels = self.dim
        scale = heads[1]//heads[0]
        dim = scale*dim
       
        self.stage2enc_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h = (self.image_size)//4, w = (image_size)//4),
            nn.LayerNorm(dim)
        )
        
        self.stage2dec_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h = (image_size)//4, w = (image_size)//4),
            nn.LayerNorm(dim)
        )
        
        self.stage2enc_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(image_size)//4, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
        )
        
        self.stage2dec_transformer = nn.Sequential(
            Transformer1(dim=dim, img_size=(image_size)//4, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
        )
        
        
        
        
        
        ##### Stage 3 #######
        in_channels = dim
        # scale = heads[1]//heads[0]
        # dim = scale*dim
        # dim = 36
        self.stage3_transformer = Transformer2(dim= dim, img_size=(image_size)//4, depth=depth[1], heads=4, dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout)
        
        # self.stage4_transformer = Transformer2(dim= dim, img_size=(image_size)//4, depth=depth[1], heads=4, dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout)
        
        
        
        # nn.Sequential(
        #     Transformer2(dim=dim, img_size=image_size//8, depth=depth[1], heads=4, dim_head=self.dim,
        #                                       mlp_dim=dim * scale_dim, dropout=dropout),
        #     Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
        # )

        self.stage4_conv_embed = nn.ConvTranspose2d(dim, 32, kernels[1], strides[1], padding=1)
        
        self.stage5_conv_embed = nn.ConvTranspose2d(32, out_dim, kernels[0], strides[0], padding=2)



    def forward(self, img1, img2):
        
        # img1 = rearrange(img1, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)
        # img2 = rearrange(img2, 'b c h w -> b (h w) c', h=self.image_size, w=self.image_size)

        xs = self.stage1enc_conv_embed(img1)
        xs = self.stage1enc_transformer(xs)
        xs = self.stage2enc_conv_embed(xs)
        xs1 = self.stage2enc_transformer(xs)
        xs = self.stage1dec_conv_embed(img2)
        xs = self.stage1dec_transformer(xs)
        xs = self.stage2dec_conv_embed(xs)
        xs2 = self.stage2dec_transformer(xs)
        xs3 = self.stage3_transformer(xs1,xs2)
        # xs3 = self.stage4_transformer(xs1,xs3)
        # xs3 = rearrange(xs3,'b (h w) c -> b c h w', h = (self.image_size)//4, w = (self.image_size)//4)
        xsr = rearrange(xs3,'b (h w) c -> b c h w', h = (self.image_size)//4, w = (self.image_size)//4)
        xsr = self.stage4_conv_embed(xsr)
        xsr = self.stage5_conv_embed(xsr)
       
        return xsr
"""Full official pretrained Utonia plus input affine and state regression readout."""
from __future__ import annotations

import torch
from torch import nn
import torch_scatter
import utonia


def disable_stochastic_regularizers(model):
    """Existing official runtime switches only; no architecture/parameter changes."""
    for module in model.modules():
        if isinstance(module,nn.Dropout) or type(module).__name__=='DropPath':
            module.eval()


class SensorAffine(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.extra = nn.Linear(11, original.out_features, bias=False)
        nn.init.zeros_(self.extra.weight)

    def forward(self, x):
        return self.original(x[:, :9]) + self.extra(x[:, 9:])


class ObjectPredictor(nn.Module):
    def __init__(self, checkpoint, history=8, deterministic_pooling=False):
        super().__init__()
        config = utonia.load(str(checkpoint), ckpt_only=True)['config']
        if config['in_channels'] != 9:
            raise ValueError(f"Expected released XYZ/RGB/normal input, got {config['in_channels']}")
        self.history=history
        self.deterministic_pooling=deterministic_pooling
        # Official dense attention fallback; retain all checkpoint layers/widths.
        self.backbone=utonia.load(str(checkpoint), custom_config={'enable_flash':False,
                                                                'shuffle_orders':False,
                                                                'freeze_encoder':False})
        # The official constructor does not forward shuffle_orders to its four
        # GridPooling children. Set the same public runtime option there too;
        # otherwise eval() still randomly changes serialization between calls.
        for module in self.backbone.modules():
            if hasattr(module, 'shuffle_orders'):
                module.shuffle_orders=False
        self.original_parameter_count=sum(p.numel() for p in self.backbone.parameters())
        self.backbone.embedding.stem.linear=SensorAffine(self.backbone.embedding.stem.linear)
        self.feature_dim=2*sum(config['enc_channels'])
        self.head=nn.Linear(self.feature_dim*history,13)
        nn.init.normal_(self.head.weight,std=1e-4)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[3]=1
        self.head.bias.data[7]=1

    def forward(self, inputs):
        point=self.backbone({k:inputs[k] for k in ('coord','grid_coord','feat','offset')})
        summaries=[]
        n_frames=len(inputs['offset'])
        while True:
            if self.deterministic_pooling:
                order=torch.argsort(point.batch,stable=True)
                counts=torch.bincount(point.batch,minlength=n_frames)
                pointers=torch.cat([counts.new_zeros(1),counts.cumsum(0)])
                values=point.feat[order]
                mean=torch_scatter.segment_csr(values,pointers,reduce='mean')
                maximum=torch_scatter.segment_csr(values,pointers,reduce='max')
            else:
                mean=torch_scatter.scatter_mean(point.feat,point.batch,dim=0,dim_size=n_frames)
                maximum=torch_scatter.scatter_max(point.feat,point.batch,dim=0,dim_size=n_frames)[0]
            summaries.append(torch.cat((mean,maximum),dim=-1))
            if 'pooling_parent' not in point:
                break
            point=point.pooling_parent
        frame_features=torch.cat(summaries,dim=-1)
        if frame_features.shape[-1] != self.feature_dim:
            raise RuntimeError('Full multiscale encoder features missing')
        return self.head(frame_features.reshape(-1,self.feature_dim*self.history))


def rotation_matrix(rot6):
    a,b=rot6[...,:3],rot6[...,3:]
    x=torch.nn.functional.normalize(a,dim=-1)
    y=torch.nn.functional.normalize(b-(x*b).sum(-1,keepdim=True)*x,dim=-1)
    z=torch.linalg.cross(x,y,dim=-1)
    return torch.stack((x,y,z),dim=-1)


def loss_components(pred,target):
    position=((pred[:,:3]-target[:,:3])/.1).square().mean()
    rotation=(rotation_matrix(pred[:,3:9])-rotation_matrix(target[:,3:9])).square().mean()
    size=((pred[:,9:12]-target[:,9:12])/.3).square().mean()
    mass=((pred[:,12]-target[:,12])/.7).square().mean()
    return dict(position=position,rotation=rotation,size=size,mass=mass)


def metrics(pred,target):
    rel=rotation_matrix(pred[:,3:9]).transpose(-1,-2)@rotation_matrix(target[:,3:9])
    cosine=((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    return dict(position_cm=(pred[:,:3]-target[:,:3]).norm(dim=-1)*100,
                rotation_deg=cosine.acos()*180/torch.pi,
                size_relative=(pred[:,9:12].exp()/target[:,9:12].exp()-1).abs().mean(-1),
                mass_relative=(pred[:,12].exp()/target[:,12].exp()-1).abs())

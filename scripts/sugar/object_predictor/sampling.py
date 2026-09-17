"""Balanced acquisition batches, without changing observations or target losses."""
import math
import torch
from torch.utils.data import Sampler


class BalancedAcquisitionBatchSampler(Sampler):
    def __init__(self,dataset,batch_size,seed):
        if batch_size<2 or batch_size%2:
            raise ValueError('Balanced acquisition requires an even batch size')
        self.batch_size=batch_size
        self.generator=torch.Generator().manual_seed(seed)
        self.groups={name:[] for name in ('probe','support')}
        for index,(episode,_) in enumerate(dataset.index):
            name=dataset.episodes[episode][0].get('acquisition_group')
            if name not in self.groups: raise ValueError(f'Unknown acquisition group {name}')
            self.groups[name].append(index)
        if not all(self.groups.values()): raise ValueError('Both acquisition groups must be present')
        self.groups={k:torch.tensor(v,dtype=torch.int64) for k,v in self.groups.items()}
        self.batches=math.ceil(len(dataset)/batch_size)

    def __len__(self): return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            chosen=torch.cat([indices[torch.randint(len(indices),(self.batch_size//2,),generator=self.generator)]
                              for indices in self.groups.values()])
            yield chosen[torch.randperm(self.batch_size,generator=self.generator)].tolist()

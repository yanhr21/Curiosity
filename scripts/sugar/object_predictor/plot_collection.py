"""Physical observation audit for a compact predictor episode."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(path):
    path=Path(path)
    with np.load(path) as src:
        a={k:src[k] for k in src.files}
    t=a['timestamp_s']
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j,label in enumerate('XYZ'):
        ax[0].plot(t,a['object_pose_w'][:,j],label='object '+label)
    ax[0].set_ylabel('World position [m]');ax[0].legend(ncol=3)
    force=a['normal_load_n']
    for side,label in enumerate(['left','right']):
        ax[1].plot(t,force[:,side*27:(side+1)*27].sum(1),label=label)
    ax[1].set_ylabel('Assigned normal load [N]');ax[1].legend()
    image=ax[2].imshow(np.log1p(force.T),aspect='auto',origin='lower',extent=[t[0],t[-1],0,54])
    ax[2].set_ylabel('Anatomical patch');ax[2].set_xlabel('Time [s]')
    fig.colorbar(image,ax=ax[2],label='log(1 + load [N])')
    fig.suptitle(path.parent.name+' | actual physical observations')
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=140);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('path');main(ap.parse_args().path)

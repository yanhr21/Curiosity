"""Resumable HTTP range download of the official Utonia checkpoint."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import re
import time
import subprocess

URL='https://huggingface.co/Pointcept/Utonia/resolve/main/utonia.pth?download=true'


def main(output):
    dest=Path(output)
    chunks=dest.parent/'utonia_download_chunks'
    chunks.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['curl','-sS','-L','--fail','--retry','4','--retry-all-errors','--max-time','60',
                           '-r','0-0','-D','-','-o',str(chunks/'probe'),URL+'&range_probe=0'],
                          capture_output=True,check=True)
    match=re.search(rb'content-range: bytes 0-0/(\d+)',result.stdout,re.I)
    if not match:
        raise RuntimeError('Server did not honor initial byte range')
    size=int(match[1])
    block=2*1024*1024
    def fetch(i):
        start=i*block
        end=min(start+block,size)-1
        path=chunks/f'{i:05d}'
        if path.exists() and path.stat().st_size==end-start+1:
            return
        for attempt in range(4):
            try:
                temporary=path.with_suffix('.partial')
                result=subprocess.run(['curl','-sS','-L','--fail','--max-time','60',
                                       '-r',f'{start}-{end}','-D','-','-o',str(temporary),URL+f'&range_start={start}'],
                                      capture_output=True,check=True)
                expected=f'content-range: bytes {start}-{end}/{size}'.encode()
                if expected not in result.stdout.lower():
                    raise RuntimeError('Server returned a different byte range')
                if temporary.stat().st_size!=end-start+1:
                    raise RuntimeError('Incomplete range')
                temporary.rename(path)
                if i%16==0:
                    print(f'completed chunk {i}, total {(size+block-1)//block}',flush=True)
                return
            except Exception:
                if attempt==3:
                    raise
                time.sleep(1+attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,range((size+block-1)//block)))
    temporary=dest.with_suffix('.assembling')
    with temporary.open('wb') as stream:
        for i in range((size+block-1)//block):
            stream.write((chunks/f'{i:05d}').read_bytes())
    assert temporary.stat().st_size==size
    temporary.rename(dest)
    print(f'complete official checkpoint: {size} bytes at {dest}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    main(ap.parse_args().output)

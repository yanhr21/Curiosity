"""Assemble all preregistered grip cases, retaining failed physical cases."""
import hashlib,json,shutil
from pathlib import Path


def main():
    root=Path('experiments/object_predictor_v1/grip_transfer_v1')
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    out=root/'data_v1';out.mkdir(exist_ok=False)
    records=[]
    for case in protocol['configurations']:
        source=root/case['directory'];report=json.loads((source/'RESULT.json').read_text())
        config=json.loads((source/'PROTOCOL.json').read_text())
        assert config['mass_kg']==case['mass'] and config['target_load_n']==case['grip_target_per_hand_n']
        for name,digest in protocol['collection_source_sha256'].items():
            if name in config['source_sha256']:assert config['source_sha256'][name]==digest
        assert report['frames']==1200
        src=source/'episode_4000.npz';dst=out/f"episode_{case['episode']:04d}.npz"
        shutil.copyfile(src,dst)
        digest=hashlib.file_digest(src.open('rb'),'sha256').hexdigest()
        assert digest==hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
        report.update(episode=case['episode'],source_episode=4000,source=str(src),source_sha256=digest,
                      sampling_stride=5,primary_sampling_stride=25,grip_target_per_hand_n=case['grip_target_per_hand_n'])
        dst.with_suffix('.json').write_text(json.dumps(report,indent=2));records.append(report)
    result=dict(configurations=records,all_cases_included=True,all_physical_qualification_passed=all(r['passed'] for r in records),
                evaluation='10Hz visual readbacks at31+5k; primary metrics retain original31+25k clocks20..24s; no prediction selection',
                source_protocol_sha256=hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest())
    (out/'PROTOCOL.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Publish an immutable split-preserving view of completed real acquisitions."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(probe, support,normal_policy='stored'):
    protocol=dict(
        sources={'probe':str(Path(probe).resolve()),'support':str(Path(support).resolve())},
        expected_episodes_per_source=36, expected_split_per_source={'train':24,'val':6,'test':6},
        sampling_stride={'probe':50,'support':5},
        batch={'probe':2,'support':2},history=32,
        history_policy='episode_uniform_recent',time_scale_s=150.,
        modes=['geometry','geometry_contact','geometry_contact_force'],
        steps_per_arm=2000,total_new_optimizer_updates=6000,evaluate_every=500,
        training_seed=310016,train_sampler='balanced_acquisitions',
        backbone_lr=1e-5,head_lr=1e-5,sensor_affine_lr=5e-4,
        deterministic_pooling=True,disable_stochastic_regularizers=True,
        loss='Unchanged full position + rotation + log-size + log-mass objective; no target masks.',
        primary_geometry={'acquisition':'probe','time_window_s':[140.,148.],
                          'position_cm_max':5.,'rotation_deg_max':15.,'size_relative_max':.1},
        primary_mass={'acquisition':'support','mass_relative_max':.1},
        required_reporting=['all windows','equal episode means','every original TEST configuration including physical failures',
                            'force zero preserving contact geometry and area','fresh 12-mass transfer separately'],
        endpoint='Final update only; no best checkpoint or test-derived calibration',
        scope='One scanned asset family. Ground-supported probe mass is diagnostic; support fixture supplies weight information. No unseen shape or policy benefit claim.',
        preparation_only=True,training_admitted=False)
    if normal_policy=='hand_surface':
        from .hand_surface_normals import geometry_signature
        protocol.update(normal_policy='hand_surface',normal_geometry_signature=geometry_signature(),
                        normal_input='Complete known-hand CAD: anatomical-site normals for geometry; observed-contact-centroid normals for contact/force. No true object normals.',
                        supersedes_unexecuted_preparation='mixed_geometry_mass_v1; no additional optimizer budget')
    elif normal_policy!='stored':raise ValueError(normal_policy)
    return protocol


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    protocol=plan(args.probe,args.support,args.normal_policy)
    path=root/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol: raise RuntimeError('Existing prospective protocol differs')
    else: path.write_text(json.dumps(protocol,indent=2))
    if args.prepare_only:
        print('MIXED_CORPUS_PROTOCOL_PREPARED');return
    if (root/'MANIFEST.json').exists(): raise FileExistsError('Completed corpus is immutable')
    # Validate every source before publishing anything. Qualification failures
    # are retained; incomplete traces and controller-routing failures are not.
    records=[];used=set()
    for group,source in protocol['sources'].items():
        source=Path(source)
        result=json.loads((source/'RESULT.json').read_text())
        assert len(result['episodes'])==36
        if group=='probe':
            assert result['complete']
        else:
            # The completed support collector predates the probe report schema.
            # Its endpoint is the full cohort plus per-episode balance evidence.
            assert result['all_support_balance_pass']
            assert all(x['frames']==200 and x['support_balance_pass'] and
                       x['normal_balance_pass'] for x in result['episodes'])
        result_records={x['episode']:x for x in result['episodes']}
        assert len(result_records)==36
        metas=sorted(source.glob('episode_*.json'));assert len(metas)==36
        split_counts={s:0 for s in ('train','val','test')}
        for meta_path in metas:
            meta=json.loads(meta_path.read_text());episode=meta['episode']
            assert meta==result_records[episode]
            assert episode not in used;used.add(episode)
            split_counts[meta['split']]+=1
            raw=meta_path.with_suffix('.npz').resolve();assert raw.is_file()
            assert meta['frames']==(7500 if group=='probe' else 200)
            if group=='probe':
                case=Path(meta['source_case'])
                assert json.loads((case/'CONTROLLER_REPLAY_AUDIT.json').read_text())['passed']
            else:
                assert meta['support_balance_pass'] and meta['normal_balance_pass']
            meta.update(acquisition_group=group,sampling_stride=protocol['sampling_stride'][group],
                        source_metadata=str(meta_path),source_metadata_sha256=digest(meta_path),
                        source_data_sha256=digest(raw))
            records.append((meta,raw))
        assert split_counts==protocol['expected_split_per_source']
    for meta,raw in records:
        destination=root/f"episode_{meta['episode']:04d}"
        if destination.with_suffix('.json').exists() or destination.with_suffix('.npz').exists():
            raise FileExistsError('Partial view requires inspection; never overwrite')
        destination.with_suffix('.npz').symlink_to(raw)
        destination.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    manifest=dict(complete=True,episodes=[m for m,_ in records],protocol_sha256=digest(path),
                  physical_failures_retained=True,training_admitted=False)
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('MIXED_CORPUS_COMPLETE',len(records))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',required=True);ap.add_argument('--support',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--normal-policy',choices=('stored','hand_surface'),default='stored');main(ap.parse_args())

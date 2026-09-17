"""Explicit binding to the one current retained allocation and serial GPU lock."""
import fcntl
import json
import os
from pathlib import Path
import socket


ROOT=Path(__file__).resolve().parents[3]
RESOURCE=ROOT/'experiments/object_predictor_v1/overfit_repair_v1/ACTIVE_RESOURCE.json'
LOCK=ROOT/'experiments/object_predictor_v1/prospective_carry_v1/held_pipeline.lock'


def require_active_resource():
    actual=json.loads(RESOURCE.read_text())
    if actual.get('state')!='RUNNING':
        raise RuntimeError('Current explicitly retained resource is not recorded RUNNING')
    if (os.environ.get('SLURM_JOB_ID')!=str(actual['job_id']) or
            os.environ.get('SLURM_STEP_ID')!=str(actual['step_id']) or
            socket.gethostname().split('.')[0]!=actual['host'] or
            actual['host'].startswith(('login','mgmtserver'))):
        raise RuntimeError('Use the explicitly bound retained compute allocation/step/host')
    try:
        descriptor=os.fstat(9);expected=LOCK.stat()
        if (descriptor.st_dev,descriptor.st_ino)!=(expected.st_dev,expected.st_ino):
            raise RuntimeError('Descriptor9 is not the shared serial GPU lock')
        fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        raise RuntimeError('Retained runner must own the shared exclusive GPU lock on fd9') from error
    return actual

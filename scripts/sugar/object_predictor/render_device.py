"""Use explicit Mesa software EGL for faithful mesh replay on the retained node.

The current H200 NVIDIA EGL path returned black RGB; selecting its DRM device
failed initialization. Software rasterization changes no mesh, pose or model.
"""
import functools
import json
import os


@functools.lru_cache(maxsize=1)
def configure_retained_egl():
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Rendering requires the retained compute step')
    # Mesa documents this option at https://docs.mesa3d.org/egl.html.
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = 'true'
    os.environ.setdefault('LP_NUM_THREADS', '2')
    from pyrender.platforms.egl import query_devices, _eglQueryDeviceStringEXT
    from OpenGL.EGL import EGL_EXTENSIONS
    matches = []
    for index, device in enumerate(query_devices()):
        extensions = _eglQueryDeviceStringEXT(device._display, EGL_EXTENSIONS) or b''
        if b'EGL_MESA_device_software' in extensions.split():
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError('Cannot identify a unique Mesa software EGL device')
    os.environ['EGL_DEVICE_ID'] = str(matches[0])
    result = dict(backend='Mesa software EGL', egl_device_id=matches[0],
                  reason='Current NVIDIA EGL black-frame/initialization failure; full original meshes retained')
    print('RENDER_DEVICE ' + json.dumps(result), flush=True)
    return result

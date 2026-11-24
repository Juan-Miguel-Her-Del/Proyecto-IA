"""Políticas numéricas V23 explícitas, verificables y limitadas al contexto."""
from contextlib import contextmanager
import os
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

VERSION='v20_backend_v23_1'


def politica(modo):
    if modo not in ['minimo','completo']:raise ValueError('Backend numérico desconocido')
    completo=modo=='completo'
    return {'version':VERSION,'modo':modo,'sdpa':'math','amp':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
        'deterministic_algorithms':completo,'tf32_matmul':False,'tf32_cudnn':not completo,
        'float32_matmul_precision':'highest','CUBLAS_WORKSPACE_CONFIG':':4096:8' if completo else None,
        'clip_foreach':False if completo else None,'adam_foreach':False if completo else None,'adam_fused':False if completo else None}


def validar_politica(p):
    if p!=politica(p.get('modo')):raise ValueError('Política numérica alterada')


def registro_efectivo(opt=None):
    return {'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'threads':torch.get_num_threads(),
        'threads_interop':torch.get_num_interop_threads(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'sdpa':{'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math':torch.backends.cuda.math_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()},
        'adam':[{k:g.get(k) for k in ['foreach','fused']} for g in opt.param_groups] if opt else None}


def verificar_registro(p,r,anterior=None):
    validar_politica(p)
    for k in ['cudnn_deterministic','cudnn_benchmark','deterministic_algorithms','tf32_matmul','tf32_cudnn',
        'float32_matmul_precision','CUBLAS_WORKSPACE_CONFIG']:
        if r[k]!=p[k]:raise ValueError('Opción numérica incompatible: '+k)
    if r['sdpa']!={'flash':False,'efficient':False,'math':True,'cudnn':False}:raise ValueError('SDPA incompatible')
    if r.get('adam') is not None and any(g!={'foreach':p['adam_foreach'],'fused':p['adam_fused']} for g in r['adam']):
        raise ValueError('AdamW numéricamente incompatible')
    if anterior is not None and r!=anterior:raise ValueError('Entorno efectivo distinto del checkpoint')


def configurar_optimizador(p,opt):
    validar_politica(p)
    if p['modo']=='completo':
        for g in opt.param_groups:g.update(foreach=False,fused=False)
    elif any(g.get('foreach') is not None or g.get('fused') is not None for g in opt.param_groups):
        raise ValueError('El mínimo debe conservar AdamW histórico')


@contextmanager
def aplicar_backend(p):
    validar_politica(p)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=p['CUBLAS_WORKSPACE_CONFIG']:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG debe fijarse antes de iniciar Python')
    antes=registro_efectivo()
    if p['modo']=='minimo':
        # Solo dos flags cuDNN y restricción SDPA; rechazar otros defaults distintos.
        for k in ['deterministic_algorithms','tf32_matmul','tf32_cudnn','float32_matmul_precision']:
            if antes[k]!=p[k]:raise ValueError('Opción histórica distinta: '+k)
    else:
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            verificar_registro(p,registro_efectivo());yield
    finally:
        torch.use_deterministic_algorithms(antes['deterministic_algorithms'])
        torch.backends.cudnn.deterministic=antes['cudnn_deterministic'];torch.backends.cudnn.benchmark=antes['cudnn_benchmark']
        torch.backends.cudnn.allow_tf32=antes['tf32_cudnn'];torch.backends.cuda.matmul.allow_tf32=antes['tf32_matmul']
        torch.set_float32_matmul_precision(antes['float32_matmul_precision'])

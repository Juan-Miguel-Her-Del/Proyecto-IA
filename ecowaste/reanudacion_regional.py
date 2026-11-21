"""Diario de épocas inmutables y restauración verificable del experimento regional."""
from contextlib import contextmanager
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile
import copy
import torch
from ecowaste.ablacion_fondo import huella,restaurar_aleatorio,estado_aleatorio

VERSION=1
RNG=('rng_torch','rng_cuda','rng_loader','rng_auxiliar')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def huellas_estado(ck):
    return {'pesos':huella(ck['pesos']),'adamw':huella(ck['optimizador']),
        'scaler':huella(ck['scaler']),'rng':huella({k:ck[k] for k in RNG})}


def restaurar_estados(modelo,opt,scaler,ck,rng,aux):
    modelo.load_state_dict(ck['pesos'],strict=True)
    opt.load_state_dict(copy.deepcopy(ck['optimizador']))
    scaler.load_state_dict(copy.deepcopy(ck['scaler']))
    restaurar_aleatorio(ck,rng,aux)
    real={'pesos':huella(modelo.state_dict()),'adamw':huella(opt.state_dict()),
        'scaler':huella(scaler.state_dict()),'rng':huella(estado_aleatorio(rng,aux))}
    if real!=huellas_estado(ck):raise ValueError('Restauración incompleta')
    return real


def publicar(path,valor,tensor=False):
    """Publica completo sin reemplazar un destino existente (también ante carreras)."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.pendiente',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            if tensor:torch.save(valor,f)
            else:f.write(json.dumps(valor,ensure_ascii=False,indent=2).encode('utf-8'))
            f.flush();os.fsync(f.fileno())
        os.link(tmp,path)  # falla si existe; nunca deja un destino parcialmente escrito
    finally:
        Path(tmp).unlink(missing_ok=True)


@contextmanager
def bloqueo(path):
    """El SO libera el bloqueo al morir el proceso; el archivo no indica propiedad."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def validar_estado(ck,cfg,rama,diagnostico,codigos,previo):
    """No acepta puntos intermedios, otra rama, protocolo, código o trazas."""
    p={k:v for k,v in cfg.items() if k!='entrenamiento_completo_habilitado'}
    actual={k:v for k,v in ck['protocolo_regional'].items() if k!='entrenamiento_completo_habilitado'}
    r=ck['reanudacion'];epoca=ck['epoca'];hist=ck['historial']
    if actual!=p or r['version']!=VERSION or r['rama']!=rama or r['commit_previo_sha256']!=previo:
        raise ValueError('Protocolo, rama o cadena de reanudación incompatible')
    if ck['sha256_origen']!=cfg['sha256_checkpoint_origen'] or ck['hashes_codigo']!=codigos:
        raise ValueError('Origen o código incompatibles')
    if ck['huella_regiones']!=diagnostico['huella_regiones'] or ck['hashes_manifiestos']!=diagnostico['hashes_datos']:
        raise ValueError('Datos o regiones incompatibles')
    if r['sha256_diagnostico']!=diagnostico['sha256_archivo'] or not r['epoca_completa']:
        raise ValueError('Diagnóstico distinto o época incompleta')
    if not 20<=epoca<=31 or len(hist)!=epoca-19:raise ValueError('Historial incompleto')
    for i,h in enumerate(hist):
        t=diagnostico['trazas_preparadas'][rama][i]
        if h['epoca']!=20+i or h['actualizaciones']!=121*(i+1) or h['paso_adamw']!=2299+121*(i+1):
            raise ValueError('Presupuesto o contador alterados')
        if h['coeficiente_regional']!=cfg['ramas'][rama] or h['learning_rate']!=.0005 or h['huella_orden_y_aumentos']!=t['huella_orden_y_aumentos']:
            raise ValueError('Coeficiente, LR o entradas incompatibles')
        organic=h['region_organica'];referencia=diagnostico['region_organica_elegible']
        if organic['presencia']!=1 or organic['peso']!=referencia['peso'] or len(organic['eventos'])!=1:
            raise ValueError('Presencia o peso orgánico alterado')
        evento=organic['eventos'][0]
        if any(evento[k]!=referencia[k] for k in ['imagen','id_region','indice_train']):raise ValueError('Otra región orgánica')
        if not math.isclose(evento['contribucion_loss_total'],cfg['ramas'][rama]*evento['contribucion_ce'],rel_tol=1e-9,abs_tol=1e-12):
            raise ValueError('Contribución orgánica incoherente')
    pasos=121*(epoca-19)
    if ck['actualizaciones_desde_origen']!=pasos or pasos>1452:raise ValueError('Presupuesto excedido')
    if {int(s['step']) for s in ck['optimizador']['state'].values()}!={2299+pasos}:
        raise ValueError('AdamW no corresponde al contador')
    if any(g['lr']!=.0005 for g in ck['optimizador']['param_groups']):raise ValueError('LR alterado')
    if r['huellas_estado']!=huellas_estado(ck):raise ValueError('Estado de pesos/AdamW/scaler/RNG alterado')
    if r['estado_inicial_origen']!=diagnostico['coste_diagnostico'][rama]['estado_inicial']:
        raise ValueError('Estados iniciales distintos del origen común')
    coef=ck['config_entrenamiento']['supervision_regional']
    if coef['coeficiente']!=cfg['ramas'][rama] or coef['pesos_train']!=diagnostico['estadistica_regiones']['pesos_clase']:
        raise ValueError('Peso regional alterado')
    mejor=max(hist,key=lambda h:h['map50_val'])
    if ck['mejor_epoca']!=mejor['epoca'] or ck['mejor_map']!=mejor['map50_val']:
        raise ValueError('Criterio del mejor alterado')
    return pasos


class DiarioRama:
    def __init__(self,raiz,cfg,rama,diagnostico,codigos):
        self.raiz=Path(raiz);self.cfg=cfg;self.rama=rama;self.diag=diagnostico;self.codigos=codigos
        self.nombre=f"{cfg['nombre']}_{rama}"
        self.directorio=self.raiz/'resultados'/cfg['nombre']/f'rama_{rama}'

    def checkpoint(self,epoca):return self.raiz/'checkpoints'/f'{self.nombre}_epoca_{epoca:03d}.pt'
    def registro(self,epoca):return self.directorio/f'epoca_{epoca:03d}.json'

    def evento(self,ck,path,recuperado=False):
        return {'version':VERSION,'rama':self.rama,'epoca':ck['epoca'],
            'checkpoint':path.relative_to(self.raiz).as_posix(),'sha256_checkpoint':sha(path),
            'commit_previo_sha256':ck['reanudacion']['commit_previo_sha256'],
            'huellas_estado':ck['reanudacion']['huellas_estado'],'historial_epoca':ck['historial'][-1],
            'recuperado_tras_interrupcion':recuperado}

    def leer(self,recuperar=False):
        registros=[];ultimo=None;previo=None;hueco=False
        for epoca in range(20,32):
            cp=self.checkpoint(epoca);rp=self.registro(epoca)
            if not cp.exists() and not rp.exists():hueco=True;continue
            if hueco or not cp.exists():raise ValueError('Hueco en el diario o falta checkpoint')
            ck=torch.load(cp,map_location='cpu',weights_only=True)
            validar_estado(ck,self.cfg,self.rama,self.diag,self.codigos,previo)
            if ck['epoca']!=epoca:raise ValueError('Época y nombre incompatibles')
            if rp.exists():
                ev=json.loads(rp.read_text(encoding='utf-8'))
                esperado=self.evento(ck,cp,ev.get('recuperado_tras_interrupcion',False))
                if ev!=esperado:raise ValueError('Registro o checksum alterados')
            elif recuperar:
                ev=self.evento(ck,cp,True);publicar(rp,ev)
            else:raise ValueError('Checkpoint completo sin commit: solicitar --reanudar para recuperar')
            if ultimo is not None and ck['historial'][:-1]!=ultimo['historial']:
                raise ValueError('La rama reescribió su historia')
            registros.append(ev);ultimo=ck;previo=sha(rp)
        return ultimo,registros,previo

    def confirmar(self,ck,previo):
        validar_estado(ck,self.cfg,self.rama,self.diag,self.codigos,previo)
        cp=self.checkpoint(ck['epoca']);rp=self.registro(ck['epoca'])
        publicar(cp,ck,tensor=True);publicar(rp,self.evento(ck,cp))
        return sha(rp)

    def actualizar_alias(self,registros):
        """Solo reemplaza aliases derivados cuyo hash pertenece al mismo diario."""
        if not registros:return
        conocidos={r['sha256_checkpoint'] for r in registros}
        mejor=max(registros,key=lambda r:r['historial_epoca']['map50_val'])
        for sufijo,r in [('ultimo',registros[-1]),('mejor',mejor)]:
            destino=self.raiz/'checkpoints'/f'{self.nombre}_{sufijo}.pt'
            if destino.exists() and sha(destino) not in conocidos:raise ValueError('Alias ajeno al diario; no se sobrescribe')
            fuente=self.raiz/r['checkpoint']
            if destino.exists() and sha(destino)==r['sha256_checkpoint']:continue
            fd,tmp=tempfile.mkstemp(prefix='.'+destino.name,dir=destino.parent);os.close(fd)
            try:
                # Copia independiente: no compartir inodos mutables con el histórico.
                import shutil
                shutil.copyfile(fuente,tmp)
                with open(tmp,'r+b') as f:os.fsync(f.fileno())
                os.replace(tmp,destino)
            finally:Path(tmp).unlink(missing_ok=True)

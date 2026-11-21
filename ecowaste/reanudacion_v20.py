"""Diario exclusivo V20; reutiliza publicación, bloqueos y aliases seguros V18."""
import json
import math
import torch
from ecowaste.reanudacion_regional import DiarioRama,publicar,sha,huellas_estado
from ecowaste.backend_numerico import verificar_registro


def validar_estado_v20(ck,cfg,rama,diag,codigos,previo):
    p={k:v for k,v in cfg.items() if k!='entrenamiento_completo_habilitado'}
    r=ck['reanudacion'];hist=ck['historial'];ep=ck['epoca']
    if ck.get('protocolo_v20')!=p or ck.get('hashes_codigo_v20')!=codigos:
        raise ValueError('Protocolo o código V20 incompatible; no adoptar V18')
    if cfg.get('backend_numerico'):
        if ck.get('backend_numerico')!=cfg['backend_numerico']:raise ValueError('Checkpoint de otro backend numérico')
        verificar_registro(cfg['backend_numerico'],ck['backend_efectivo'])
    if r.get('tipo')!='v20_margen_huber' or r['rama']!=rama or r['commit_previo_sha256']!=previo or not r['epoca_completa']:
        raise ValueError('Rama, cadena o época incompatible')
    if ck['sha256_origen']!=cfg['sha256_checkpoint_origen'] or r['sha256_diagnostico']!=diag['sha256_archivo']:
        raise ValueError('Origen o diagnóstico distinto')
    if ck['huella_regiones']!=diag['huella_regiones'] or ck['hashes_manifiestos']!=diag['hashes_datos']:
        raise ValueError('Datos o regiones distintos')
    if r['estado_inicial_origen']!=diag['estado_inicial'] or r['huellas_estado']!=huellas_estado(ck):
        raise ValueError('Estados de pesos/AdamW/scaler/RNG distintos')
    if not 20<=ep<=31 or len(hist)!=ep-19:raise ValueError('Historial incompleto')
    for i,h in enumerate(hist):
        if h['epoca']!=20+i or h['actualizaciones']!=121*(i+1) or h['paso_adamw']!=2299+121*(i+1):
            raise ValueError('Presupuesto alterado')
        if h['coeficiente_discriminativo']!=cfg['ramas'][rama] or h['learning_rate']!=.0005:
            raise ValueError('Coeficiente o LR alterados')
        if h['huella_orden_y_aumentos']!=diag['trazas_preparadas'][rama][i]['huella_orden_y_aumentos']:
            raise ValueError('Traza de entradas distinta')
        org=h['region_organica']
        if org['presencias']!=1 or len(org['eventos'])!=1:raise ValueError('Presencia orgánica alterada')
        ev=org['eventos'][0]
        for k in ['imagen','id_region']:
            if ev[k]!=diag['organico_elegible'][k]:raise ValueError('Otra instancia orgánica')
        if ev['peso_clase']!=1 or not math.isclose(ev['contribucion_loss_total'],cfg['ramas'][rama]*ev['contribucion'],abs_tol=1e-12):
            raise ValueError('Peso o contribución orgánica alterado')
    pasos=121*(ep-19)
    if ck['actualizaciones_desde_origen']!=pasos or pasos>1452:raise ValueError('Presupuesto inválido')
    if {int(s['step']) for s in ck['optimizador']['state'].values()}!={2299+pasos}:raise ValueError('AdamW inconsistente')
    if any(g['lr']!=.0005 for g in ck['optimizador']['param_groups']):raise ValueError('LR diferente')
    esperado={'coeficiente':cfg['ramas'][rama],'margen':1.,'fraccion':.25,'delta_huber':1.,'pesos':[1.,1.,2.,2.,1.,1.]}
    if ck['config_entrenamiento']['v20']!=esperado:raise ValueError('Hipótesis modificada')
    if cfg.get('diagnostico_reanudacion_larga'):
        if not cfg['nombre'].startswith('v23_control_') or cfg['entrenamiento_completo_habilitado'] or ep!=20:
            raise ValueError('Diario diagnóstico ajeno al límite de época aprobado')
        if any('map50_val' in h for h in hist) or ck.get('mejor_epoca') is not None or ck.get('mejor_map') is not None:
            raise ValueError('El control train-only no selecciona modelos')
    else:
        mejor=max(hist,key=lambda h:h['map50_val'])
        if ck['mejor_epoca']!=mejor['epoca'] or ck['mejor_map']!=mejor['map50_val']:raise ValueError('Selección del mejor alterada')


class DiarioV20(DiarioRama):
    def actualizar_alias(self,registros):
        if self.cfg.get('diagnostico_reanudacion_larga'):return # checkpoints inmutables; sin selección
        return super().actualizar_alias(registros)
    def evento(self,ck,path,recuperado=False):
        return {**super().evento(ck,path,recuperado),'version':20,'tipo':'v20_margen_huber'}

    def leer(self,recuperar=False):
        registros=[];ultimo=None;previo=None;hueco=False
        for ep in range(20,32):
            cp=self.checkpoint(ep);rp=self.registro(ep)
            if not cp.exists() and not rp.exists():hueco=True;continue
            if hueco or not cp.exists():raise ValueError('Hueco o checkpoint ausente')
            ck=torch.load(cp,map_location='cpu',weights_only=True)
            validar_estado_v20(ck,self.cfg,self.rama,self.diag,self.codigos,previo)
            if ck['epoca']!=ep:raise ValueError('Época incompatible')
            if rp.exists():
                ev=json.loads(rp.read_text(encoding='utf-8'))
                if ev!=self.evento(ck,cp,ev.get('recuperado_tras_interrupcion',False)):raise ValueError('Registro alterado')
            elif recuperar:
                ev=self.evento(ck,cp,True);publicar(rp,ev)
            else:raise ValueError('Checkpoint sin commit: usar reanudación explícita')
            if ultimo is not None and ck['historial'][:-1]!=ultimo['historial']:raise ValueError('Se reescribió el historial')
            registros.append(ev);ultimo=ck;previo=sha(rp)
        return ultimo,registros,previo

    def confirmar(self,ck,previo):
        validar_estado_v20(ck,self.cfg,self.rama,self.diag,self.codigos,previo)
        cp=self.checkpoint(ck['epoca']);publicar(cp,ck,tensor=True)
        publicar(self.registro(ck['epoca']),self.evento(ck,cp))
        return sha(self.registro(ck['epoca']))

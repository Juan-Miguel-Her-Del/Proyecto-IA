"""Captura sin mutar el paso compartido; conserva detalles si una huella cambia."""
from pathlib import Path
from ecowaste.ablacion_fondo import huella,estado_aleatorio,pasos_adam
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.reproducibilidad_v20 import cpu,gradientes,adam_nominal
import json


class ObservadorBackend:
    def __init__(self,m,opt,scaler,rng,ra,salida,referencia,pasos=0):
        self.m=m;self.opt=opt;self.sc=scaler;self.rng=rng;self.ra=ra
        self.salida=Path(salida);self.ref=Path(referencia);self.k=pasos+1;self.d={}

    def __call__(self,etapa,**datos):
        if etapa=='inicio':
            self.d={'pesos_antes':cpu(self.m.state_dict()),'adamw_antes':adam_nominal(self.m,self.opt),'scaler_antes':self.sc.state_dict(),
                'rng_inicio':cpu(estado_aleatorio(self.rng,self.ra))}
        elif etapa=='forward_taco':self.d.update(logits_taco=cpu(datos.pop('logits')),losses_taco=cpu(datos))
        elif etapa=='forward_aux':self.d.update(logits_aux=cpu(datos['logits']),loss_global=cpu(datos['loss']))
        elif etapa in ['grad_taco','grad_total','sin_escala']:self.d[etapa]=gradientes(self.m)
        elif etapa=='clip':self.d.update(grad_clip=gradientes(self.m),norma=cpu(datos['norma']),entradas=cpu(datos['entradas']))
        elif etapa=='post':self.d.update(pesos_despues=cpu(self.m.state_dict()),adamw_despues=adam_nominal(self.m,self.opt),
            scaler_despues=self.sc.state_dict(),rng_fin=cpu(estado_aleatorio(self.rng,self.ra)))

    def finalizar(self,s,pasos):
        if pasos!=self.k:raise ValueError('Observador desalineado')
        hs={n:huella(v) for n,v in self.d.items()};nombre=f'paso_{pasos:04d}'
        ref=self.ref/(nombre+'.json');iguales=False
        if ref.exists():iguales=json.loads(ref.read_text(encoding='utf-8'))['hashes_etapas']==hs
        persistir=self.salida==self.ref or pasos in [1,2,120,121,122,125] or not iguales
        p=self.salida/(nombre+'.pt')
        if persistir:publicar(p,self.d,tensor=True)
        publicar(self.salida/(nombre+'.json'),{'paso':pasos,'paso_adamw':pasos_adam(self.opt),'hashes_etapas':hs,
            'loss_original':s['loss_original'],'loss_nueva':s['loss_nueva'],'loss_global':s['loss_global'],
            'norma_antes_clip':s['norma_antes_clip'],'huella_entrada':s['huella_entrada'],'scaler':self.sc.state_dict(),
            'instantanea_completa':persistir,'sha256_instantanea':sha(p) if persistir else None,
            'detalle_regional':s['detalle']})
        self.k+=1;self.d={}

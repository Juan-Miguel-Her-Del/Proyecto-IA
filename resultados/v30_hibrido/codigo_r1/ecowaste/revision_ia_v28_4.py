"""Revisión humana de RLE IA, aislada de V26 y de los diarios V27.2.

Las decisiones por propuesta y la confirmación explícita de cobertura son eventos
distintos. Toda salida sigue bloqueada para entrenamiento y aceptación V26.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import threading
import uuid
import cv2
import numpy as np
from PIL import Image
from ecowaste.incorporacion import leer, publicar, sha, digest, bloqueo, identificador
from ecowaste.auxiliares_publicos import rle_coco, MATERIALES
from ecowaste.ajustes_ia_v28_3 import canonizar, decodificar, caja, CLASES
from ecowaste.validacion_anotaciones import celdas, max_componente

VERSION = 'revision_humana_RLE_IA_v28_4_1'
RAIZ = Path(__file__).resolve().parents[1]
RESUELTAS = {'aceptar', 'rechazar', 'editar'}


def sello():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]


def semantica(a):
    if a is None:
        return None
    return {k:a.get(k) for k in ['id','material','descartada','motivo','segmentacion_canonica_RLE']}


def validar_anotacion(a, im):
    if not isinstance(a, dict):
        raise ValueError('Objeto de anotación requerido')
    b=deepcopy(a);b['id']=identificador(b.get('id'))
    if b.get('material') not in MATERIALES:
        raise ValueError('Material no admitido: '+b['id'])
    if type(b.get('descartada',False)) is not bool:
        raise ValueError('Estado de descarte inválido')
    b['descartada']=b.get('descartada',False)
    if not isinstance(b.get('motivo',''),str):
        raise ValueError('Motivo debe ser texto')
    if b['descartada']:
        if len(b.get('motivo','').strip())<5:
            raise ValueError('Descarte sin motivo: '+b['id'])
        return b
    if b['material']=='sin_asignar':
        raise ValueError('Falta asignar material: '+b['id'])
    if b['material'] not in CLASES and len(b.get('motivo','').strip())<5:
        raise ValueError('Región ignore sin justificación: '+b['id'])
    r=b.get('segmentacion_canonica_RLE')
    if not isinstance(r,dict) or r.get('size')!=[im['height'],im['width']]:
        raise ValueError('RLE de otras dimensiones: '+b['id'])
    cs=r.get('counts');n=im['width']*im['height']
    if not isinstance(cs,list) or not 0<len(cs)<=2*n+1 or any(type(v) is not int or v<0 for v in cs) or sum(cs)!=n:
        raise ValueError('RLE no válido: '+b['id'])
    m=decodificar(r)
    if not m.any():
        raise ValueError('Máscara vacía: '+b['id'])
    b.update(bbox=caja(m),area_px=int(m.sum()),ignore=b['material'] not in CLASES,
             aceptada_V26=False,revision_independiente=False,codificacion='RLE_COCO_canonico')
    return b


def construir_unidades(antes, despues, im, base_completa):
    """Agrupar deltas coincidentes: nunca aceptar medio solape o medio despiece."""
    ini={a['id']:a for a in antes};fin={a['id']:a for a in despues};candidatos=[];deltas={}
    for aid in sorted(set(ini)|set(fin)):
        a,b=ini.get(aid),fin.get(aid)
        cambia=digest(semantica(a))!=digest(semantica(b))
        formato=any(x['accion']=='relleno_canonico_explicito' for x in (b or {}).get('ajustes_IA',[]))
        if not (cambia or formato or not base_completa):
            continue
        tipos=[]
        if a is None:tipos.append('objeto_nuevo')
        if a and b and a.get('material')!=b.get('material'):tipos.append('material')
        if a and b and a.get('segmentacion_canonica_RLE')!=b.get('segmentacion_canonica_RLE'):tipos.append('contorno')
        if b and b.get('material') not in CLASES:tipos.append('ignore')
        if formato:tipos.append('relleno_RLE')
        if not tipos:tipos.append('confirmar_base_sin_revision_completa')
        candidatos.append({'objeto':aid,'tipos':tipos})
        z=np.zeros((im['height'],im['width']),bool)
        ma=decodificar(a['segmentacion_canonica_RLE']) if a and not a.get('descartada') and a.get('segmentacion_canonica_RLE') else z
        mb=decodificar(b['segmentacion_canonica_RLE']) if b and not b.get('descartada') and b.get('segmentacion_canonica_RLE') else z
        delta=(ma|mb) if a and b and a.get('material')!=b.get('material') else ma^mb
        yy,xx=np.where(delta)
        box=None if not len(xx) else (int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max()))
        deltas[aid]=(np.packbits(delta.ravel()),box)
    padres={c['objeto']:c['objeto'] for c in candidatos}
    def raiz(x):
        while padres[x]!=x:x=padres[x]
        return x
    for i,c in enumerate(candidatos):
        p,bp=deltas[c['objeto']]
        if bp is None:continue
        for d in candidatos[:i]:
            q,bq=deltas[d['objeto']]
            if bq is not None and max(bp[0],bq[0])<=min(bp[2],bq[2]) and max(bp[1],bq[1])<=min(bp[3],bq[3]) and np.any(p&q):
                padres[raiz(c['objeto'])]=raiz(d['objeto'])
    grupos={}
    for c in candidatos:grupos.setdefault(raiz(c['objeto']),[]).append(c)
    unidades=[]
    for grupo in grupos.values():
        ids=sorted(c['objeto'] for c in grupo)
        dato={'objetos':ids,'antes':[ini[i] for i in ids if i in ini],
              'propuesta':[fin[i] for i in ids if i in fin],
              'tipos':sorted({t for c in grupo for t in c['tipos']}),'dependencias_atomicas':len(ids)>1,
              'restaurar_base_requiere_revision':not base_completa and any(i in ini and not ini[i].get('descartada') for i in ids)}
        dato['huella']=digest(dato);dato['id']='prop_'+dato['huella'][:24];unidades.append(dato)
    return sorted(unidades,key=lambda u:('objeto_nuevo' not in u['tipos'],u['id']))


def qa_rle(im, anns):
    errores=[];avisos=[];objetos=[];masks={};maps={};w,h=im['width'],im['height']
    s=min(384/w,384/h);nw,nh=round(w*s),round(h*s);dx,dy=(384-nw)//2,(384-nh)//2
    contenido=np.zeros((384,384),np.uint8);contenido[dy:dy+nh,dx:dx+nw]=1
    tokens=celdas(contenido)==1
    for a in anns:
        try:b=validar_anotacion(a,im)
        except (ValueError,TypeError,KeyError) as e:
            errores.append({'objeto':a.get('id'),'motivo':str(e)});continue
        if b['descartada']:continue
        m=decodificar(b['segmentacion_canonica_RLE']);masks[b['id']]=m
        out=Image.new('L',(384,384));out.paste(Image.fromarray(np.uint8(m)).resize((nw,nh),Image.Resampling.NEAREST),(dx,dy))
        maps[b['id']]=celdas(np.asarray(out))
        _,_,stats,_=cv2.connectedComponentsWithStats(np.uint8(m),8)
        if sum(int(x)>=max(4,.01*b['area_px']) for x in stats[1:,cv2.CC_STAT_AREA])>1:
            avisos.append({'objetos':[b['id']],'motivo':'Componentes separados: confirmar identidad, oclusión o granularidad.'})
        objetos.append({'id':b['id'],'material':b['material'],'area_px':b['area_px']})
    solapes=[]
    for i,a in enumerate(objetos):
        ma=masks[a['id']]
        for b in objetos[:i]:
            mb=masks[b['id']];inter=int(np.count_nonzero(ma&mb))
            if inter>4 and inter/max(1,min(a['area_px'],b['area_px']))>.01:
                datos={'objetos':sorted([a['id'],b['id']]),'pixeles':inter,'motivo':'Solape visible: comprobar contornos y justificar identidad/oclusión.'}
                avisos.append(datos);solapes.append(datos)
                if np.array_equal(ma,mb):errores.append({'objetos':datos['objetos'],'motivo':'Máscaras idénticas para instancias distintas: resolver duplicado.'})
    materiales=np.zeros((6,48,48),bool);ign=np.zeros((48,48),bool)
    for a in objetos:
        if a['material'] in CLASES:materiales[CLASES.index(a['material'])]|=maps[a['id']]>0
        else:ign|=maps[a['id']]>0
    mezcla=materiales.sum(0)>1
    for a in objetos:
        a['componente_maximo_48']=0;a['tokens_regionales']=0
        if a['material'] not in CLASES:continue
        c=maps[a['id']];cand=(c>=1/64)&tokens&~mezcla&~ign;interior=cand&(c==1)
        region=interior if interior.any() else cand
        a['componente_maximo_48']=max_componente(region);a['tokens_regionales']=int(region.sum())
    fusiones=[]
    for ci,material in enumerate(CLASES):
        n,lab,_,_=cv2.connectedComponentsWithStats(np.uint8(materiales[ci]&tokens&~mezcla&~ign),8)
        for k in range(1,n):
            ids=[a['id'] for a in objetos if a['material']==material and np.count_nonzero((maps[a['id']]>0)&(lab==k))>=2]
            if len(ids)>1:fusiones.append({'material':material,'objetos':ids,'no_es_error_GT_demostrado':True})
    return {'errores':errores,'avisos':avisos,'objetos':objetos,'solapes':solapes,
            'fusiones_48':fusiones,'tokens_mezclados':int(mezcla.sum()),'tokens_ignore':int(ign.sum()),
            'rasterizacion':'mascara_RLE_nativa_a_letterbox384_NEAREST; no_equivale_al_rasterizador_poligonal_historico'}


class RevisionIA:
    def __init__(self, carpeta, raiz=RAIZ, taller=None):
        self.path=Path(carpeta);self.raiz=Path(raiz);self.taller=taller;self.lock=threading.RLock()
        self.catalogo=leer(self.path/'catalogo.json')
        if self.catalogo['version']!=VERSION or self.catalogo['huella']!=digest({k:v for k,v in self.catalogo.items() if k!='huella'}):
            raise ValueError('Catálogo incompatible o alterado')
        self.id=self.catalogo['huella'];self.imagenes={r['imagen']['id']:r for r in self.catalogo['imagenes']}
        self.catalogo_archivo_sha256=sha(self.path/'catalogo.json')
        self.verificar_fuentes()
        self._eventos_cache=None;self._cache_paths=None
        self.eventos()

    def verificar_fuentes(self):
        if sha(self.path/'catalogo.json')!=self.catalogo_archivo_sha256:raise ValueError('Catálogo congelado alterado')
        for nombre,h in self.catalogo.get('archivos_fuente',{}).items():
            if sha(self.raiz/nombre)!=h:raise ValueError('Fuente IA alterada: '+nombre)

    def origen_vigente(self,iid):
        if self.taller is None:return True
        ev=next((e for e in reversed(self.taller.diario(iid)) if e['modo']=='asistida'),None)
        return (ev or {}).get('huella')==self.imagenes[iid]['revision_humana_base']

    def eventos(self):
        paths=tuple(sorted((self.path/'eventos').glob('*.json')))
        # Se verifican bytes en cada lectura: una caché de rutas no detecta alteraciones.
        evs=[];prev=None;requests=set()
        for n,p in enumerate(paths,1):
            e=leer(p)
            if p.name!=f'{n:06d}.json' or e['numero']!=n or e['anterior']!=prev or e['proyecto_sha256']!=self.id or e['version']!=VERSION or e['huella']!=digest({k:v for k,v in e.items() if k!='huella'}):
                raise ValueError('Cadena de eventos humanos V28.4 alterada')
            if e['id_solicitud'] in requests:raise ValueError('Solicitud duplicada en diario')
            requests.add(e['id_solicitud']);evs.append(e);prev=e['huella']
        return evs

    def estado(self,iid,eventos=None):
        r=self.imagenes[iid];evs=self.eventos() if eventos is None else eventos
        decisiones={};cobertura=None;ultima=None
        for e in evs:
            if e['imagen_id']!=iid:continue
            ultima=e['huella']
            if e['accion']=='confirmar_cobertura':cobertura=e
            else:
                for d in e['decisiones']:decisiones[d['unidad_id']]={**d,'evento_huella':e['huella'],'persona_id':e['persona_id']}
        anns={a['id']:deepcopy(a) for a in r['base']}
        unidades=[]
        for u in r['unidades']:
            d=decisiones.get(u['id']);estado=d['accion'] if d else 'pendiente'
            elegidas=d['anotaciones'] if d and estado in RESUELTAS else u['antes']
            for aid in u['objetos']:anns.pop(aid,None)
            for a in elegidas:
                obj=deepcopy(a)
                if d and estado in RESUELTAS:
                    obj['revision_V284']={'accion':estado,'persona_id':d['persona_id'],'evento_huella':d['evento_huella'],
                        'ajuste_IA_aceptado':estado in ['aceptar','editar'],'revision_independiente':False,'aceptada_V26':False}
                anns[a['id']]=obj
            unidades.append({**u,'estado':estado,'decision':d})
        final=list(anns.values());contenido=digest([semantica(a) for a in sorted(final,key=lambda a:a['id'])])
        pendiente=sum(u['estado'] not in RESUELTAS for u in unidades)
        # Cualquier decisión posterior invalida la cobertura anterior, incluso posponer sin cambiar píxeles.
        ev_ultima_decision=next((e['huella'] for e in reversed(evs) if e['imagen_id']==iid and e['accion']!='confirmar_cobertura'),None)
        completa=bool(cobertura and cobertura['contenido_sha256']==contenido and cobertura['ultima_decision_sha256']==ev_ultima_decision and not pendiente)
        huella=digest({'proyecto':self.id,'imagen':iid,'ultima':ultima,'contenido':contenido})
        return {'imagen':r['imagen'],'unidades':unidades,'anotaciones':final,'contenido_sha256':contenido,
                'estado_sha256':huella,'ultima_decision_sha256':ev_ultima_decision,'cobertura_completa':completa,
                'cobertura_evento':cobertura,'pendientes':pendiente,'origen_vigente':self.origen_vigente(iid),
                'advertencias_fuente':r['advertencias'],'aceptada_V26':False}

    def advertencias(self,st,qa):
        filas=[]
        for v in [*st['advertencias_fuente'],*qa['avisos']]:
            motivo=v if isinstance(v,str) else v['motivo']
            dato={'motivo':motivo,'objetos':v.get('objetos',[]) if isinstance(v,dict) else []}
            dato['id']='aviso_'+digest({'dato':dato,'contenido':st['contenido_sha256']})[:24]
            if not any(x['id']==dato['id'] for x in filas):filas.append(dato)
        return filas

    def tarea(self,iid):
        st=self.estado(iid);qa=qa_rle(st['imagen'],st['anotaciones'])
        return {**st,'qa':qa,'advertencias':self.advertencias(st,qa)}

    def guardar(self,d):
        """Un commit atómico e idempotente; nunca escribe el Taller histórico."""
        if not isinstance(d,dict) or d.get('proyecto_sha256')!=self.id:raise ValueError('Otro proyecto o versión')
        iid=d['imagen_id'];r=self.imagenes[iid]
        persona=identificador(d.get('persona_id'));sol=identificador(d.get('id_solicitud'))
        accion=d.get('accion')
        if accion not in [*RESUELTAS,'posponer','confirmar_cobertura']:raise ValueError('Acción inválida')
        payload={k:v for k,v in d.items() if k not in ['token_vista','tokens_inspeccion']};firma=digest(payload)
        with self.lock,bloqueo(self.path/'eventos.lock'):
            evs=self.eventos()
            for e in evs:
                if e['id_solicitud']==sol:
                    if e['solicitud_sha256']!=firma:raise ValueError('Misma solicitud con contenido distinto')
                    return {'evento':e,'reintento_idempotente':True}
            self.verificar_fuentes()
            if not self.origen_vigente(iid):raise ValueError('El guardado humano original cambió; no aplicar esta copia antigua')
            st=self.estado(iid,evs)
            if d.get('estado_sha256')!=st['estado_sha256']:raise ValueError('Conflicto: hay otra decisión guardada; conservar borrador y recargar')
            nota=d.get('nota','')
            if not isinstance(nota,str):raise ValueError('Nota inválida')
            decisiones=[];extra={}
            if accion=='confirmar_cobertura':
                if st['pendientes']:raise ValueError('Quedan propuestas pendientes o pospuestas')
                if any(u['estado']=='rechazar' and u['restaurar_base_requiere_revision'] for u in st['unidades']):
                    raise ValueError('El rechazo restaura propuestas sin revisión humana previa: editar/descartar o aceptar explícitamente esa base')
                qa=qa_rle(st['imagen'],st['anotaciones'])
                if qa['errores']:raise ValueError('Resolver errores técnicos antes de confirmar cobertura')
                vistas=self.advertencias(st,qa)
                if d.get('confirmo_cobertura') is not True or set(d.get('advertencias_inspeccionadas',[]))!={v['id'] for v in vistas} or len(nota.strip())<10:
                    raise ValueError('Confirmación explícita, advertencias inspeccionadas y nota de cobertura requeridas')
                extra={'contenido_sha256':st['contenido_sha256'],'ultima_decision_sha256':st['ultima_decision_sha256'],
                       'advertencias_inspeccionadas':vistas}
            else:
                seleccion=d.get('unidades',[]);porid={u['id']:u for u in st['unidades']}
                if not isinstance(seleccion,list) or not seleccion or len(seleccion)!=len(set(seleccion)) or any(x not in porid for x in seleccion):raise ValueError('Seleccionar propuestas de esta imagen sin duplicados')
                if set(d.get('inspeccionadas',[]))!=set(seleccion):raise ValueError('Inspeccionar explícitamente cada propuesta seleccionada')
                if accion in ['rechazar','editar','posponer'] and len(nota.strip())<5:raise ValueError('Explica el rechazo, edición o revisión posterior')
                if accion=='editar' and len(seleccion)!=1:raise ValueError('Editar una propuesta o grupo atómico por vez')
                for uid in seleccion:
                    u=porid[uid]
                    if d.get('huellas_unidades',{}).get(uid)!=u['huella']:raise ValueError('Propuesta de otra versión')
                    if accion=='editar':
                        elegidas=d.get('anotaciones_editadas')
                        if not isinstance(elegidas,list):raise ValueError('Falta borrador RLE editado')
                        ids=[a.get('id') for a in elegidas]
                        previas={a['id'] for a in (u.get('decision') or {}).get('anotaciones',[])}
                        if len(ids)!=len(set(ids)) or not (set(u['objetos'])|previas).issubset(ids):raise ValueError('No borrar objetos sin descarte explícito')
                        for aid in set(ids)-set(u['objetos']):
                            if aid not in previas and (not str(aid).startswith('usuario_v284_') or any(a['id']==aid for a in st['anotaciones'])):raise ValueError('ID añadido ajeno o duplicado')
                        elegidas=[validar_anotacion(a,st['imagen']) for a in elegidas]
                        for a in elegidas:
                            if a.get('sam_v284_id'):
                                sid=identificador(a['sam_v284_id']);sp=self.path/'sam'/(sid+'.json')
                                sam=leer(sp)
                                if sam['imagen_id']!=iid or sam['proyecto_sha256']!=self.id or sam['huella']!=digest({k:v for k,v in sam.items() if k!='huella'}):
                                    raise ValueError('Procedencia SAM de otra imagen o proyecto')
                    elif accion=='aceptar':elegidas=[validar_anotacion(a,st['imagen']) for a in u['propuesta']]
                    else:elegidas=deepcopy(u['antes'])
                    decisiones.append({'unidad_id':uid,'unidad_origen_sha256':u['huella'],'accion':accion,
                        'anotaciones':elegidas,'inspeccion_explicita':True,'nota':nota,
                        'propuesta_fuente_sha256':r['fuente_sha256']})
            e={'version':VERSION,'proyecto_sha256':self.id,'numero':len(evs)+1,'anterior':evs[-1]['huella'] if evs else None,
               'id_solicitud':sol,'solicitud_sha256':firma,'imagen_id':iid,'persona_id':persona,
               'accion':accion,'decisiones':decisiones,'nota':nota,'registrado_UTC':datetime.now(timezone.utc).isoformat(),
               'version_IA_origen':self.catalogo['version_IA_origen'],'publicacion_IA_sha256':self.catalogo['publicacion_IA_sha256'],
               'revision_humana_base':r['revision_humana_base'],'estado_anterior_sha256':st['estado_sha256'],
               'revision_humana_individual':True,'revision_independiente':False,'aceptada_V26':False,**extra}
            e['huella']=digest(e);publicar(self.path/'eventos'/f"{e['numero']:06d}.json",e)
            return {'evento':e,'reintento_idempotente':False}

    def codigo_qa(self):
        return digest({str(p):sha(p) for p in [Path(__file__),Path(canonizar.__code__.co_filename),Path(celdas.__code__.co_filename)]})

    def auditar(self):
        """Cache por imagen; snapshots encadenados y nunca sobrescritos."""
        with bloqueo(self.path/'auditoria.lock'):
            evs=self.eventos();head=evs[-1]['huella'] if evs else None;codigo=self.codigo_qa();filas=[];reusadas=0
            for iid,r in self.imagenes.items():
                st=self.estado(iid,evs);key=digest({'estado':st['estado_sha256'],'codigo':codigo,'vigente':st['origen_vigente']})
                cp=self.path/'cache_qa'/(key+'.json')
                if cp.exists():
                    cache=leer(cp)
                    if cache['huella']!=digest(cache['qa']):raise ValueError('Caché QA alterada')
                    qa=cache['qa'];reusadas+=1
                else:
                    qa=qa_rle(r['imagen'],st['anotaciones']);publicar(cp,{'huella':digest(qa),'qa':qa})
                lista=[o for o in qa['objetos'] if o['material'] in CLASES]
                propia=bool(st['unidades']);heredada=r.get('candidata_heredada_V283',False)
                confirmados=[a for u in st['unidades'] if u['estado'] in ['aceptar','editar'] for a in u['decision']['anotaciones'] if not a.get('descartada')]
                candidata=bool(st['origen_vigente'] and not qa['errores'] and (st['cobertura_completa'] if propia else heredada))
                filas.append({'imagen':iid,'fuente':r['imagen']['fuente'],'estado_sha256':st['estado_sha256'],
                    'contenido_sha256':st['contenido_sha256'],'origen_vigente':st['origen_vigente'],
                    'unidades':len(st['unidades']),'pendientes':st['pendientes'],
                    'decisiones':dict(Counter(u['estado'] for u in st['unidades'])),
                    'positivos_confirmados_V284':sum(a['material'] in CLASES for a in confirmados),
                    'ignore_confirmados_V284':sum(a['material'] not in CLASES for a in confirmados),
                    'cobertura_humana_V284':st['cobertura_completa'],'candidata_heredada_V283':heredada,
                    'utilizable_exploratoria':candidata,'aceptada_V26':False,
                    'positivos_por_material':dict(Counter(o['material'] for o in lista)),
                    'qa':qa,'objetos_utilizables':len(lista) if candidata else 0,
                    'regiones_48':dict(Counter(o['material'] for o in lista if o['componente_maximo_48']>=2)) if candidata else {},
                    'alimentarias_HPI_48':sum(o['material']=='organico' and o['componente_maximo_48']>=2 for o in lista) if candidata and r['imagen']['fuente']=='hpi' else 0})
            viejos=sorted((self.path/'auditorias').glob('*/auditoria.json'));prev=sha(viejos[-1]) if viejos else None
            carpeta=self.path/'auditorias'/sello();publicar(carpeta/'imagenes.json',filas)
            listas=[f for f in filas if f['utilizable_exploratoria']];dist=Counter();reg=Counter()
            for f in listas:dist.update(f['positivos_por_material']);reg.update(f['regiones_48'])
            estados=Counter()
            for f in filas:estados.update(f['decisiones'])
            resumen={'version':VERSION,'eventos_head':head,'codigo_qa':codigo,'snapshot_previo_sha256':prev,
                'sha256_imagenes':sha(carpeta/'imagenes.json'),'imagenes_utilizables':len(listas),
                'imagenes_nuevas_confirmadas':sum(f['cobertura_humana_V284'] and f['utilizable_exploratoria'] for f in filas),
                'objetos_utilizables':sum(f['objetos_utilizables'] for f in filas),'por_material':dict(dist),'regiones_48':dict(reg),
                'alimentarias_HPI_48':sum(f['alimentarias_HPI_48'] for f in filas),'cache_reusadas':reusadas,
                'decisiones_humanas_V284':len(evs),'aceptadas_V26':0,'entrenamiento_autorizado':False,
                'propuestas_por_estado':dict(estados),
                'positivos_confirmados_en_propuestas_V284':sum(f['positivos_confirmados_V284'] for f in filas),
                'ignore_confirmados_en_propuestas_V284':sum(f['ignore_confirmados_V284'] for f in filas),
                'limitacion':'Candidatura auxiliar de usuario único; no certifica exhaustividad ni independencia física. Rasterizador RLE versionado.'}
            publicar(carpeta/'auditoria.json',resumen);return carpeta/'auditoria.json',resumen

    def ultima_auditoria(self):
        ps=sorted((self.path/'auditorias').glob('*/auditoria.json'));prev=None
        for p in ps:
            q=leer(p)
            if q['snapshot_previo_sha256']!=prev or sha(p.parent/'imagenes.json')!=q['sha256_imagenes']:raise ValueError('Cadena de auditorías alterada')
            prev=sha(p)
        if not ps:return {'vigente':False,'resumen':None}
        q=leer(ps[-1]);evs=self.eventos();filas=leer(ps[-1].parent/'imagenes.json')
        vigente=q['eventos_head']==(evs[-1]['huella'] if evs else None) and q['codigo_qa']==self.codigo_qa()
        vigente=vigente and all(f['origen_vigente']==self.origen_vigente(f['imagen']) for f in filas)
        return {'vigente':vigente,'resumen':q,'archivo':str(ps[-1]),'imagenes':filas}

    def exportar(self,solo_utilizables=True):
        self.verificar_fuentes()
        au=self.ultima_auditoria()
        if not au['vigente']:raise ValueError('Actualizar auditoría antes de exportar')
        permitidas={f['imagen'] for f in au['imagenes'] if f['utilizable_exploratoria'] or not solo_utilizables}
        evs=self.eventos();coco={'info':{'version':VERSION,'estado':'NO_ENTRENAR_AUXILIAR_USUARIO_UNICO',
            'solo_utilizables':solo_utilizables,'aceptadas_V26':0,'revision_independiente':False},
            'images':[],'annotations':[],'categories':[{'id':n+1,'name':c} for n,c in enumerate(MATERIALES)]}
        cats={c:n+1 for n,c in enumerate(MATERIALES)}
        for iid in sorted(permitidas):
            st=self.estado(iid,evs);im=st['imagen'];imid=len(coco['images'])+1
            coco['images'].append({'id':imid,'file_name':im['archivo'],'width':im['width'],'height':im['height'],
                'eco_imagen_id':iid,'eco_procedencia':im,'estado_sha256':st['estado_sha256'],
                'cobertura_humana_V284':st['cobertura_completa'],'aceptada_V26':False})
            humanas={a['id'] for u in st['unidades'] if u['estado'] in RESUELTAS for a in (u.get('decision') or {}).get('anotaciones',[])}
            for a in st['anotaciones']:
                if a.get('descartada'):continue
                b=validar_anotacion(a,im)
                coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':imid,'category_id':cats[b['material']],
                    'segmentation':deepcopy(b['segmentacion_canonica_RLE']),'bbox':b['bbox'],'area':b['area_px'],
                    'iscrowd':0,'ignore':int(b['material'] not in CLASES),'eco_objeto_id':b['id'],
                    'eco':{'decision_humana_V284':b['id'] in humanas,'revision_V284':b.get('revision_V284'),
                        'base_humana_anterior':self.imagenes[iid]['base_completa'] and b['id'] in {a['id'] for a in self.imagenes[iid]['base']},
                        'aceptada_V26':False,'revision_independiente':False,'motivo':b.get('motivo',''),
                        'proyecto_sha256':self.id,'eventos_head':evs[-1]['huella'] if evs else None}})
        carpeta=self.path/'exportaciones'/sello();publicar(carpeta/'annotations.json',coco)
        publicar(carpeta/'manifiesto.json',{'version':VERSION,'COCO_sha256':sha(carpeta/'annotations.json'),
            'auditoria_sha256':sha(au['archivo']),'proyecto_sha256':self.id,'solo_utilizables':solo_utilizables,
            'entrenamiento_autorizado':False,'aceptadas_V26':0,'imagenes':len(coco['images']),'instancias_y_ignore':len(coco['annotations'])})
        return carpeta,coco

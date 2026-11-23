"""Incidencias y QA incremental: sin modificar contornos, entrenar o aceptar V26."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import threading
import uuid
from ecowaste.incorporacion import leer,publicar,sha,digest,bloqueo,CLASES,identificador
from ecowaste.auxiliares_publicos import normalizar
from ecowaste.validacion_anotaciones import analizar_imagen,revision_humana_final

VERSION='correcciones_prioritarias_v28_3_1'
RAIZ=Path(__file__).resolve().parents[1]
HPI=['hpi:'+n for n in ['0','132','155','78','96','192']]
VIDRIO=['warp:POSAD_1_13-Sep_02-24-53','warp:POSAD_1_13-Sep_03-20-46',
    'warp:Robo_25-Mar_14-39-46','warp:prepared_data_all_MGS_20-Oct_01-03-59']
PAPEL=['zerowaste:'+n for n in ['1001','1076','1709','2670','2688','2706','2709','2710','2843','331','588','896']]
AVISOS_MECANICOS={'fragmentacion_o_objetos_fusionados_requiere_inspeccion',
    'solape_visible_requiere_decision_no_prueba_de_occlusion','relleno_paridad_editor_difiere_union_exportador',
    'bbox_no_corresponde_a_poligonos','mascaras_identicas_instancias_duplicadas_o_conflicto_material',
    'material_sin_asignar','mascara_vacia','sin_contorno'}


def aviso_ya_no_aplica(f,qa):
    """Solo predicados mecánicos: jamás inferir que una omisión visual se corrigió."""
    if f['tipo'] not in ['visual','error_bloqueante','qa_incremental']:return False
    motivos=set(f['motivo'].split('; '))
    if not motivos or not motivos<=AVISOS_MECANICOS:return False
    actuales={m for a in [*qa['errores'],*qa['avisos_visuales']]
        if set(ids(a.get('objetos',a.get('objeto'))))==set(f['objetos']) for m in a['motivos']}
    return not bool(motivos&actuales)


def prioridad_vigente(r):
    # Conservar catálogos históricos; corregir la navegación derivada sin tocar sus bytes.
    return 4 if r['imagen'] in PAPEL and r['prioridad'] not in [1,2,3,5,7] else r['prioridad']


def ids(x):
    if x is None:return []
    if isinstance(x,str):return [x]
    return sorted({a for y in x for a in ids(y)})


def auditorias_verificadas(salida):
    ps=sorted(Path(salida).glob('*/auditoria.json'));previo=None
    for p in ps:
        s=leer(p)
        if s['snapshot_previo_sha256']!=previo:raise ValueError('Cadena de auditorías incompleta o alterada')
        for nombre,h in s['sha256_archivos'].items():
            if Path(nombre).name!=nombre or sha(p.parent/nombre)!=h:raise ValueError('Archivo de auditoría alterado: '+nombre)
        previo=sha(p)
    return ps


def catalogo_historico(raiz=RAIZ):
    raiz=Path(raiz);s=raiz/'resultados/v28_validacion_anotaciones';filas=[]
    def agregar(iid,objetos,motivo,tipo,origen,bloquea=True):
        objetos=ids(objetos);p=1 if iid in HPI else 2 if iid in VIDRIO else 3 if any(t in motivo for t in ['milk-cardboard','juice-cardboard','multicapa']) else 4 if iid.startswith('zerowaste:') and 'contorno' in motivo else 5 if tipo=='guardado' else 6
        if not bloquea:p=7
        r={'imagen':iid,'objetos':objetos,'motivo':motivo,'tipo':tipo,'origen':origen,
            'prioridad':p,'bloqueante':bloquea,'posible_omision':not objetos and tipo!='guardado' or any(t in motivo.lower() for t in ['sin contorno','sin anot','omiti','cobertura','granos'])}
        r['id']='inc_'+digest({'imagen':iid,'objetos':objetos,'motivo':motivo,'tipo':tipo})[:24]
        if not any(f['id']==r['id'] for f in filas):filas.append(r)
    for o in leer(s/'pendientes.json'):
        agregar(o['imagen'],o.get('objeto'),'; '.join(o['motivos']),o['nivel'],'V28_QA')
    for o in leer(s/'observaciones_visuales_y_taxonomia.json'):
        agregar(o['imagen'],o.get('objetos'),o['motivo'],'decision_visual','V28_observacion_asistente_no_revision_independiente')
    c=leer(raiz/'datos/nuevos/v28_2/coleccion/annotations.json');cats={x['id']:x['name'] for x in c['categories']}
    for im in c['images']:
        for a in c['annotations']:
            material=cats[a['category_id']]
            if a['image_id']==im['id'] and material in ['vidrio','metal']:
                agregar(im['eco']['imagen_id_auxiliar'],a['eco']['id_auxiliar'],
                    f'V28.2: AP50 auxiliar de {material}=0. Comprobar material y contorno; un fallo del modelo no demuestra error de anotación.',
                    'diagnostico_modelo','V28.2',False)
    for iid in ['hpi:40','hpi:66']:
        agregar(iid,None,'V28.2: comprobar separación de restos, fondo del plato y granularidad de migas. Sugerencia diagnóstica, no error GT demostrado.',
            'diagnostico_modelo','V28.2',False)
    return sorted(filas,key=lambda f:(f['prioridad'],HPI.index(f['imagen']) if f['imagen'] in HPI else 99,f['imagen'],f['id']))


class GestionCorrecciones:
    def __init__(self,taller,carpeta,catalogo=None):
        self.t=taller;self.path=Path(carpeta);self.lock=threading.RLock()
        if catalogo is not None:
            c={'version':VERSION,'proyecto_sha256':taller.base['huella'],'incidencias':catalogo,
                'revisiones_base':{i:(revision_humana_final(taller.diario(i)) or {}).get('huella') for i in taller.imagenes}}
            if (self.path/'catalogo.json').exists():
                if leer(self.path/'catalogo.json')!=c:raise ValueError('No sustituir catálogo existente')
            else:publicar(self.path/'catalogo.json',c)
        self.catalogo=leer(self.path/'catalogo.json')
        if self.catalogo['version']!=VERSION or self.catalogo['proyecto_sha256']!=taller.base['huella']:raise ValueError('Catálogo de otro proyecto')
        self.porid={r['id']:r for r in self.catalogo['incidencias']}

    def todas(self):
        detectadas=[leer(p) for p in sorted((self.path/'detectadas').glob('*.json'))]
        self.porid={r['id']:{**r,'prioridad':prioridad_vigente(r)} for r in [*self.catalogo['incidencias'],*detectadas]}
        return sorted(self.porid.values(),key=lambda r:(r['prioridad'],HPI.index(r['imagen']) if r['imagen'] in HPI else 99,r['imagen'],r['id']))

    def firma_ambito(self,f,ev):
        if not ev:return None
        anns=[a for a in ev['anotaciones'] if not f['objetos'] or a['id'] in f['objetos']]
        return digest({'cobertura_completa':ev['cobertura_completa'],'anotaciones':sorted(anns,key=lambda a:a['id'])})

    def codigo_qa(self):
        return digest({p.name:sha(p) for p in [Path(__file__),Path(analizar_imagen.__code__.co_filename),
            *[Path(__file__).parent/n for n in ['auxiliares_publicos.py','incorporacion.py','poligonos.py','datasets.py','supervision_regional.py']]]})

    def eventos(self):
        salida=[];anterior=None
        for n,p in enumerate(sorted((self.path/'incidencias').glob('*.json')),1):
            e=leer(p)
            if p.name!=f'{n:06d}.json' or e['numero']!=n or e['anterior']!=anterior or e['huella']!=digest({k:v for k,v in e.items() if k!='huella'}) or e['proyecto_sha256']!=self.t.base['huella']:
                raise ValueError('Diario de incidencias alterado')
            salida.append(e);anterior=sha(p)
        return salida

    def listar(self,imagenes=None):
        es=self.eventos();ultimos={e['incidencia_id']:e for e in es};revs={};filas=[];codigo=None;qas={}
        for f in self.todas():
            iid=f['imagen']
            if imagenes is not None and iid not in imagenes:continue
            if iid not in revs:
                revs[iid]=revision_humana_final(self.t.diario(iid))
            ev=revs[iid];h=(ev or {}).get('huella');declarada=ultimos.get(f['id'])
            vigente=bool(declarada and (declarada['revision_huella']==h or declarada.get('firma_ambito')==self.firma_ambito(f,ev)))
            estado=declarada['estado'] if vigente else 'pendiente'
            verificacion=None
            if f['tipo']=='guardado' and ev and ev['cobertura_completa'] and h!=self.catalogo['revisiones_base'].get(iid):
                estado='corregida';verificacion='guardado_completo_humano_real_observado'
            if estado=='pendiente' and not vigente and ev and ev['cobertura_completa'] and h!=self.catalogo['revisiones_base'].get(iid):
                if codigo is None:codigo=self.codigo_qa()
                if iid not in qas:
                    im=self.t.imagen(iid);clave=digest({'imagen_sha256':im['sha256_archivo'],'revision':h,'codigo_qa':codigo,'proyecto':self.t.base['huella']})
                    p=self.path/'cache_qa'/(clave+'.json');qas[iid]=None
                    if p.exists():
                        registro=leer(p)
                        if registro['clave']!=clave or registro['huella']!=digest({k:v for k,v in registro.items() if k!='huella'}):raise ValueError('Caché QA alterada')
                        qas[iid]=registro['qa']
                if qas[iid] is not None and aviso_ya_no_aplica(f,qas[iid]):
                    estado='no_vigente';verificacion='QA_actual_no_reproduce_aviso_tecnico; no_es_revision_humana'
            anns=(ev or {}).get('anotaciones',self.t.imagen(iid)['anotaciones'])
            afectados=[{'id':a['id'],'material':a['material'],'contornos':len(a.get('poligonos',[])),
                'descartada':a.get('descartada',False)} for a in anns if not f['objetos'] or a['id'] in f['objetos']]
            filas.append({**f,'estado':estado,'resolucion':declarada if vigente else None,
                'resolucion_anterior_desactualizada':bool(declarada and not vigente),'verificacion':verificacion,
                'revision_huella':h,'estado_guardado':'sin_revision' if not ev else 'completa' if ev['cobertura_completa'] else 'borrador',
                'objetos_actuales':afectados})
        return filas

    def resolver(self,d):
        self.todas()
        f=self.porid[d['incidencia_id']];estado=d['estado'];persona=identificador(d['persona_id'])
        if estado not in ['pendiente','corregida','requiere_decision']:raise ValueError('Estado de incidencia inválido')
        nota=str(d.get('nota','')).strip()
        if len(nota)<5:raise ValueError('Explica la corrección o decisión pendiente (mínimo cinco caracteres)')
        with self.lock,bloqueo(self.path/'gestion.lock'):
            ev=revision_humana_final(self.t.diario(f['imagen']))
            if not ev or d.get('revision_huella')!=ev['huella']:raise ValueError('Guardar primero y usar la revisión actual de la imagen')
            if estado=='corregida':
                if not ev['cobertura_completa'] or persona!=ev['persona_id']:raise ValueError('Corregida requiere guardado completo del mismo usuario; no es revisión independiente')
                if ev['huella']==self.catalogo['revisiones_base'].get(f['imagen']):raise ValueError('Primero guardar la comprobación/corrección; no cerrar sobre la instantánea antigua')
                q=analizar_imagen(self.t.imagen(f['imagen']),ev['anotaciones'],True)
                if q['errores']:raise ValueError('Quedan errores técnicos: '+str(q['errores']))
            es=self.eventos();e={'version':VERSION,'proyecto_sha256':self.t.base['huella'],'numero':len(es)+1,
                'anterior':sha(self.path/'incidencias'/f'{len(es):06d}.json') if es else None,
                'incidencia_id':f['id'],'imagen':f['imagen'],'persona_id':persona,'estado':estado,'nota':nota,
                'revision_huella':ev['huella'],'firma_ambito':self.firma_ambito(f,ev),'registrado_UTC':datetime.now(timezone.utc).isoformat(),
                'aceptada_V26':False,'revision_independiente':False,'entrenamiento_autorizado':False}
            e['huella']=digest(e);publicar(self.path/'incidencias'/f'{len(es)+1:06d}.json',e)
        return e

    def borrador(self,d):
        iid=d['imagen_id'];self.t.imagen(iid)
        if d.get('proyecto_sha256')!=self.t.base['huella']:raise ValueError('Borrador de otro proyecto')
        # Deliberadamente sin normalizar: conservar incluso una geometría inválida.
        e={'version':VERSION,'tipo':'borrador_emergencia_NO_revision_humana','contenido':d,
            'registrado_UTC':datetime.now(timezone.utc).isoformat(),'aceptada_V26':False,'entrenamiento_autorizado':False}
        e['huella']=digest(e);p=self.path/'borradores'/digest(iid)[:24]/(uuid.uuid4().hex+'.json');publicar(p,e)
        return {'archivo':p.relative_to(self.path).as_posix(),'sha256':sha(p),'huella':e['huella']}

    def validar_guardado(self,d):
        im=self.t.imagen(d['imagen_id']);anns=d.get('anotaciones')
        if not isinstance(anns,list):raise ValueError('Lista de anotaciones requerida')
        for j,a in enumerate(anns,1):
            if not isinstance(a,dict):raise ValueError(f'Objeto {j}: se requiere objeto JSON con ID, material y contornos')
            try:normalizar(a,im,completa=d.get('cobertura_completa') is True)
            except (ValueError,KeyError,TypeError) as e:raise ValueError(f'Objeto {j} ({a.get("id","sin ID")}, {a.get("material","sin material")}): {e}') from e

    def guardar(self,d,vista):
        sid=identificador(d['id_solicitud']);p=self.path/'recibos'/(sid+'.json')
        # Token de vista excluido: un recibo se puede consultar tras reiniciar el servidor.
        hd=digest({k:v for k,v in d.items() if k!='token_vista'})
        with self.lock:
            if p.exists():
                r=leer(p)
                if r['solicitud_sha256']!=hd:raise ValueError('ID de solicitud reutilizado con otro contenido')
                return {**r,'reintento_idempotente':True}
            if not vista or vista['id']!=d['imagen_id']:raise ValueError('Vista inválida o servidor reiniciado; conservar borrador y recargar contexto')
            if d.get('revision')!=vista['revision']:raise ValueError('La solicitud no corresponde a la versión visualizada')
            self.validar_guardado(d)
            intento=self.path/'solicitudes'/(sid+'.json')
            if intento.exists():
                i=leer(intento)
                if i['solicitud_sha256']!=hd:raise ValueError('Solicitud pendiente con distinto contenido')
                if len(self.t.diario(d['imagen_id']))!=d['revision']:raise ValueError('Resultado incierto de un guardado anterior: comprobar diario; no duplicar automáticamente')
            else:publicar(intento,{'solicitud_sha256':hd,'imagen_id':d['imagen_id'],'revision':d['revision']})
            self.t.guardar(d['imagen_id'],d,modo=vista['modo'],revision_base=vista['base'])
            ev=self.t.diario(d['imagen_id'])[-1]
            r={'guardado':True,'solicitud_sha256':hd,'id_solicitud':sid,'imagen_id':d['imagen_id'],'modo':ev['modo'],
                'revision_guardada':ev['numero'],'revision_huella':ev['huella'],'aceptada_V26':False}
            publicar(p,r);return r

    def auditar(self,salida):
        with bloqueo(self.path/'auditoria.lock'):
            return self._auditar(salida)

    def _auditar(self,salida):
        """Caché por última revisión y código QA; snapshots inmutables y encadenados."""
        salida=Path(salida);codigo=self.codigo_qa()
        estados={i:revision_humana_final(self.t.diario(i)) for i in self.t.imagenes}
        eventos_hash=digest(self.eventos());incidentes=self.listar();filas=[];reusadas=0;conteos=Counter();utiles=Counter();comida=0
        for iid,ev in estados.items():
            im=self.t.imagen(iid);imrel=im['archivo'];self.t.archivo_imagen(iid)
            clave=digest({'imagen_sha256':im['sha256_archivo'],'revision':(ev or {}).get('huella'),
                'codigo_qa':codigo,'proyecto':self.t.base['huella']})
            cache=self.path/'cache_qa'/(clave+'.json')
            if cache.exists():
                registro=leer(cache)
                if registro['clave']!=clave or registro['huella']!=digest({k:v for k,v in registro.items() if k!='huella'}):raise ValueError('Caché QA alterada')
                q=registro['qa'];reusadas+=1
            else:
                q=analizar_imagen(im,ev['anotaciones'] if ev else [],bool(ev and ev['cobertura_completa']))
                registro={'clave':clave,'qa':q};registro['huella']=digest(registro);publicar(cache,registro)
            incidentes=[r for r in incidentes if r['imagen']!=iid]+self.listar({iid})
            # Registrar avisos nuevos como incidencias navegables; nunca inventar su resolución.
            for aviso in [*q['avisos_visuales'],*q['errores']]:
                objs=ids(aviso.get('objetos',aviso.get('objeto')))
                for motivo in aviso['motivos']:
                    conocidas=[r for r in incidentes if r['imagen']==iid and set(r['objetos'])==set(objs) and motivo in r['motivo']]
                    if not conocidas:
                        r={'imagen':iid,'objetos':objs,'motivo':motivo,'tipo':'qa_incremental','origen':'QA_automatica_no_revision_humana',
                            'prioridad':1 if iid in HPI else 2 if iid in VIDRIO else 6,'bloqueante':True,'posible_omision':False}
                        r['id']='inc_'+digest({'imagen':iid,'objetos':objs,'motivo':motivo,'tipo':r['tipo']})[:24]
                        p=self.path/'detectadas'/(r['id']+'.json')
                        if not p.exists():publicar(p,r)
                        incidentes=self.listar()
            incs=[r for r in incidentes if r['imagen']==iid and r['bloqueante'] and r['estado'] not in ['corregida','no_vigente']]
            # Un aviso nuevo no se puede ocultar cerrando una observación antigua diferente.
            nuevas=[]
            for aviso in q['avisos_visuales']:
                objs=ids(aviso.get('objetos',aviso.get('objeto')))
                for motivo in aviso['motivos']:
                    compatibles=[r for r in incidentes if r['imagen']==iid and r['estado']=='corregida' and
                        set(r['objetos'])==set(objs) and motivo in r['motivo']]
                    if not compatibles:nuevas.append({'objetos':objs,'motivo':motivo})
            lista=bool(ev and ev['cobertura_completa'] and not q['errores'] and not incs and not nuevas)
            rs=[a for a in q['objetos'] if not a['descartada'] and a['material'] in CLASES and a['valido_geometria']]
            if lista:
                conteos.update(a['material'] for a in rs);utiles.update(a['material'] for a in rs if a['componente_maximo_48']>=2)
                if im['fuente']=='hpi':comida+=sum(a['organico_utilizable_48'] for a in rs)
            filas.append({'imagen':iid,'fuente':im['fuente'],'archivo':imrel,'revision_huella':(ev or {}).get('huella'),
                'guardado':'sin_revision' if not ev else 'completa' if ev['cobertura_completa'] else 'borrador',
                'candidata_exploratoria':lista,'aceptada_V26':False,'errores':q['errores'],
                'incidencias_pendientes':[r['id'] for r in incs],'avisos_nuevos':nuevas,'qa':q})
        # Si hubo edición simultánea, el snapshot no se publica como vigente.
        if any((revision_humana_final(self.t.diario(i)) or {}).get('huella')!=(ev or {}).get('huella') for i,ev in estados.items()):
            raise ValueError('Cambió una revisión durante QA: repetir incremental; caché y borradores conservados')
        if digest(self.eventos())!=eventos_hash:raise ValueError('Cambió una resolución durante QA: repetir incremental')
        anteriores=auditorias_verificadas(salida)
        marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8];dest=salida/marca
        publicar(dest/'imagenes.json',filas);publicar(dest/'incidencias.json',incidentes)
        n=sum(f['candidata_exploratoria'] for f in filas)
        faltan={c:max(0,8-utiles[c]) for c in CLASES}
        resumen={'version':VERSION,'proyecto_sha256':self.t.base['huella'],'fecha_UTC':datetime.now(timezone.utc).isoformat(),
            'snapshot_previo_sha256':sha(anteriores[-1]) if anteriores else None,'imagenes_totales':len(filas),
            'imagenes_candidatas':n,'anotaciones_validas_candidatas':sum(conteos.values()),
            'materiales_poligonos_validos':{c:conteos[c] for c in CLASES},'materiales_regiones_48':{c:utiles[c] for c in CLASES},
            'regiones_alimentarias_HPI_utilizables':comida,'faltan_imagenes_hasta40':max(0,40-n),
            'faltan_regiones_hasta8_por_material':faltan,'faltan_alimentarias_hasta8':max(0,8-comida),
            'cumple_minimos_numericos_exploratorios':n>=40 and not any(faltan.values()) and comida>=8,
            'revisiones_humanas':sum(len(self.t.diario(i)) for i in self.t.imagenes),'cache_reusadas':reusadas,
            'cache_recalculadas':len(filas)-reusadas,'aceptadas_V26':0,'entrenamiento_autorizado':False,
            'exhaustividad_no_certificada':True,'test_evaluado':False,'generalizacion_desbloqueada':False,
            'sha256_archivos':{p.name:sha(p) for p in dest.iterdir() if p.is_file()},'codigo_qa':codigo}
        publicar(dest/'auditoria.json',resumen);return dest/'auditoria.json',resumen

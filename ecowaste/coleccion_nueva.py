"""Colecciones V26 versionadas: grupos, cuotas reales y exportaciones bloqueadas."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import math

from ecowaste.incorporacion import (VERSION,RAIZ,POLITICA,CLASES,IGNORADAS,cargar,leer,sha,digest,politica,
    candidatos_duplicados,referencias_historicas,qa_geometria,publicar,publicar_bytes,ruta_segura,destino_nuevo,copiar_imagen_verificada)
from ecowaste.poligonos import rasterizar_contornos
from ecowaste.revision_datos import estado_revision

PROTOCOLO=RAIZ/'configuracion/protocolo_generalizacion_v26.json'


def consolidar(paquetes):
    imgs=[];fuentes=[];objetos={};tipos=set();pixeles=set();nombres=set()
    for path in sorted(map(Path,paquetes),key=lambda p:str(p.resolve())):
        c,estados,evs=estado_revision(path);cid=c['coleccion_id'];tipos.add(c['tipo_datos'])
        if cid in nombres:raise ValueError('Colección repetida entre lotes')
        nombres.add(cid)
        for oid,obj in c['objetos_fisicos'].items():
            if oid in objetos and objetos[oid]!=obj:raise ValueError('Identidad física global contradictoria: '+oid)
            objetos[oid]=obj
        fuentes.append({'paquete':str(path.resolve()),'coleccion_id':cid,'tipo_datos':c['tipo_datos'],
            'sha256_importado':sha(path/'IMPORTADO.json'),'sha256_coleccion':digest(c),
            'coleccion_importada':c,'entrada_original_utf8':(path/'entrada_annotations.json').read_bytes().decode('utf-8'),
            'importado':leer(path/'IMPORTADO.json'),
            'revisiones':evs,'revisores':c['revisores'],'registro_objetos':c['objetos_fisicos'],
            'sha256_historicos':leer(path/'IMPORTADO.json')['sha256_historicos']})
        for im,est in zip(c['imagenes'],estados):
            if im['sha256_pixeles'] in pixeles:raise ValueError('Duplicados exactos entre paquetes')
            pixeles.add(im['sha256_pixeles'])
            imgs.append({**im,'id':cid+':'+im['id'],'imagen_id_origen':im['id'],'coleccion_id':cid,
                'estado':est['estado'],'tipo_datos':c['tipo_datos'],'instancias_propuestas':len(im['instancias']),
                'instancias':est['instancias_finales'],'objetos_para_grupos':sorted(set(im['objetos_fisicos_ids'])|{a['objeto_fisico_id'] for a in [*im['instancias'],*[a for ev in evs if ev['imagen_id']==im['id'] for a in ev['instancias']]]}),
                'sha256_adjudicacion':est['huella_adjudicacion'],'archivo_origen':str((path/im['file_name']).resolve())})
    if len(tipos)>1:raise ValueError('No mezclar lotes reales y fixtures')
    imgs.sort(key=lambda im:im['id'])
    conflictos=[];materiales=defaultdict(set)
    for im in imgs:
        for a in im['instancias']:materiales[a['objeto_fisico_id']].add(a['material'])
    for oid,cs in materiales.items():
        if len(cs)>1:conflictos.append({'objeto_fisico_id':oid,'materiales':sorted(cs)})
    return {'version':VERSION,'tipo_datos':next(iter(tipos)) if tipos else 'real','imagenes':imgs,
        'fuentes':fuentes,'objetos_fisicos':objetos,'conflictos_material_fisico':conflictos,
        'sha256_politica':sha(POLITICA)}


def particionar(cohorte):
    p=politica()['particion'];imgs=cohorte['imagenes'];parent={im['id']:im['id'] for im in imgs}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def unir(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    vistos={};motivos=[]
    for im in imgs:
        keys=[(k,im[k]) for k in ['sitio_id','sesion_id','escena_id','grupo_captura_id','secuencia_id'] if im.get(k)]
        keys.extend(('objeto_fisico_id',oid) for oid in im['objetos_para_grupos'])
        for key in keys:
            if key in vistos:unir(im['id'],vistos[key]);motivos.append({'a':im['id'],'b':vistos[key],'por':key[0]})
            else: vistos[key]=im['id']
    dups=candidatos_duplicados(imgs,[])['nuevos']
    for d in dups:unir(d['a'],d['b']);motivos.append({'a':d['a'],'b':d['b'],'por':'duplicado_perceptual_candidato'})
    grupos=defaultdict(list)
    for im in imgs:grupos[find(im['id'])].append(im)
    gs=[];asignacion={}
    for miembros in grupos.values():
        ids=sorted(im['id'] for im in miembros);gid=digest(ids)
        fijos={im['destino_previsto'] for im in miembros if im['destino_previsto']!='sin_asignar'}
        if len(fijos)>1:raise ValueError('Grupos conectados ya reservados para destinos opuestos; no separar fotogramas/objetos/sitios')
        gs.append({'grupo':gid,'imagenes':ids,'escenas':len({im['escena_id'] for im in miembros}),
            'destino':next(iter(fijos)) if fijos else None})
    gs.sort(key=lambda g:digest({'semilla':p['semilla'],'grupo':g['grupo']}))
    objetivo=math.ceil(p['fraccion_generalizacion']*len({im['escena_id'] for im in imgs}));gen=sum(g['escenas'] for g in gs if g['destino']=='generalizacion_bloqueada')
    for g in gs:
        if g['destino'] is None:
            g['destino']='generalizacion_bloqueada' if gen<objetivo and len(gs)>1 else 'train_nuevo'
            if g['destino']=='generalizacion_bloqueada':gen+=g['escenas']
        for iid in g['imagenes']:asignacion[iid]=g['destino']
    # Si la unión reduce todo a un componente, no existe evaluación independiente.
    return {'version':VERSION,'sha256_cohorte':digest(cohorte),'sha256_politica':sha(POLITICA),
        'semilla':p['semilla'],'grupos':sorted(gs,key=lambda g:g['grupo']),'asignacion':dict(sorted(asignacion.items())),
        'uniones':motivos,'duplicados_candidatos':dups,'evaluacion_autorizada':False}


def verificar_plan(cohorte,plan):
    esperado=particionar(cohorte)
    if plan!=esperado:raise ValueError('Plan de otra versión, alterado o con fuga de grupos')


def inventario(cohorte,plan=None):
    if plan is None:plan=particionar(cohorte)
    verificar_plan(cohorte,plan);p=politica();salidas={}
    for destino in ['train_nuevo','generalizacion_bloqueada']:
        imgs=[im for im in cohorte['imagenes'] if plan['asignacion'][im['id']]==destino];aceptadas=[im for im in imgs if im['estado']=='aceptada']
        por_material={};qa={im['id']:qa_geometria(im) for im in aceptadas}
        for c in CLASES:
            ff=[(im,a) for im in aceptadas for a in im['instancias'] if a['material']==c]
            por_escena=defaultdict(set);fisicos=set();org_por_objeto=defaultdict(set)
            for im,a in ff:
                por_escena[im['escena_id']].add(a['objeto_fisico_id']);fisicos.add(a['objeto_fisico_id'])
                calidad=next(x for x in qa[im['id']]['instancias'] if x['id']==a['id'])
                if calidad['organico_utilizable_48']:org_por_objeto[a['objeto_fisico_id']].add(im['escena_id'])
            computables=sum(min(p['maximo_por_material_y_escena_para_cuota'],len(ids)) for ids in por_escena.values())
            # Una misma región de un objeto en la misma escena/video cuenta una vez.
            org_escenas=defaultdict(set)
            for oid,escenas in org_por_objeto.items():
                for scene in sorted(escenas)[:p['organicos_utilizables_maximo_por_objeto']]:org_escenas[scene].add(oid)
            org_util=sum(min(p['maximo_por_material_y_escena_para_cuota'],len(ids)) for ids in org_escenas.values())
            target=p['cuotas_train'][c] if destino=='train_nuevo' else p['generalizacion']['instancias_por_clase']
            por_material[c]={'instancias_adjudicadas':len(ff),'instancias_computables_cuota':computables,'escenas':len(por_escena),
                'objetos_fisicos_distintos':len(fisicos),'objetivo':target,'pendientes':max(0,target-computables),
                'organicos_utilizables_48':org_util if c=='organico' else None}
        salidas[destino]={'imagenes_recibidas':len(imgs),'escenas_recibidas':len({im['escena_id'] for im in imgs}),
            'imagenes_adjudicadas':len(aceptadas),'escenas_adjudicadas':len({im['escena_id'] for im in aceptadas}),
            'por_material':por_material,'estados':dict(Counter(im['estado'] for im in imgs)),
            'datos_reales_computables':cohorte['tipo_datos']=='real'}
    # Un ensayo puede probar la mecánica, nunca acreditar las cuotas reales.
    if cohorte['tipo_datos']!='real':
        for s in salidas.values():
            for c,x in s['por_material'].items():
                x['instancias_fixture']=x['instancias_computables_cuota'];x['instancias_computables_cuota']=0
                x['pendientes']=x['objetivo'];x['organicos_utilizables_48']=0 if c=='organico' else None
    problemas=[]
    if not cohorte['imagenes']:problemas.append('coleccion_sin_imagenes_reales')
    if cohorte['tipo_datos']!='real':problemas.append('fixtures_no_congelables_como_datos_reales')
    if any(im['estado']!='aceptada' for im in cohorte['imagenes']):problemas.append('revisiones_humanas_pendientes')
    if cohorte['conflictos_material_fisico']:problemas.append('material_inconsistente_del_mismo_objeto')
    train=salidas['train_nuevo'];gen=salidas['generalizacion_bloqueada']
    for c,x in train['por_material'].items():
        if x['pendientes']:problemas.append('cuota_train_'+c)
        if c in p['doble_anotacion_obligatoria']:
            if x['escenas']<p['escenas_prioritarias_minimas']:problemas.append('diversidad_escenas_'+c)
            if x['objetos_fisicos_distintos']<p['objetos_fisicos_prioritarios_minimos']:problemas.append('diversidad_fisica_'+c)
    if train['escenas_adjudicadas']<p['escenas_train_minimas']:problemas.append('escenas_train_insuficientes')
    if train['por_material']['organico']['organicos_utilizables_48']<p['organicos_utilizables_minimos']:problemas.append('menos_300_organicos_utilizables_48')
    if gen['escenas_adjudicadas']<p['generalizacion']['escenas_minimas']:problemas.append('escenas_generalizacion_insuficientes')
    for c,x in gen['por_material'].items():
        if x['pendientes'] or x['escenas']<p['generalizacion']['escenas_por_clase']:problemas.append('generalizacion_soporte_'+c)
    return {'version':VERSION,'tipo_datos':cohorte['tipo_datos'],'splits':salidas,'bloqueos':problemas,
        'coleccion_suficiente_y_revisada':not problemas,'entrenamiento_autorizado':False,'evaluacion_autorizada':False,
        'sha256_cohorte':digest(cohorte),'sha256_plan':digest(plan),'test_original_evaluado':False}


def exportar_taco(imgs,split='train_nuevo'):
    if split not in ['train_nuevo','generalizacion_bloqueada']:raise ValueError('Split no autorizado')
    categories=[{'id':i+1,'name':c,'supercategory':'material'} for i,c in enumerate([*CLASES,*sorted(IGNORADAS)])]
    cat={c['name']:c['id'] for c in categories};images=[];anns=[];records=[]
    for i,im in enumerate(sorted(imgs,key=lambda x:x['id'])):
        images.append({'id':i,'file_name':im['file_name'],'width':im['width'],'height':im['height'],
            'eco':{k:im[k] for k in ['id','escena_id','sitio_id','sesion_id','grupo_captura_id','secuencia_id','dispositivo_id','procedencia','capturado_en','sha256_archivo','sha256_pixeles']}})
        cajas=[]
        for a in sorted(im['instancias'],key=lambda a:a['objeto_fisico_id']):
            aid=len(anns)+1;label=a['material'] if a['material'] in CLASES else None
            ann={'id':aid,'image_id':i,'category_id':cat[a['material']],'segmentation':a['poligonos'],'bbox':a['bbox'],
                'area':a['area_mask_px'],'iscrowd':0,'eco':{'objeto_fisico_id':a['objeto_fisico_id'],'origen':'humana_adjudicada',
                    'visible_only':True,'oclusion':a['oclusion'],'truncado':a['truncado'],'transparente':a['transparente'],
                    'motivo_ambiguo':a['motivo_ambiguo'],'solapa_con':a['solapa_con'],'id_anotacion_previo':a['id']}}
            anns.append(ann);cajas.append({'id_original':aid,'id_interno':aid,'bbox':a['bbox'],'bbox_original':a['bbox'],
                'clase':label,'categoria_original':a['material'],'iscrowd':0,'objeto_fisico_id':a['objeto_fisico_id']})
        records.append({'id':im['id'],'fuente':'ecowaste_real_v26','id_original':i,'ruta':im['file_name'],
            'ancho':im['width'],'alto':im['height'],'sha256_archivo':im['sha256_archivo'],'sha256_pixeles':im['sha256_pixeles'],
            'split':split,'grupo':im['grupo_captura_id'],'grupo_adquisicion':im['sesion_id'],'cajas':cajas})
    return {'info':{'version':VERSION,'area':'pixeles_union_visible_rasterizada_no_suma_de_poligonos'},'images':images,'annotations':anns,'categories':categories},records


def congelar(cohorte,plan,destino,*,ensayo=False):
    destino=destino_nuevo(destino)
    if cohorte!=consolidar([f['paquete'] for f in cohorte['fuentes']]):raise ValueError('La cohorte no reproduce sus fuentes y diarios')
    verificar_plan(cohorte,plan);estado=inventario(cohorte,plan)
    if cohorte['conflictos_material_fisico']:raise ValueError('El mismo objeto físico cambia de material; adjudicar antes de exportar')
    if ensayo:
        if cohorte['tipo_datos']!='fixture' or any(im['estado']!='aceptada' for im in cohorte['imagenes']):raise ValueError('El ensayo solo admite fixtures revisadas')
    else:
        if not estado['coleccion_suficiente_y_revisada']:raise ValueError('No congelar: '+','.join(estado['bloqueos']))
        actual=digest(referencias_historicas())
        if any(f['sha256_historicos']!=actual for f in cohorte['fuentes']):raise ValueError('Registro histórico de duplicados incompatible')
    if not cohorte['imagenes']:raise ValueError('No publicar colección vacía')
    protocolo=leer(PROTOCOLO)
    if protocolo['evaluacion_autorizada'] or protocolo['entrenamiento_autorizado'] or protocolo['version']!='generalizacion_v26_1':raise ValueError('Protocolo no bloqueado')
    destino.mkdir(parents=True)
    for im in cohorte['imagenes']:
        path=Path(im['archivo_origen'])
        if sha(path)!=im['sha256_archivo']:raise ValueError('Imagen cambió después de consolidar')
        copiar_imagen_verificada(path,destino/im['file_name'],im['sha256_archivo'])
    train=[im for im in cohorte['imagenes'] if plan['asignacion'][im['id']]=='train_nuevo']
    gen=[im for im in cohorte['imagenes'] if plan['asignacion'][im['id']]=='generalizacion_bloqueada']
    for split,imgs in [('train_nuevo',train),('generalizacion_bloqueada',gen)]:
        taco,records=exportar_taco(imgs,split=split)
        publicar(destino/split/'annotations.json',taco)
        publicar_bytes(destino/split/'imagenes.jsonl',b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in records))
    publicar(destino/'procedencia_y_revisiones.json',cohorte);publicar(destino/'plan_particiones.json',plan)
    publicar(destino/'inventario.json',estado);publicar(destino/'protocolo_generalizacion.json',protocolo)
    files={p.relative_to(destino).as_posix():sha(p) for p in destino.rglob('*') if p.is_file()}
    nombre='ENSAYO_DESCARTABLE.json' if ensayo else 'CONGELADA.json'
    publicar(destino/nombre,{'version':VERSION,'tipo_datos':cohorte['tipo_datos'],'archivos':files,
        'sha256_politica':sha(POLITICA),'sha256_protocolo_generalizacion':sha(PROTOCOLO),'sha256_cohorte':digest(cohorte),
        'entrenamiento_autorizado':False,'evaluacion_autorizada':False,'test_original_evaluado':False})
    return estado


class ColeccionCongelada:
    """Adaptador de ensayo/preparación; nunca abre GT generalización ni rutas históricas."""
    def __init__(self,raiz,*,ensayo=False,split='train_nuevo'):
        self.raiz=Path(raiz)
        if split!='train_nuevo':raise PermissionError('Generalización bloqueada; requiere un protocolo futuro de evaluación explícito')
        marca='ENSAYO_DESCARTABLE.json' if ensayo else 'CONGELADA.json';commit=leer(self.raiz/marca)
        if commit.get('version')!=VERSION or commit.get('entrenamiento_autorizado') or commit.get('evaluacion_autorizada'):raise ValueError('Marcador fuera del protocolo de preparación V26')
        if commit['tipo_datos']!=('fixture' if ensayo else 'real'):raise ValueError('Tipo de colección incompatible')
        if commit['sha256_politica']!=sha(POLITICA) or commit['sha256_protocolo_generalizacion']!=sha(PROTOCOLO):raise ValueError('Protocolo de otra versión')
        for f,h in commit['archivos'].items():
            if sha(ruta_segura(self.raiz,f))!=h:raise ValueError('Colección congelada alterada')
        self.registros=[json.loads(l) for l in (self.raiz/'train_nuevo/imagenes.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(r['split']!='train_nuevo' for r in self.registros):raise ValueError('No mezclar generalización con train')
        taco=leer(self.raiz/'train_nuevo/annotations.json');self.anotaciones=taco['annotations'];self.categorias={c['id']:c['name'] for c in taco['categories']}

    def __len__(self):return len(self.registros)

    def __getitem__(self,i):
        from PIL import Image
        from ecowaste.datasets import letterbox,objetivo_deteccion
        r=self.registros[i]
        if sha(ruta_segura(self.raiz,r['ruta']))!=r['sha256_archivo']:raise ValueError('Imagen congelada modificada')
        with Image.open(ruta_segura(self.raiz,r['ruta'])) as im:x,g=letterbox(im.convert('RGB'),384)
        return x,objetivo_deteccion(r,g)

    def contornos(self,registro,caja):
        a=self.anotaciones[caja['id_interno']-1]
        if a['id']!=caja['id_original'] or a['image_id']!=registro['id_original'] or a['bbox']!=caja['bbox_original']:
            raise ValueError('Polígono y caja de distinta instancia')
        if self.categorias[a['category_id']]!=caja['categoria_original'] or caja['clase']!=(caja['categoria_original'] if caja['categoria_original'] in CLASES else None):
            raise ValueError('Polígono y material de distinta instancia')
        return a['segmentation']

    def rasterizar(self,registro,lado=48):
        from ecowaste.datasets import objetivo_deteccion
        return rasterizar_contornos(objetivo_deteccion(registro),[(self.contornos(registro,a),CLASES.index(a['clase']))
            for a in registro['cajas'] if a['clase']],lado,1/64)

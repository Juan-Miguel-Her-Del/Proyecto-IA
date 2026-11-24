"""Diarios humanos ligados a imágenes y versiones, sin aprobación automática de GT."""
from pathlib import Path
from datetime import datetime,timezone
from ecowaste.incorporacion import (cargar,leer,digest,sha,politica,normalizar_instancia,
    identificador,texto,qa_geometria,CLASES,IGNORADAS,publicar,bloqueo,VERSION,destino_nuevo,firma_geometria)


def leer_diario(paquete):
    c=cargar(paquete);anterior=None;eventos=[]
    for i,f in enumerate(sorted((Path(paquete)/'revisiones').glob('*.json')),1):
        r=leer(f)
        if f.name!=f'{i:06d}.json' or r['numero']!=i or r['anterior']!=anterior or r['sha256_coleccion']!=digest(c):
            raise ValueError('Diario incompleto, de otra versión o reordenado')
        if r['huella']!=digest({k:v for k,v in r.items() if k!='huella'}):raise ValueError('Revisión alterada')
        eventos.append(r);anterior=sha(f)
    return c,eventos


def doble_obligatoria(c,im,eventos=(),final=()):
    clases={a['material'] for a in [*im['instancias'],*final]}
    clases.update(c['objetos_fisicos'][oid]['material_declarado'] for oid in im['objetos_fisicos_ids'])
    for ev in eventos:
        if ev['imagen_id']==im['id']:clases.update(a['material'] for a in ev['instancias'])
    p=politica()
    # Ambiguos también requieren dos miradas; no evitar el control cambiando una clase.
    if clases & (set(p['doble_anotacion_obligatoria'])|IGNORADAS):return True
    return int(digest({'coleccion':c['coleccion_id'],'imagen':im['sha256_archivo']})[:8],16)/2**32<p['muestreo_doble_resto']


def normalizar_revision(instancias,im,c):
    if not isinstance(instancias,list):raise ValueError('Anotación humana de imagen completa ausente')
    salida=[];ids=set();fisicos=set();geometrias=set()
    for a in instancias:
        material=a['material'];aid=identificador(a['id']);oid=identificador(a['objeto_fisico_id'])
        if aid in ids or oid in fisicos:raise ValueError('Instancia física repetida en la revisión')
        ids.add(aid);fisicos.add(oid)
        taco={'id':aid,'category_id':'1','segmentation':a['poligonos'],
            'eco':{'objeto_fisico_id':oid,'visible_only':a.get('visible_only'),'holes':a.get('holes',[]),
                'origen':'humana','oclusion':a['oclusion'],'truncado':a['truncado'],'transparente':a['transparente'],
                'solapa_con':a['solapa_con'],'motivo_ambiguo':a.get('motivo_ambiguo')}}
        normal=normalizar_instancia(taco,im,{'1':material},c['objetos_fisicos']);firma=firma_geometria(normal)
        if firma in geometrias:raise ValueError('Anotación geométrica duplicada entre identidades físicas')
        geometrias.add(firma);salida.append(normal)
    return sorted(salida,key=lambda a:a['objeto_fisico_id'])


def registrar_revision(paquete,entrada):
    paquete=Path(paquete)
    cargar(paquete)  # Verificar antes de crear siquiera el archivo de bloqueo.
    if entrada.get('version')!=VERSION:raise ValueError('Revisión de otro protocolo')
    with bloqueo(paquete/'revision.lock'):
        c,evs=leer_diario(paquete);imagen=identificador(entrada['imagen_id']);im=next((i for i in c['imagenes'] if i['id']==imagen),None)
        if im is None or entrada.get('sha256_coleccion')!=digest(c) or entrada.get('sha256_imagen')!=im['sha256_archivo']:
            raise ValueError('Revisión de otra imagen o versión')
        fase=entrada.get('fase');rid=identificador(entrada.get('revisor_id'));actor=c['revisores'].get(rid)
        if fase not in ['primera','segunda','adjudicacion'] or not actor:raise ValueError('Fase/revisor no registrado')
        if ('adjudicador' if fase=='adjudicacion' else 'anotador') not in actor['roles']:raise ValueError('Rol no autorizado')
        if entrada.get('cobertura_completa') is not True:raise ValueError('La revisión debe comprobar la imagen completa, incluidas omisiones')
        declaracion='humana_adjudicada' if fase=='adjudicacion' else 'humana_independiente'
        if entrada.get('declaracion')!=declaracion:raise ValueError('No se aceptan aprobaciones automáticas ni solo un score')
        anteriores=[e for e in evs if e['imagen_id']==imagen]
        if any(e['fase']==fase or e['fase']=='adjudicacion' for e in anteriores):raise ValueError('Fase ya publicada; corregir mediante paquete nuevo')
        if any(e['persona_id']==actor['persona_id'] for e in anteriores):raise ValueError('Las fases requieren personas físicas distintas')
        instancias=normalizar_revision(entrada['instancias'],im,c);motivo=texto(entrada.get('motivo'),'motivo de revisión')
        doble=doble_obligatoria(c,im,anteriores,instancias)
        if fase=='adjudicacion':
            requeridas={'primera','segunda'} if doble else {'primera'}
            if not requeridas<= {e['fase'] for e in anteriores}:raise ValueError('Faltan anotaciones humanas independientes')
            if entrada.get('solapes_revisados') is not True:raise ValueError('Falta revisión explícita de solapes')
            omitidos=set(im['objetos_fisicos_ids'])-{a['objeto_fisico_id'] for a in instancias}
            no_visibles=entrada.get('no_visibles',{})
            if not isinstance(no_visibles,dict) or set(no_visibles)!=omitidos:raise ValueError('Justificar cada objeto registrado que no recibe polígono final')
            for oid,decision in no_visibles.items():
                if decision.get('motivo') not in ['totalmente_ocluido','fuera_cuadro','no_es_residuo'] or not decision.get('detalle'):
                    raise ValueError('La ambigüedad no permite omitir un objeto visible: anotar ignore o justificar no visibilidad')
            qa=qa_geometria({**im,'instancias':instancias})
            if any(x['materiales_distintos'] and not x['declarado'] and x['fraccion_menor']>.01 and x['pixeles']>4 for x in qa['solapes']):
                raise ValueError('Solape visible entre materiales sin declaración recíproca; adjudicar, no autocorregir')
            if c['tipo_datos']=='real':
                candidatos=[d for d in leer(paquete/'duplicados.json')['historicos'] if d['imagen']==imagen]
                decisiones={d['candidato']:d for d in entrada.get('duplicados_resueltos',[])}
                for d in candidatos:
                    res=decisiones.get(digest(d),{})
                    if d['exacto'] or res.get('decision')!='distinto' or not res.get('motivo'):raise ValueError('Duplicado histórico pendiente de adjudicación')
        r={'numero':len(evs)+1,'anterior':sha(paquete/'revisiones'/f'{len(evs):06d}.json') if evs else None,
            'sha256_coleccion':digest(c),'imagen_id':imagen,'sha256_imagen':im['sha256_archivo'],'fase':fase,
            'revisor_id':rid,'persona_id':actor['persona_id'],'tipo_datos':c['tipo_datos'],
            'instancias':instancias,'cobertura_completa':True,'declaracion':declaracion,'motivo':motivo,
            'registrada_en_UTC':datetime.now(timezone.utc).isoformat(),
            'no_visibles':entrada.get('no_visibles',{}),
            'doble_obligatoria':doble,'solapes_revisados':entrada.get('solapes_revisados',False),
            'duplicados_resueltos':entrada.get('duplicados_resueltos',[])}
        r['huella']=digest(r);publicar(paquete/'revisiones'/f'{len(evs)+1:06d}.json',r)
    return r


def estado_revision(paquete):
    c,evs=leer_diario(paquete);filas=[]
    for im in c['imagenes']:
        es=[e for e in evs if e['imagen_id']==im['id']];por_fase={e['fase']:e for e in es}
        doble=doble_obligatoria(c,im,es)
        estado='aceptada' if 'adjudicacion' in por_fase else 'pendiente_primera' if 'primera' not in por_fase else 'pendiente_segunda' if doble and 'segunda' not in por_fase else 'pendiente_adjudicacion'
        final=por_fase.get('adjudicacion');discrepancias=[]
        if 'primera' in por_fase and 'segunda' in por_fase:
            a={x['objeto_fisico_id']:x for x in por_fase['primera']['instancias']};b={x['objeto_fisico_id']:x for x in por_fase['segunda']['instancias']}
            from ecowaste.incorporacion import mascara
            for oid in sorted(set(a)|set(b)):
                x,y=a.get(oid),b.get(oid)
                if x is None or y is None:discrepancias.append({'objeto':oid,'tipo':'omision_o_instancia_adicional'});continue
                mx=mascara(x['poligonos'],im['width'],im['height']);my=mascara(y['poligonos'],im['width'],im['height']);iou=float((mx&my).sum())/max(1,int((mx|my).sum()))
                if x['material']!=y['material'] or iou<politica()['qa']['iou_anotadores_revision']:
                    discrepancias.append({'objeto':oid,'tipo':'material_o_geometria','material_1':x['material'],'material_2':y['material'],'iou_mask':iou})
        filas.append({'imagen_id':im['id'],'estado':estado,'doble_obligatoria':doble,'discrepancias':discrepancias,
            'instancias_finales':final['instancias'] if final else [],'huella_adjudicacion':final['huella'] if final else None})
    return c,filas,evs


def exportar_tareas(paquete,salida,*,adjudicacion=False):
    c,estados,evs=estado_revision(paquete);salida=destino_nuevo(salida)
    salida.mkdir(parents=True)
    from ecowaste.incorporacion import copiar_imagen_verificada
    tareas=[]
    for im,estado in zip(c['imagenes'],estados):
        copiar_imagen_verificada(Path(paquete)/im['file_name'],salida/im['file_name'],im['sha256_archivo'])
        for fase in (['adjudicacion'] if adjudicacion else ['primera','segunda']):
            item={'version':VERSION,'sha256_coleccion':digest(c),'imagen_id':im['id'],'sha256_imagen':im['sha256_archivo'],
                'file_name':im['file_name'],'fase':fase,'revisor_id':None,'cobertura_completa':False,
                'objetos_fisicos_ids':im['objetos_fisicos_ids'],
                'declaracion':'humana_adjudicada' if adjudicacion else 'humana_independiente',
                'instancias':[],'motivo':'','solapes_revisados':False,'duplicados_resueltos':[],'no_visibles':{}}
            if adjudicacion:
                anteriores=[e for e in evs if e['imagen_id']==im['id']]
                if 'primera' not in {e['fase'] for e in anteriores} or estado['doble_obligatoria'] and 'segunda' not in {e['fase'] for e in anteriores}:
                    raise ValueError('No exportar adjudicación antes de recibir las anotaciones independientes')
                item['anotaciones_previas']=anteriores;item['discrepancias']=estado['discrepancias']
                item['duplicados_historicos']=[{'candidato':digest(d),**d} for d in leer(Path(paquete)/'duplicados.json')['historicos'] if d['imagen']==im['id']]
            tareas.append(item)
    # Las tareas independientes no contienen materiales, máscaras ni cajas propuestas.
    publicar(salida/'tareas.json',{'tipo_datos':c['tipo_datos'],'objetos_fisicos_ids':sorted(c['objetos_fisicos']),
        'tareas':tareas,'sin_GT_previo':not adjudicacion,'aviso':'No publicar como entrenamiento. Completar imagen entera y registrar cada fase.'})
    return tareas

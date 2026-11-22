"""Tablas de decisiones y comprobaciones específicas a partir del respaldo humano."""
from collections import Counter
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,CLASES
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import revision_humana_final
from scripts.respaldar_anotaciones_v28 import SALIDA


def main():
    fs=leer(SALIDA/'imagenes.json');os=leer(SALIDA/'objetos.json');obs=leer(SALIDA/'observaciones_visuales_y_taxonomia.json')
    t=Taller(RAIZ/leer(SALIDA/'respaldo_verificado.json')['destino'],RAIZ)
    tablas=[]
    for c in [*CLASES,'compuesto','ambiguo','fuera_alcance','sin_asignar']:
        act=[a for a in os if a['material']==c and not a['descartada']]
        completas=[a for a in act if a['imagen_cobertura_completa']]
        anillos_validos=lambda a:not any(e!='relleno_paridad_editor_difiere_union_exportador' for e in a['errores'])
        tablas.append({'material':c,'objetos_guardados':len(act),'objetos_en_imagenes_completas':len(completas),
            'objetos_con_contornos_individuales_validos':sum(anillos_validos(a) for a in completas),
            'objetos_con_mascara_sin_discrepancia_relleno':sum(a['valido_geometria'] for a in completas),
            'imagenes_completas_distintas':len({a['imagen'] for a in completas}),
            'borrador_o_error_mascara':sum(not a['imagen_cobertura_completa'] or bool(a['errores']) for a in act)})
    publicar(SALIDA/'tabla_informe_materiales.json',tablas)
    wa=[]
    for old in leer(RAIZ/'resultados/v27_2_anotacion/piloto_sam2_cpu.json')['incidencias']:
        f=next(f for f in fs if f['imagen']==old['imagen'])
        a=next(a for a in f['qa_objetos'] if a['id']==old['instancia_auxiliar'])
        wa.append({'imagen':f['imagen'],'objeto':a['id'],'indice_editor':a['indice_editor'],
                   'tiene_contorno_valido':a['valido_geometria'],'descartada':a['descartada'],
                   'material_guardado':a['material'],'estado_imagen':f['estado']})
    org_wsc=[]
    for iid,im in t.imagenes.items():
        if im['fuente']!='wsc':continue
        ev=revision_humana_final(t.diario(iid));por_id={a['id']:a for a in ev['anotaciones']} if ev else {}
        for a in im['anotaciones']:
            if a.get('categoria_original')=='Organic':
                org_wsc.append({'imagen':iid,'objeto':a['id'],'material_inicial':a['material'],
                    'material_guardado':por_id.get(a['id'],{}).get('material'),'revision_persona':ev['persona_id'] if ev else None,
                    'cobertura_completa_declarada':ev['cobertura_completa'] if ev else False,
                    'asignacion_proviene_de_evento_humano':bool(ev),'no_se_acredita_como_restos_alimentarios':True})
    cambios=set()
    for f in fs:
        d=f['comparacion']
        if d:
            cambios.update((f['imagen'],a) for a in d['contornos_cambiados'])
            cambios.update((f['imagen'],a['id']) for a in d['reclasificados'])
    ids=set(leer(SALIDA/'suficiencia_exploratoria.json')['imagenes_candidatas_sin_incidencias_identificadas'])
    audit=leer(SALIDA/'auditoria.json')
    publicar(SALIDA/'comprobaciones_especificas.json',{'once_cajas_WaRP':wa,'WSC_Organic':org_wsc,
        'propuestas_con_cambio_geometria_o_material_sin_doble_conteo':len(cambios),
        'organicos_geometricamente_potenciales':audit['organicos_utilizables_48'],'organicos_HPI_geometricamente_potenciales':audit['organicos_HPI_utilizables_48'],
        'organicos_candidatos_sin_incidencias':sum(a['organico_utilizable_48'] for a in os if a['imagen'] in ids),
        'organicos_HPI_candidatos_sin_incidencias':sum(a['organico_utilizable_48'] for a in os if a['imagen'] in ids and a['fuente']=='hpi'),
        'todas_las_HPI_tienen_evento_y_contorno':all(f['estado']=='completa_usuario_unico' and f['objetos_activos_guardados']>0 for f in fs if f['fuente']=='hpi'),
        'cobertura_visual_HPI_confirmada':False,'omisiones_ZeroWaste_pendientes':True})
    lineas=['# V28: lista concreta de correcciones pendientes','',
        'Consultar la imagen por su número y su ID en el editor V27.2. No se modificaron anotaciones. Las observaciones del asistente no son revisión humana independiente.','',
        '| Nº imagen editor | Imagen | Decisiones pendientes |','|---:|---|---|']
    cola=[]
    for n,f in enumerate(fs,1):
        motivos=[]
        if f['estado']!='completa_usuario_unico':motivos.append(f['estado'])
        motivos.extend('Objetos '+str(x.get('objeto',x.get('objetos')))+': '+','.join(x['motivos']) for x in f['errores'])
        motivos.extend('Revisar objetos '+str(x.get('objeto',x.get('objetos')))+': '+','.join(x['motivos']) for x in f['avisos_visuales'])
        motivos.extend(x['motivo'] for x in obs if x['imagen']==f['imagen'])
        if not motivos:continue
        texto='; '.join(motivos).replace('|','/').replace('\n',' ')
        lineas.append(f'| {n} | `{f["imagen"]}` | {texto} |')
        cola.append({'numero_imagen_editor':n,'imagen':f['imagen'],'motivos':motivos})
    from ecowaste.incorporacion import publicar_bytes
    publicar_bytes(SALIDA/'lista_correcciones.md',('\n'.join(lineas)+'\n').encode('utf-8'))
    publicar(SALIDA/'cola_correcciones.json',cola)
    print({'imagenes_en_cola':len(cola),'propuestas_con_cambios':len(cambios),'once_cajas_corregidas':all(a['tiene_contorno_valido'] for a in wa)},flush=True)


if __name__=='__main__':main()

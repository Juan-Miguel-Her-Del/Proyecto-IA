"""Instantánea y vistas para corregir anotaciones como IA, nunca suplantando al usuario."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import sys
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import mask_poligonos,revision_humana_final
from ecowaste.incorporacion import publicar,sha
from scripts.respaldar_anotaciones_v28 import respaldar
from scripts.anotar_v27_2 import PROYECTO
from scripts.auditar_v28_3 import gestion,SALIDA
COLORES={'plastico':'#22bbee','papel':'#ffe555','carton':'#ee9955','vidrio':'#55ddbb','metal':'#bb99ff','organico':'#99ee33','compuesto':'#ff77cc','ambiguo':'#ff5555','sin_asignar':'#ff5555'}


def vista(t,iid,anns,path,ancho=1100):
    im=t.imagen(iid)
    with Image.open(t.archivo_imagen(iid)) as p:foto=p.convert('RGB')
    foto.thumbnail((ancho,850));sx,sy=foto.width/im['width'],foto.height/im['height'];over=Image.new('RGBA',foto.size)
    d=ImageDraw.Draw(over);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    leyenda=[]
    for n,a in enumerate(anns,1):
        if a.get('descartada'):continue
        color=COLORES.get(a['material'],'white')
        for pol in a.get('poligonos',[]):d.line([(x*sx,y*sy) for x,y in zip(pol[::2],pol[1::2])]+[(pol[0]*sx,pol[1]*sy)],fill=color,width=2)
        for pol in a.get('huecos',[]):d.line([(x*sx,y*sy) for x,y in zip(pol[::2],pol[1::2])]+[(pol[0]*sx,pol[1]*sy)],fill='#ff4444',width=2)
        if a.get('bbox'):
            x,y,w,h=a['bbox'];d.text((x*sx,y*sy),str(n),fill=color,stroke_width=1,stroke_fill='black',font=font)
        leyenda.append(f'{n} {a["material"]} | {a["id"]}')
    tile=Image.new('RGB',(foto.width*2,max(foto.height+110,120+len(leyenda)*21)),'#17202b')
    tile.paste(foto,(0,60));tile.paste(Image.alpha_composite(foto.convert('RGBA'),over).convert('RGB'),(foto.width,60))
    out=ImageDraw.Draw(tile);out.text((10,10),iid+' | original / contornos actuales',fill='white',font=font)
    for n,txt in enumerate(leyenda):out.text((10+(n%2)*foto.width,foto.height+70+(n//2)*21),txt,fill='white',font=font)
    path.parent.mkdir(parents=True,exist_ok=True);tile.save(path)


def main():
    import torch
    torch.set_num_threads(4)
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    datos=RAIZ/'datos/nuevos/v28_3/gestion'/('ajustes_IA_'+marca);res=SALIDA/'ajustes_asistente'/marca
    hashes=respaldar(PROYECTO,datos/'respaldo_humano');t=Taller(datos/'respaldo_humano',RAIZ)
    g=gestion(Taller(PROYECTO,RAIZ));pendientes=[p for p in g.listar() if p['bloqueante'] and p['estado'] in ['pendiente','requiere_decision']]
    ids=sorted({p['imagen'] for p in pendientes},key=lambda iid:(min(p['prioridad'] for p in pendientes if p['imagen']==iid),iid))
    registros=[]
    for n,iid in enumerate(ids):
        tarea=t.tarea(iid);ev=revision_humana_final(t.diario(iid));anns=deepcopy(tarea['anotaciones'])
        row={'imagen':tarea['imagen'],'anotaciones_base':anns,'revision_humana_base':(ev or {}).get('huella'),
            'revision_base':tarea['revision'],'incidencias':[p for p in pendientes if p['imagen']==iid],
            'estado':'pendiente_ajuste_asistente_IA','aceptada_V26':False,'revision_humana_simulada':False}
        publicar(datos/'imagenes'/f'{n:02d}.json',row);vista(t,iid,anns,datos/'vistas'/f'{n:02d}.png')
        registros.append({'indice':n,'imagen':iid,'archivo':(datos/'imagenes'/f'{n:02d}.json').relative_to(RAIZ).as_posix(),
            'vista':(datos/'vistas'/f'{n:02d}.png').relative_to(RAIZ).as_posix()})
    publicar(res/'preparacion.json',{'datos':datos.relative_to(RAIZ).as_posix(),'imagenes':registros,
        'archivos_respaldo':hashes,'eventos_humanos_base':sum(len(t.diario(i)) for i in t.imagenes),
        'proyecto_sha256':t.base['huella'],'entrenamiento':False,'test_evaluado':False})
    print(res/'preparacion.json',flush=True)


if __name__=='__main__':main()

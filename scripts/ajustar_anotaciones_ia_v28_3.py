"""Ajustes visuales IA aislados de los diarios humanos; SAM sólo en CPU."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ecowaste.incorporacion import leer, publicar, digest, sha
from ecowaste.auxiliares_publicos import Taller, rle_coco, contornos_mask
from ecowaste.validacion_anotaciones import mask_poligonos


def decodificar(rle):
    h, w = rle['size']
    counts = rle['counts']
    if any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) != h*w:
        raise ValueError('RLE inválido')
    flat = np.repeat(np.arange(len(counts)) % 2, counts).astype(bool)
    return flat.reshape((h, w), order='F')


def bbox_mask(m):
    y, x = np.where(m)
    if not len(x):
        raise ValueError('Máscara vacía')
    return [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)]


def comprobar_origen(datos, taller):
    """Rechazar una base humana obsoleta; nunca escribir en el proyecto humano."""
    from ecowaste.validacion_anotaciones import revision_humana_final
    for p in sorted((datos/'imagenes').glob('*.json')):
        r = leer(p)
        actual = revision_humana_final(taller.diario(r['imagen']['id']))
        if (actual or {}).get('huella') != r['revision_humana_base']:
            raise ValueError('Nueva revisión humana; rebasar explícitamente: '+r['imagen']['id'])


class DestinoIA:
    def __init__(self, taller, destino, clave):
        self.taller, self.destino, self.clave = taller, destino, clave
    def imagen(self, iid): return self.taller.imagen(iid)
    def archivo_imagen(self, iid): return self.taller.archivo_imagen(iid)
    def registrar_ia(self, iid, resultado, indicaciones, modelo):
        r = {'id':'asistente_'+self.clave, 'imagen_id':iid, 'resultado':resultado,
             'indicaciones':indicaciones, 'modelo':modelo, 'actor_tipo':'IA',
             'revision_humana':False, 'aceptada_V26':False}
        publicar(self.destino, r)
        return r


def proponer(datos, specs):
    from ecowaste.asistente_sam2 import AsistenteSAM2
    t = Taller(datos/'respaldo_humano', RAIZ)
    motor = AsistenteSAM2(RAIZ)
    for n, s in enumerate(specs['propuestas']):
        fila = leer(datos/'imagenes'/f"{s['indice']:02d}.json")
        iid = fila['imagen']['id']; im = fila['imagen']; w,h = im['width'],im['height']
        clave = digest(s)[:24]; salida = datos/'sam_IA'/(clave+'.json')
        if salida.exists():
            continue
        b = s['caja']; pts = s.get('puntos', [[(b[0]+b[2])/2,(b[1]+b[3])/2]])
        ind = {'caja':[b[0]*w,b[1]*h,b[2]*w,b[3]*h],
               'puntos':[[x*w,y*h] for x,y in pts], 'etiquetas':s.get('etiquetas',[1]*len(pts))}
        r = motor.proponer(DestinoIA(t,salida,clave), iid, ind)
        print(f"{n+1}/{len(specs['propuestas'])} {iid} {s['detalle']} {r['resultado']['segundos']:.1f}s", flush=True)
    motor.liberar()


def contactos(datos, specs):
    """Fotografía con cada propuesta separada: inspección antes de incorporarla."""
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for page in range((len(specs['propuestas'])+5)//6):
        out=Image.new('RGB',(1500,1125),'#17202b'); d=ImageDraw.Draw(out)
        for j,s in enumerate(specs['propuestas'][page*6:page*6+6]):
            n=page*6+j; fila=leer(datos/'imagenes'/f"{s['indice']:02d}.json"); im=fila['imagen']
            clave=digest(s)[:24]; r=leer(datos/'sam_IA'/(clave+'.json'))
            foto=Image.open(RAIZ/im['archivo']).convert('RGB');m=decodificar(r['resultado']['rle_ia_original'])
            a=np.array(foto); a[m]=(a[m]*.55+np.array([0,240,80])*.45).astype(np.uint8)
            p=Image.fromarray(a);p.thumbnail((495,325));x=(j%3)*500;y=(j//3)*560
            out.paste(p,(x,y+35));d.text((x+4,y+4),f"{n} · {im['id']}",fill='white',font=font)
            d.text((x+4,y+365),s['detalle'][:55],fill='white',font=font)
            d.text((x+4,y+390),f"{s['material']} · SAM {r['resultado']['score_SAM_no_probabilidad_material']:.3f}",fill='white',font=font)
        destino=datos/specs['version'];destino.mkdir(exist_ok=True)
        out.save(destino/f'propuestas_{page:02}.png')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('accion',choices=['proponer','contactos'])
    p.add_argument('--preparacion',type=Path,required=True);p.add_argument('--criterios',type=Path,required=True);a=p.parse_args()
    prep=leer(a.preparacion);datos=RAIZ/prep['datos'];specs=leer(a.criterios)
    if a.accion=='proponer':proponer(datos,specs)
    else:contactos(datos,specs)


if __name__=='__main__':main()

"""Contactos privados de las anotaciones guardadas, sin nuevas etiquetas."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image,ImageDraw,ImageFont
from ecowaste.incorporacion import leer,publicar
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import iniciales
from scripts.respaldar_anotaciones_v28 import SALIDA

COLORES={'plastico':'#40c7ff','papel':'#fff066','carton':'#ffa33c','vidrio':'#40f7c4','metal':'#c6a1ff','organico':'#89f42e','compuesto':'#fca2ef'}


def main():
    b=leer(SALIDA/'respaldo_verificado.json');t=Taller(RAIZ/b['destino'],RAIZ);filas=leer(SALIDA/'imagenes.json')
    destino=RAIZ/'datos/nuevos/v28/visuales';destino.mkdir(parents=True,exist_ok=True)
    registros=[]
    grupos={'HPI':[f for f in filas if f['fuente']=='hpi'],'ZeroWaste':[f for f in filas if f['fuente']=='zerowaste'],
            'avisos':[f for f in filas if f['avisos_visuales'] or f['errores']]}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for nombre,fs in grupos.items():
        for k in range(0,len(fs),12):
            parte=fs[k:k+12];sheet=Image.new('RGB',(1600,310*((len(parte)+3)//4)),'white')
            for n,f in enumerate(parte):
                iid=f['imagen'];im=t.imagen(iid);ev=t.diario(iid);normal=[e for e in ev if e['modo']=='asistida']
                anns=normal[-1]['anotaciones'] if normal else iniciales(t,iid)
                with Image.open(t.archivo_imagen(iid)) as pil:pil=pil.convert('RGB')
                pil.thumbnail((395,245));sx,sy=pil.width/im['width'],pil.height/im['height'];p=pil.convert('RGBA')
                overlay=Image.new('RGBA',p.size);d=ImageDraw.Draw(overlay)
                for j,a in enumerate(anns,1):
                    if a.get('descartada') or not a.get('poligonos'):continue
                    color=COLORES.get(a['material'],'#ffffff');rgb=tuple(bytes.fromhex(color[1:]))
                    for pol in a['poligonos']:d.polygon([(x*sx,y*sy) for x,y in zip(pol[::2],pol[1::2])],fill=(*rgb,50),outline=(*rgb,255),width=1)
                    for pol in a.get('huecos',[]):d.polygon([(x*sx,y*sy) for x,y in zip(pol[::2],pol[1::2])],fill=(0,0,0,0),outline=(*rgb,255),width=1)
                    if a.get('bbox'):
                        x,y,*_=a['bbox'];d.text((x*sx,y*sy),str(j),fill='white',stroke_width=1,stroke_fill='black',font=font)
                tile=Image.new('RGB',(400,310),'white');tile.paste(Image.alpha_composite(p,overlay).convert('RGB'),(0,0));d=ImageDraw.Draw(tile)
                d.text((2,248),iid[:52],font=font,fill='black');d.text((2,268),f['estado'],font=font,fill='black')
                d.text((2,287),'SUPERVISIÓN GUARDADA' if normal else 'SIN GUARDADO; propuestas originales',font=font,fill='black')
                sheet.paste(tile,((n%4)*400,(n//4)*310))
            path=destino/f'{nombre}_{k//12:02d}.jpg'
            if path.exists():raise FileExistsError('Contacto existente')
            sheet.save(path,quality=94);registros.append(path.relative_to(RAIZ).as_posix())
    publicar(SALIDA/'visuales.json',{'contactos_privados':registros,'no_es_revision_independiente':True,'anotaciones_generadas':0})


if __name__=='__main__':main()

"""CLI de incorporación V26; no tiene comandos de entrenamiento ni evaluación."""
import argparse
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import importar,leer,publicar,destino_nuevo
from ecowaste.revision_datos import registrar_revision,estado_revision,exportar_tareas
from ecowaste.coleccion_nueva import consolidar,particionar,inventario,congelar


def argumentos(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='orden',required=True)
    q=sub.add_parser('importar');q.add_argument('--anotaciones',required=True,type=Path);q.add_argument('--imagenes',required=True,type=Path)
    q.add_argument('--salida',required=True,type=Path);q.add_argument('--revisores',required=True,type=Path);q.add_argument('--ensayo',action='store_true')
    q=sub.add_parser('tareas');q.add_argument('--paquete',required=True,type=Path);q.add_argument('--salida',required=True,type=Path);q.add_argument('--adjudicacion',action='store_true')
    q=sub.add_parser('revisar');q.add_argument('--paquete',required=True,type=Path);q.add_argument('--entrada',required=True,type=Path)
    q=sub.add_parser('estado');q.add_argument('--paquete',required=True,type=Path);q.add_argument('--salida',required=True,type=Path)
    q=sub.add_parser('plan');q.add_argument('--paquetes',required=True,nargs='+',type=Path);q.add_argument('--salida',required=True,type=Path)
    q=sub.add_parser('inventario');q.add_argument('--paquetes',nargs='*',type=Path,default=[]);q.add_argument('--plan',type=Path);q.add_argument('--salida',required=True,type=Path)
    q=sub.add_parser('congelar');q.add_argument('--paquetes',required=True,nargs='+',type=Path);q.add_argument('--plan',required=True,type=Path)
    q.add_argument('--salida',required=True,type=Path);q.add_argument('--ensayo',action='store_true')
    return p.parse_args(argv)


def salida_nueva(path):
    """Rechaza todo destino dentro de raíces científicas históricas o de código."""
    return destino_nuevo(path)


def main(argv=None):
    a=argumentos(argv)
    if hasattr(a,'salida'):salida_nueva(a.salida)
    if a.orden=='importar':
        r=importar(a.anotaciones,a.imagenes,a.salida,leer(a.revisores)['revisores'],ensayo=a.ensayo)
        print(f"Importadas {len(r['imagenes'])} imágenes en cuarentena; cero GT aceptado")
    elif a.orden=='tareas':exportar_tareas(a.paquete,a.salida,adjudicacion=a.adjudicacion)
    elif a.orden=='revisar':
        # cargar() exige un paquete V26 íntegro; no acepta datasets históricos.
        r=registrar_revision(a.paquete,leer(a.entrada));print('Revisión inmutable '+r['huella'])
    elif a.orden=='estado':
        _,estados,_=estado_revision(a.paquete);publicar(a.salida,estados)
    elif a.orden=='plan':publicar(a.salida,particionar(consolidar(a.paquetes)))
    elif a.orden=='inventario':publicar(a.salida,inventario(consolidar(a.paquetes),leer(a.plan) if a.plan else None))
    elif a.orden=='congelar':congelar(consolidar(a.paquetes),leer(a.plan),a.salida,ensayo=a.ensayo)


if __name__=='__main__':main()

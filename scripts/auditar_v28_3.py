"""Auditoría incremental de guardados reales; snapshots nuevos, sin entrenar/exportar GT."""
import argparse
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.auxiliares_publicos import Taller
from ecowaste.correcciones_v28_3 import GestionCorrecciones,catalogo_historico
from ecowaste.incorporacion import leer,sha
from scripts.anotar_v27_2 import PROYECTO
SALIDA=RAIZ/'resultados/v28_3_correcciones'


def gestion(t):
    carpeta=RAIZ/'datos/nuevos/v28_3/gestion'/t.base['huella'][:16]
    return GestionCorrecciones(t,carpeta,None if (carpeta/'catalogo.json').exists() else catalogo_historico(RAIZ))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proyecto',type=Path,default=PROYECTO);a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    respaldo=leer(SALIDA/'respaldo.json')
    for r,h in respaldo['archivos'].items():
        if sha(RAIZ/respaldo['destino']/r)!=h:raise ValueError('Respaldo alterado')
        # Eventos históricos son append-only; guardar una nueva revisión no sustituye los anteriores.
        if r.startswith('revisiones/') and sha(a.proyecto/r)!=h:raise ValueError('Evento humano histórico alterado')
    t=Taller(a.proyecto,RAIZ);g=gestion(t);archivo,resumen=g.auditar(SALIDA/'auditorias')
    print(archivo,flush=True);print(resumen,flush=True)


if __name__=='__main__':main()

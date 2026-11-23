"""Actualizar QA RLE y sus conteos; sin crear decisiones humanas ni entrenar."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer
from ecowaste.auxiliares_publicos import Taller
from ecowaste.revision_ia_v28_4 import RevisionIA
from scripts.anotar_v27_2 import PROYECTO


if __name__=='__main__':
    cfg=leer(RAIZ/'configuracion/revision_ia_v28_4.json')
    p,s=RevisionIA(RAIZ/cfg['proyecto'],RAIZ,Taller(PROYECTO,RAIZ)).auditar()
    print(p,flush=True);print(s,flush=True)

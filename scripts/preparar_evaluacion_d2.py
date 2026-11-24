"""Cruza solo metadatos de partición; excluye test y no importa código ajeno."""
import argparse,json,hashlib
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--origen',type=Path,required=True);a=p.parse_args();path=a.origen/'data/reference/mixed_D2/manifest.json'
    # No se trasladan ni analizan cajas de test: únicamente ID, archivo y split.
    registros=json.loads(path.read_text(encoding='utf8'));meta=[{k:r[k] for k in ['id','file_name','split','sha256']} for r in registros if str(r['file_name']).startswith('batch_')];del registros
    por={r['id']:r for r in meta};filenames={r['file_name']:r['split'] for r in meta};autorizados={};val=[]
    for sp in ['train','val']:
        for line in (RAIZ/f'datos/procesados/deteccion/{sp}.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);autorizados[r['id_original']]={k:r[k] for k in ['id','id_original','ruta','sha256_archivo','ancho','alto','split']}
            if sp=='val':val.append(r['id_original'])
    fuera=[i for i in por if por[i]['split']=='val' and i not in autorizados]
    d2val=[dict(autorizados[i],split_d2='val') for i in sorted(autorizados) if i in por and por[i]['split']=='val']
    eco=[dict(autorizados[i],split_d2=por[i]['split'] if i in por else 'desconocido') for i in sorted(val) if i in por and por[i]['split']!='test']
    config={'fuente_manifest_sha256':sha(path),'fuente_reservados_sha256':sha(a.origen/'models/runtime/reserved_hashes.json'),'por_archivo':filenames,'por_id':{str(r['id']):r['split'] for r in meta},'nota':'Solo metadatos de partición, no GT test. D2 usa miniaturas, EcoWaste usa originales; no basta comparar hashes binarios.'}
    save(RAIZ/'checkpoints/vision_equipo/particiones_taco.json',config)
    save(RAIZ/'resultados/integracion_d2/cohortes.json',{'d2_val_autorizada':d2val,'ecowaste_val_conocida_por_d2':eco,'d2_val_excluida_por_no_estar_en_train_val_ecowaste':len(fuera),'test_evaluado':False,'seleccion':'Todos los registros de validación D2 que pertenecen a train/val autorizado EcoWaste; ninguna selección por aciertos','limitaciones':['Cohorte de desarrollo ya usada por Vision-IA; no prueba independiente','EcoWaste val es train de D2: no medir generalización D2 allí','La referencia de mapas puede haber entrenado fotos de D2 val; no es comparación causal de entrenamiento','No se abren fotografías ni anotaciones del test EcoWaste']})
    print('Cohortes:',len(d2val),'D2 val autorizada;',len(eco),'EcoWaste val; excluidas',len(fuera),flush=True)
if __name__=='__main__':main()

"""Análisis offline de los estados train V22; no ejecuta el modelo ni AdamW."""
import csv
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.reanudacion_regional import publicar
from ecowaste.reproducibilidad_v20 import RTOL,ATOL


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(4);root=RAIZ/'resultados/v22_reproducibilidad_v20';r=leer(root/'resumen.json');filas=[];evidencia={}
    for modo,rr in r['modos'].items():
        evidencia[modo]={}
        for rama,x in rr.items():
            entorno=leer(root/modo/rama/'C1/entorno_inicio.json');ops=leer(root/modo/rama/'C1/operadores.json')
            evidencia[modo][rama]={'entorno':entorno,'operadores_atencion':[o for o in ops if 'attention' in o['operador']],
                'restauracion_exacta':x['restauracion_exacta'],'entradas_y_RNG_exactos':x['entradas_y_RNG_exactos'],
                'envolvente':x['sin_divergencia_adicional_segun_envolvente'],'dentro_tolerancia':x['todos_dentro_tolerancia_original'],
                'pasos':[]}
            for s in x['pasos']:
                for par in ['C1_C2','C1_R']:
                    for etapa,z in s[par]['etapas'].items():
                        filas.append({'modo':modo,'rama':rama,'paso':s['paso'],'par':par,'etapa':etapa,**z})
                top={}
                for par in ['C1_C2','C1_R']:
                    c=leer(root/'comparaciones'/modo/rama/f"paso_{s['paso']:02d}_{par}.json")
                    pesos=[f for f in c['filas'] if f['nombre'].startswith('post.pesos.')]
                    top[par]=sorted(pesos,key=lambda f:-(f.get('max_diferencia_absoluta') or 0.))[:5]
                evidencia[modo][rama]['pasos'].append({'paso':s['paso'],'etapas_C1_C2':s['C1_C2']['etapas'],
                    'etapas_C1_R':s['C1_R']['etapas'],'parametros_mayor_delta':top,'envolvente':s['envolvente']})
    publicar(root/'evidencia_etapas.json',evidencia)
    with (root/'comparacion_por_etapa.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
    qkv=[]
    for rama in ['A','B']:
        for k in range(1,5):
            a=torch.load(root/'historico'/rama/'C1'/f'paso_{k:02d}_pre.pt',weights_only=True)
            ap=torch.load(root/'historico'/rama/'C1'/f'paso_{k:02d}_post.pt',weights_only=True)
            for otro in ['C2','R']:
                b=torch.load(root/'historico'/rama/otro/f'paso_{k:02d}_pre.pt',weights_only=True)
                bp=torch.load(root/'historico'/rama/otro/f'paso_{k:02d}_post.pt',weights_only=True)
                for n,z in ap['pesos'].items():
                    if not n.endswith('atencion.qkv.bias'):continue
                    w=bp['pesos'][n];dim=z.numel()//3
                    for j,parte in enumerate(['Q','K','V']):
                        sl=slice(j*dim,(j+1)*dim);delta=(z[sl]-w[sl]).abs();i=j*dim+int(delta.argmax())
                        qkv.append({'rama':rama,'paso':k,'comparacion':'C1_'+otro,'parametro':n,'parte':parte,
                            'max_delta_peso':float(delta.max()),'indice_max':i,
                            'fuera_tolerancia':int((~torch.isclose(z[sl],w[sl],rtol=RTOL,atol=ATOL)).sum()),
                            'max_abs_gradiente_clip_C1':float(a['gradientes_clip'][n][sl].abs().max()),
                            'max_delta_gradiente_clip':float((a['gradientes_clip'][n][sl]-b['gradientes_clip'][n][sl]).abs().max()),
                            'coordenada_max':{'peso_antes_C1':float(a['pesos'][n][i]),'peso_antes_otro':float(b['pesos'][n][i]),
                                'gradiente_C1':float(a['gradientes_clip'][n][i]),'gradiente_otro':float(b['gradientes_clip'][n][i]),
                                'exp_avg_C1':float(a['adamw']['estado'][n]['exp_avg'][i]),
                                'exp_avg_sq_C1':float(a['adamw']['estado'][n]['exp_avg_sq'][i]),
                                'peso_despues_C1':float(z[i]),'peso_despues_otro':float(w[i])}})
    publicar(root/'qkv_por_particion.json',qkv)
    publicar(root/'comparacion_pesos_finales.json',{m:{b:{par:max(x['pasos'][-1][par]['etapas']['post.pesos']['max_delta'],0.)
        for par in ['C1_C2','C1_R']} for b,x in rr.items()} for m,rr in r['modos'].items()})
    print('Análisis offline completo:',len(filas),'filas de etapas;',len(qkv),'particiones QKV')


if __name__=='__main__':main()

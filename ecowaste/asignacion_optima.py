"""Emparejamiento rectangular de coste mínimo mediante caminos aumentantes."""
import numpy as np


def asignacion_optima(costes):
    """Recibe [objetivos,consultas], devuelve una consulta única por objetivo."""
    a=np.asarray(costes,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Matriz de costes inválida')
    n,m=a.shape
    if n>m:raise ValueError('Faltan consultas para representar todos los objetos')
    if not n:return np.empty(0,dtype=np.int64)
    u=np.zeros(n+1);v=np.zeros(m+1);p=np.zeros(m+1,dtype=np.int64);camino=np.zeros(m+1,dtype=np.int64)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimos=np.full(m+1,np.inf);usados=np.zeros(m+1,dtype=bool)
        while True:
            usados[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if usados[j]:continue
                actual=a[i0-1,j-1]-u[i0]-v[j]
                if actual<minimos[j]:minimos[j]=actual;camino[j]=j0
                if minimos[j]<delta:delta=minimos[j];j1=j
            for j in range(m+1):
                if usados[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimos[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=camino[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    indices=np.empty(n,dtype=np.int64)
    for j in range(1,m+1):
        if p[j]:indices[p[j]-1]=j-1
    return indices

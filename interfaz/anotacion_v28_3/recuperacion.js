"use strict";
// Ninguna recuperación aprueba anotaciones ni transforma contornos.
const cajaDB=new Promise((resolve,reject)=>{const r=indexedDB.open("ecowaste_v28_3",1);r.onupgradeneeded=()=>r.result.createObjectStore("borradores",{keyPath:"clave"});r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
async function db(modo,fn){const d=await cajaDB;return new Promise((resolve,reject)=>{const tx=d.transaction("borradores",modo);let valor;const r=fn(tx.objectStore("borradores"));r.onsuccess=()=>{valor=r.result;};tx.oncomplete=()=>resolve(valor);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error("No se guardó el borrador local"));});}
let incidenciaActual=null,incidencias=[],pendienteGuardado=null,borradorRecuperable=null,conflictoGuardado=false;
let indicacionesPorObjeto={},temporizadorLocal=null,cargando=false,primeraCarga=true,persistencia=Promise.resolve();
function claveLocal(){return `${config.proyecto_sha256}:${tarea.imagen.id}:${tarea.modo}`;}
function copiarIndicaciones(){return {puntos:structuredClone(puntos),etiquetas:[...etiquetas],caja:structuredClone(caja),manual:structuredClone(manual)};}
function estadoLocal(){if(!tarea)return null;const a=activo();if(a)indicacionesPorObjeto[a.id]=copiarIndicaciones();return {version:"borrador_editor_v28_3_1",proyecto_sha256:config.proyecto_sha256,imagen_id:tarea.imagen.id,
 modo:tarea.modo,revision:tarea.revision,revision_huella:tarea.revision_huella,persona_id:$("persona").value,cobertura_completa:$("completa").checked,
 anotaciones:structuredClone(anns),seleccion,indicaciones_por_objeto:structuredClone(indicacionesPorObjeto),herramienta:$("modo").value,
 zoom:$("zoom").value,pendiente_guardado:structuredClone(pendienteGuardado),dirty,fecha:new Date().toISOString(),aceptada_V26:false,entrenamiento_autorizado:false};}
function hayTrabajo(){return dirty||manual.length||puntos.length||caja||Object.values(indicacionesPorObjeto).some(x=>x.manual?.length||x.puntos?.length||x.caja);}
function persistir(forzar=false){if(!tarea||cargando||(!forzar&&!hayTrabajo()))return persistencia;const s=estadoLocal(),clave=claveLocal();
 persistencia=persistencia.catch(()=>{}).then(async()=>{const anterior=await db("readonly",st=>st.get(clave));const versiones=[...(anterior?.versiones||[]),...(anterior?.estado?[anterior.estado]:[])].slice(-8);
 await db("readwrite",st=>st.put({clave,estado:s,versiones}));$("persistencia").textContent="Borrador local persistido · "+new Date().toLocaleTimeString();}).catch(e=>{$("persistencia").textContent="No se pudo persistir localmente: "+e.message+". Descarga la copia de emergencia.";throw e;});return persistencia;}
function programarPersistencia(){clearTimeout(temporizadorLocal);temporizadorLocal=setTimeout(()=>persistir().catch(()=>{}),180);}
async function conservar(){try{await persistir(true);}catch(e){const r=await api("/api/borrador",{...estadoLocal(),estado_local:estadoLocal()});$("persistencia").textContent="IndexedDB no disponible; copia sin aprobar conservada en servidor: "+r.archivo;}}
function descargar(data,nombre){const b=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});const u=URL.createObjectURL(b),a=document.createElement("a");a.href=u;a.download=nombre;a.click();setTimeout(()=>URL.revokeObjectURL(u),1500);}
function emergencia(){if(!tarea)return;descargar({tipo:"copia_emergencia_NO_revision_humana",estado:estadoLocal()},"ecowaste_emergencia_"+tarea.imagen.id.replaceAll(":","_")+"_"+Date.now()+".json");}
function mostrarError(e){$("error_detalle").hidden=false;$("error_detalle").textContent=e.detalle?JSON.stringify(e.detalle,null,2):e.message;mensaje(e.message,true);}
api=async(path,data)=>{try{const r=await fetch(path,data===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Eco-Token":config.token},body:JSON.stringify(data)});const texto=await r.text();let j;try{j=JSON.parse(texto);}catch{throw Error(`HTTP ${r.status}: respuesta no JSON. El borrador permanece local.`);}
 if(!r.ok){const e=Error(`HTTP ${r.status} · ${j.error||r.statusText}${j.id_error?" · "+j.id_error:""}`);e.detalle=j;e.status=r.status;mostrarError(e);throw e;}return j;
 }catch(e){if(e instanceof TypeError)e.message="Fallo de red: "+e.message+". Conserva/descarga el borrador y reintenta; no se ha confirmado el guardado.";throw e;}};
function congelar(v){document.querySelectorAll("aside button, aside input, aside textarea, aside select").forEach(el=>{el.disabled=v&&!['emergencia'].includes(el.id);});if(!v)$("recuperar").disabled=!borradorRecuperable;}
const seleccionarBase=seleccionar;
seleccionar=n=>{const a=activo();if(a&&!cargando)indicacionesPorObjeto[a.id]=copiarIndicaciones();seleccionarBase(n);const p=indicacionesPorObjeto[activo()?.id];if(p){puntos=structuredClone(p.puntos||[]);etiquetas=[...(p.etiquetas||[])];caja=structuredClone(p.caja);manual=structuredClone(p.manual||[]);}dibujar();programarPersistencia();};
const dibujarBase=dibujar;
dibujar=()=>{dibujarBase();if(!tarea)return;const a=activo();if(a?.bbox){const [x,y]=a.bbox;const escala=canvas.width/Math.max(1,canvas.getBoundingClientRect().width);ctx.font=`${12*escala}px sans-serif`;ctx.fillStyle="#fff";ctx.strokeStyle="#13251c";ctx.lineWidth=3*escala;const t=`${seleccion+1}. ${a.material} · ${a.id}`;ctx.strokeText(t,x,Math.max(14*escala,y-5*escala));ctx.fillText(t,x,Math.max(14*escala,y-5*escala));}programarPersistencia();};
async function prepararRecuperacion(){borradorRecuperable=null;try{const d=await db("readonly",s=>s.get(claveLocal()));if(d?.estado)borradorRecuperable=d.estado;}catch(e){$("persistencia").textContent="No está disponible IndexedDB: "+e.message+". Usa copia de emergencia.";}
 try{const x=JSON.parse(localStorage.getItem("ecowaste_emergencia_"+claveLocal())||"null");if(x&&(!borradorRecuperable||x.fecha>borradorRecuperable.fecha))borradorRecuperable=x;}catch{/* La copia IndexedDB permanece disponible. */}
 if(borradorRecuperable)$("persistencia").textContent="Hay un borrador local conservado. Recuperarlo requiere tu acción; no es un guardado humano.";$("recuperar").disabled=!borradorRecuperable;}
cargar=async(iid)=>{if(cargando||busy)return;try{if(tarea&&hayTrabajo())await conservar();cargando=true;busy=true;congelar(true);mensaje("Cargando guardado actual; tus borradores se conservan…",false,true);
 const t=await api(`/api/tarea?id=${encodeURIComponent(iid)}&modo=${independiente?"independiente":"asistida"}`);const nueva=new Image();nueva.src="/api/imagen?id="+encodeURIComponent(iid);await nueva.decode();
 tarea=t;img=nueva;anns=structuredClone(t.anotaciones);seleccion=-1;indicacionesPorObjeto={};manual=[];puntos=[];etiquetas=[];caja=null;historial=[];dirty=false;pendienteGuardado=null;conflictoGuardado=false;history.replaceState(null,"","?imagen="+encodeURIComponent(iid));
 canvas.width=t.imagen.width;canvas.height=t.imagen.height;$("completa").checked=t.cobertura_completa;$("imagen").value=iid;
 $("titulo").textContent=`${iid} · ${t.imagen.width}×${t.imagen.height}`;$("contador").textContent=`${lista.findIndex(i=>i.id===iid)+1}/${lista.length} imágenes · ${anns.length} objetos actuales`;
 $("estado").textContent=`${t.estado} · revisión ${t.revision} · ${t.cobertura_completa?"cobertura completa guardada":"borrador/sin cobertura completa"} · No aceptada V26`;
 $("estado").className="estado"+(t.cobertura_completa?" completa":"");$("detalle").textContent=`${iid} · ${t.imagen.licencia} · ${t.fecha_ultimo_guardado||"Sin guardado humano"} · Metadatos físicos desconocidos`;
 seleccionar(anns.length?0:-1);ajustar();await prepararRecuperacion();mensaje("Guardado actual cargado. Corrige únicamente las observaciones que quieras resolver; no se inicia entrenamiento.");
 }catch(e){mostrarError(e);}finally{cargando=false;busy=false;congelar(false);}await refrescarPrioridades(false);};
function aplicarBorrador(s){if(!tarea||s.proyecto_sha256!==config.proyecto_sha256||s.imagen_id!==tarea.imagen.id||s.modo!==tarea.modo)throw Error("La copia corresponde a otra imagen, proyecto o modalidad.");
 if(s.revision!==tarea.revision||s.revision_huella!==tarea.revision_huella){conflictoGuardado=true;throw Error("La copia tiene otra revisión base. Se conserva, pero no se aplica sobre una revisión más reciente. Descárgala y compara con la versión guardada.");}
 if(!Array.isArray(s.anotaciones))throw Error("Copia sin lista de anotaciones");cargando=true;
 try{anns=structuredClone(s.anotaciones);indicacionesPorObjeto=structuredClone(s.indicaciones_por_objeto||{});pendienteGuardado=s.pendiente_guardado||null;
 $("persona").value=s.persona_id||$("persona").value;$("completa").checked=!!s.cobertura_completa;$("modo").value=s.herramienta||"editar";$("zoom").value=s.zoom||100;
 seleccion=-1;seleccionar(Number.isInteger(s.seleccion)?s.seleccion:anns.length?0:-1);dirty=true;conflictoGuardado=false;ajustar();mensaje("Borrador recuperado. Revisa y guarda para crear un evento humano; no se aprobó automáticamente.");
 }finally{cargando=false;}persistir(true).catch(mostrarError);}
function contenidoGuardado(){return {imagen_id:tarea.imagen.id,proyecto_sha256:config.proyecto_sha256,revision:tarea.revision,persona_id:$("persona").value.trim(),cobertura_completa:$("completa").checked,anotaciones:structuredClone(anns)};}
async function guardarSeguro(){if(busy||!tarea)return;busy=true;congelar(true);try{if(conflictoGuardado)throw Error("Conflicto pendiente: compara con la versión guardada sin sobrescribirla.");
 const datos=contenidoGuardado();if(!/^[A-Za-z0-9][A-Za-z0-9:_-]{0,127}$/.test(datos.persona_id))throw Error("Identificador personal inválido: utiliza juan_miguel, sin espacios.");
 const estado=estadoLocal();if(datos.cobertura_completa&&Object.values(estado.indicaciones_por_objeto).some(s=>s.manual?.length||s.puntos?.length||s.caja))throw Error("Hay un polígono o indicaciones SAM sin terminar/aplicar. Completa el contorno o guarda sin cobertura completa; la copia conserva el trabajo local.");
 const huella=JSON.stringify(datos);if(!pendienteGuardado||pendienteGuardado.huella!==huella)pendienteGuardado={huella,datos:{...datos,estado_local:{...estado,pendiente_guardado:null}},id_solicitud:"sol_"+crypto.randomUUID().replaceAll("-","")};
 dirty=true;await persistir(true).catch(e=>{$("persistencia").textContent="No se pudo persistir en navegador; se intentará el guardado con copia del servidor. "+e.message;});busy=true;congelar(true);$("error_detalle").hidden=true;
 const r=await api("/api/guardar",{...pendienteGuardado.datos,id_solicitud:pendienteGuardado.id_solicitud,token_vista:tarea.token_vista});
 if(r.tarea.revision_huella!==r.revision_huella){conflictoGuardado=true;throw Error("Tu guardado está confirmado, pero existe una revisión posterior. Se conservan ambas; abre la versión guardada antes de continuar.");}
 tarea=r.tarea;anns=structuredClone(r.tarea.anotaciones);pendienteGuardado=null;dirty=false;indicacionesPorObjeto={};manual=[];puntos=[];etiquetas=[];caja=null;
 clearTimeout(temporizadorLocal);await persistencia.catch(()=>{});await db("readwrite",s=>s.delete(claveLocal())).catch(()=>{});try{localStorage.removeItem("ecowaste_emergencia_"+claveLocal());}catch{}borradorRecuperable=null;$("completa").checked=tarea.cobertura_completa;
 $("estado").textContent=`${tarea.estado} · revisión ${tarea.revision} · ${tarea.cobertura_completa?"cobertura completa guardada":"borrador guardado"} · No aceptada V26`;
 $("persistencia").textContent="Guardado confirmado con recibo y copia de emergencia del servidor.";seleccionar(Math.min(seleccion,anns.length-1));
 mensaje(`Guardado confirmado · revisión ${r.revision_guardada}${r.reintento_idempotente?" · reintento sin duplicar evento":""}. Puedes registrar el estado de la incidencia.`);
 await refrescarPrioridades(false);await refrescarAuditoria();
 }catch(e){await persistir(true).catch(()=>{});mostrarError(e);}finally{busy=false;congelar(false);}}
async function recargarContexto(){if(!tarea||busy)return;try{await conservar();const anterior=tarea;const cfg=await api("/api/config");if(cfg.proyecto_sha256!==config.proyecto_sha256)throw Error("El servidor pertenece a otro proyecto.");config=cfg;
 const actual=await api(`/api/tarea?id=${encodeURIComponent(tarea.imagen.id)}&modo=${tarea.modo}`);let propio=false;
 if(pendienteGuardado){const rr=await api("/api/recibo?id="+encodeURIComponent(pendienteGuardado.id_solicitud));propio=rr.encontrado&&rr.recibo.revision_huella===actual.revision_huella;}
 if(actual.revision===anterior.revision&&actual.revision_huella===anterior.revision_huella){tarea=actual;conflictoGuardado=false;mensaje("Contexto actualizado; contornos y borrador local conservados.");}
 else if(propio){tarea=actual;pendienteGuardado=null;conflictoGuardado=false;mensaje("El recibo confirma el guardado anterior. Tus contornos locales se conservan; revisa cualquier cambio posterior antes de guardar.");}
 else{conflictoGuardado=true;descargar({tipo:"comparacion_servidor_no_revisada",tarea:actual},"ecowaste_version_servidor.json");throw Error("Hay otra revisión en el servidor. No se rebasó tu borrador automáticamente. Se descargó la versión guardada para comparar.");}
 await persistir(true);
 }catch(e){mostrarError(e);}}
function incidenciaSeleccionada(){return incidencias.find(r=>r.id===incidenciaActual);}
function observacionActual(){const r=incidenciaSeleccionada();if(!r){$("observacion").textContent="Selecciona una incidencia para abrir su imagen.";return;}
 $("observacion").textContent=[`Prioridad ${r.prioridad} · ${r.imagen}`,`Estado: ${r.estado} · guardado: ${r.estado_guardado}`,r.motivo,
 `Objetos: ${r.objetos_actuales.map(a=>a.id+" ("+a.material+", "+a.contornos+" contornos)").join("; ")||"imagen completa / objeto omitido"}`,
 r.posible_omision?"Posible omisión: revisar otros residuos y regiones ignore.":"",r.estado==="no_vigente"?"El aviso técnico ya no se reproduce en la revisión actual; no se simuló una revisión humana.":r.bloqueante?"Bloquea la candidatura mientras siga pendiente.":"Pista opcional del modelo: no demuestra error de etiqueta.",r.resolucion_anterior_desactualizada?"La resolución anterior quedó desactualizada por otro guardado.":""].filter(Boolean).join("\n");}
async function refrescarPrioridades(navegar=false){if(!config)return;try{const r=await api("/api/prioridades");incidencias=r.incidencias;const fil=incidencias.filter(i=>(!$("filtro_prioridad").value||String(i.prioridad)===$("filtro_prioridad").value)&&(!$("filtro_estado").value||($("filtro_estado").value==="abiertas"?(i.bloqueante||$("filtro_prioridad").value==="7")&&["pendiente","requiere_decision"].includes(i.estado):i.estado===$("filtro_estado").value)));
 if(!fil.some(i=>i.id===incidenciaActual))incidenciaActual=(fil.find(i=>i.imagen===tarea?.imagen.id)||fil[0])?.id||null;
 $("incidencias").replaceChildren(...fil.map(i=>{const o=document.createElement("option");o.value=i.id;o.textContent=`P${i.prioridad} · ${i.imagen} · ${i.estado} · ${i.motivo.slice(0,85)}`;o.title=i.motivo;return o;}));$("incidencias").value=incidenciaActual||"";observacionActual();if(navegar&&incidenciaActual)await abrirIncidencia(incidenciaActual);
 }catch(e){mostrarError(e);}}
async function abrirIncidencia(id){const r=incidencias.find(i=>i.id===id);if(!r||busy)return;incidenciaActual=id;$("fuente").value="";opcionImagen();if(tarea?.imagen.id!==r.imagen)await cargar(r.imagen);
 const n=anns.findIndex(a=>r.objetos.includes(a.id));if(n>=0)seleccionar(n);observacionActual();}
function moverIncidencia(delta){const opciones=[...$("incidencias").options].map(o=>o.value);const n=opciones.indexOf(incidenciaActual);if(opciones[n+delta])abrirIncidencia(opciones[n+delta]);}
async function refrescarAuditoria(){if(!config)return;try{const r=await api("/api/auditoria"),s=r.resumen;
 $("estado_qa").textContent=r.ocupada?"Auditoría CPU en curso. Tus guardados y borradores se conservan.":r.error?"Auditoría no actualizada: "+r.error:r.vigente?"Auditoría vigente para estos guardados.":"Hay cambios o falta auditoría: actualizar antes de contar candidatas.";
 if(s)$("resumen_qa").textContent=`${s.imagenes_candidatas}/40 imágenes candidatas · ${s.anotaciones_validas_candidatas} polígonos válidos\nRegiones48: ${Object.entries(s.materiales_regiones_48).map(([c,n])=>c+" "+n+"/8").join(" · ")}\nComida HPI: ${s.regiones_alimentarias_HPI_utilizables}/8 · Aceptadas V26: 0 · Entrenamiento deshabilitado`;
 if(r.ocupada)setTimeout(()=>{refrescarAuditoria();refrescarPrioridades(false);},2000);
 }catch(e){$("estado_qa").textContent="No se pudo consultar la auditoría: "+e.message;}}
$("guardar").onclick=guardarSeguro;$("contexto").onclick=recargarContexto;$("emergencia").onclick=emergencia;
$("recuperar").onclick=()=>{try{aplicarBorrador(borradorRecuperable);}catch(e){mostrarError(e);}};
$("importar").onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());const s=d.estado||d.contenido||d;aplicarBorrador(s);}catch(e){mostrarError(e);}finally{$("importar").value="";}};
$("recuperar_servidor").onclick=async()=>{try{const rs=await api("/api/borradores?id="+encodeURIComponent(tarea.imagen.id));if(!rs.length)throw Error("No hay copia del servidor para esta imagen");
 const r=await api(`/api/borradores?id=${encodeURIComponent(tarea.imagen.id)}&archivo=${rs[0].archivo}`);const s=r.contenido.estado_local||r.contenido;
 if(!s.modo){descargar(r,"ecowaste_copia_cruda_servidor.json");throw Error("Copia del servidor sin contexto completo: se descargó para comparar, no se aplica automáticamente.");}aplicarBorrador(s);
 }catch(e){mostrarError(e);}};
$("abrir_guardado").onclick=async()=>{if(!tarea||busy)return;emergencia();try{await conservar();await cargar(tarea.imagen.id);}catch(e){mostrarError(e);}};
$("incidencias").onchange=()=>abrirIncidencia($("incidencias").value);$("filtro_prioridad").onchange=()=>refrescarPrioridades(true);$("filtro_estado").onchange=()=>refrescarPrioridades(false);
$("abrir_inc").onclick=()=>abrirIncidencia($("incidencias").value);
$("prev_inc").onclick=()=>moverIncidencia(-1);$("next_inc").onclick=()=>moverIncidencia(1);
$("resolver_inc").onclick=async()=>{try{const r=incidenciaSeleccionada();if(!r||r.imagen!==tarea?.imagen.id)throw Error("Abre primero la imagen de esta incidencia");if(dirty)throw Error("Guarda primero tus cambios; la resolución se vincula al evento guardado.");
 await api("/api/incidencia",{incidencia_id:r.id,estado:$("estado_inc").value,persona_id:$("persona").value.trim(),nota:$("nota_inc").value,revision_huella:tarea.revision_huella});
 $("nota_inc").value="";mensaje("Estado registrado sobre tu revisión. No es aceptación V26 ni revisión independiente.");await refrescarPrioridades(false);await refrescarAuditoria();
 }catch(e){mostrarError(e);}};
$("auditar").onclick=async()=>{try{await api("/api/auditar",{});await refrescarAuditoria();}catch(e){mostrarError(e);}};
const samBase=sam;
$("sam").onclick=async()=>{if(busy)return;recordar();const a=activo(),antes=a?.propuesta_ia_id;await persistir(true).catch(()=>{});congelar(true);try{await samBase();if(activo()?.propuesta_ia_id&&activo().propuesta_ia_id!==antes){puntos=[];etiquetas=[];caja=null;manual=[];indicacionesPorObjeto[activo().id]=copiarIndicaciones();}await persistir(true);}catch(e){mostrarError(e);}finally{congelar(false);}};
// Persistir tras las mutaciones del editor, incluso vértices, texto y polígonos sin cerrar.
document.addEventListener("input",e=>{if(!busy){if(tarea&&["motivo","material","completa"].includes(e.target.id))dirty=true;programarPersistencia();}});
document.addEventListener("change",e=>{if(!busy&&e.target.id==="completa"){dirty=true;programarPersistencia();}});
document.addEventListener("pointerup",()=>queueMicrotask(()=>{if(!busy)persistir().catch(()=>{});}));
document.addEventListener("click",()=>{if(!busy)programarPersistencia();});
window.addEventListener("pagehide",()=>{if(tarea&&hayTrabajo())try{localStorage.setItem("ecowaste_emergencia_"+claveLocal(),JSON.stringify(estadoLocal()));}catch{ /* IndexedDB y descarga siguen disponibles. */ }});
(async()=>{await window.iniciarEditor();await refrescarPrioridades(false);await refrescarAuditoria();setInterval(()=>{if(!busy&&!cargando){refrescarAuditoria();if(!hayTrabajo())refrescarPrioridades(false);}},15000);})();

"""
═══════════════════════════════════════════════════════════════════════════
 MOTOR DE ANÁLISIS COMPLETO — procesar_partido.py
═══════════════════════════════════════════════════════════════════════════
 Procesa el video UNA vez y genera:
   1. JSON para Blender (posiciones robots+balón por frame, con score)
   2. Heatmap de posiciones por equipo      → PNG + MP4
   3. Mapa de posesión (zonas de control)   → PNG + MP4
   4. Diagrama de Voronoi (territorio)       → PNG + MP4
   5. Grafo de interacción (eventos)         → PNG + MP4 + JSON

 Uso:
   python procesar_partido.py --video datos/ejemplo/partido.mp4 --nombre partido_final
 Requiere: core.py en el mismo directorio.
═══════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from scipy.spatial import Voronoi
from ultralytics import YOLO

from core import Config, Campo, TeamDetector, GrafoInteraccion, Metricas, COL

# ═══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS  ←  por línea de comandos (defaults relativos al repo)
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Motor de análisis de fútbol robótico")
parser.add_argument("--video", default=str(BASE_DIR / "datos" / "ejemplo" / "partido.mp4"),
                    help="Ruta al video del partido")
parser.add_argument("--nombre", default="partido_final",
                    help="Nombre de salida (sin extensión, sin espacios)")
parser.add_argument("--out", default=str(BASE_DIR / "resultados" / "renders"),
                    help="Directorio de salida")
args = parser.parse_args()

VIDEO         = args.video
NOMBRE_SALIDA = args.nombre
OUT_DIR       = args.out

if not os.path.exists(VIDEO):
    raise FileNotFoundError(
        f"No se encontró el video: {VIDEO}\n"
        f"Pasa la ruta con --video o coloca el archivo en datos/ejemplo/partido.mp4"
    )

DIR_BLENDER  = os.path.join(OUT_DIR, "blender")
DIR_ANALISIS = os.path.join(OUT_DIR, "analisis", NOMBRE_SALIDA)
os.makedirs(DIR_BLENDER, exist_ok=True)
os.makedirs(DIR_ANALISIS, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════
cfg   = Config()
campo = Campo(cfg)
model = YOLO(cfg.MODELO)        # best.pt portable (NO el .engine)

cap   = cv2.VideoCapture(VIDEO)
FPS   = cap.get(cv2.CAP_PROP_FPS)
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

teams = TeamDetector(cfg)
grafo = GrafoInteraccion(cfg, campo)
metricas = Metricas(cfg, FPS)
score = {"azul": 0, "amarillo": 0}
ball_prev = None
ultimo_gol = -cfg.COOLDOWN_GOL

W, H = cfg.CAMPO_W, cfg.CAMPO_H   # campo canónico en px

# ─── Acumuladores para los análisis ───
datos_blender = {"fps": FPS, "campo_cm": [cfg.CAMPO_CM_W, cfg.CAMPO_CM_H],
                 "escala": cfg.ESCALA, "frames": []}
heat = {"azul": np.zeros((H, W), np.float32),
        "amarillo": np.zeros((H, W), np.float32)}
posesion_frames = []        # equipo que posee el balón por frame
historial = []              # posiciones canónicas por frame (para MP4s)

# ─── Carpetas para las secuencias PNG que consume Blender (proyección en césped) ───
DIR_HEAT_SEQ = os.path.join(DIR_ANALISIS, "heatmap_seq")   # PNG transparente por frame
DIR_VORO_SEQ = os.path.join(DIR_ANALISIS, "voronoi_seq")   # PNG transparente por frame
os.makedirs(DIR_HEAT_SEQ, exist_ok=True)
os.makedirs(DIR_VORO_SEQ, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PASE ÚNICO POR EL VIDEO
# ═══════════════════════════════════════════════════════════════════════════
print(f"Procesando '{NOMBRE_SALIDA}' — {TOTAL} frames (un solo pase)...")

for fidx, result in enumerate(
        model.track(source=VIDEO, conf=cfg.CONF, stream=True,
                    tracker="bytetrack.yaml", imgsz=cfg.IMG_SZ,
                    verbose=False, persist=True), start=1):

    if fidx % 100 == 0:
        print(f"  {fidx}/{TOTAL}")

    robots_bl = {}            # para JSON Blender (cm)
    pos_canon = {}            # posiciones canónicas px (para análisis)
    ball = None
    ball_canon = None

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        tid = int(box.id[0]) if box.id is not None else -1
        cx_c, cy_c = campo.proyectar(cx, cy)

        if name == "Robot" and tid >= 1:
            pos_canon[tid] = (cx_c, cy_c)
            metricas.actualizar(tid, cx_c, cy_c)
            x_cm, y_cm = cx_c/cfg.ESCALA, cy_c/cfg.ESCALA
            ex1, ey1 = campo.proyectar(x1, y1)
            ex2, ey2 = campo.proyectar(x2, y2)
            robots_bl[str(tid)] = {
                "pos":  [round(x_cm,1), round(y_cm,1)],
                "size": [round(abs(ex2-ex1)/cfg.ESCALA,1), round(abs(ey2-ey1)/cfg.ESCALA,1)]
            }
        elif name == "Ball":
            cx_c2, cy_c2 = campo.proyectar(cx, cy)
            ball_canon = (cx_c2, cy_c2)
            ball = [round(cx_c2/cfg.ESCALA,1), round(cy_c2/cfg.ESCALA,1)]

    teams.actualizar(pos_canon)

    # ── Acumular heatmap por equipo ──
    for rid, (px, py) in pos_canon.items():
        eq = teams.get_equipo(rid)
        if eq in heat and 0 <= py < H and 0 <= px < W:
            heat[eq][py, px] += 1

    # ── Grafo de interacción (usa tu lógica de core.py) ──
    aristas = grafo.actualizar(fidx, pos_canon, ball_canon, score)

    # ── Posesión: qué equipo tiene el balón este frame ──
    eq_posesion = None
    if grafo.posesion is not None:
        eq_posesion = teams.get_equipo(grafo.posesion)
    posesion_frames.append(eq_posesion)

    # ── Detección de goles (para el marcador) ──
    if ball_canon is not None and fidx - ultimo_gol > cfg.COOLDOWN_GOL:
        bx, by = ball_canon
        en_zona = None
        if campo.goal_x1 < bx < campo.goal_x2:
            if by >= campo.goal_bot: en_zona = "azul"
            elif by <= campo.goal_top: en_zona = "amarillo"
        estaba = False
        if ball_prev is not None:
            pbx, pby = ball_prev
            if campo.goal_x1 < pbx < campo.goal_x2:
                if pby >= campo.goal_bot or pby <= campo.goal_top:
                    estaba = True
        if en_zona and not estaba:
            anotador = "amarillo" if en_zona == "azul" else "azul"
            score[anotador] += 1
            ultimo_gol = fidx
    ball_prev = ball_canon if ball_canon else ball_prev

    # ── Guardar para Blender ──
    datos_blender["frames"].append({
        "frame": fidx, "robots": robots_bl, "ball": ball, "score": dict(score)
    })

    # ── Guardar snapshot para los MP4 ──
    historial.append({
        "pos": dict(pos_canon),
        "ball": ball_canon,
        "equipos": dict(teams.equipos),
        "posesion": eq_posesion,
        "aristas": aristas,
        "score": dict(score),
    })

datos_blender["equipos"] = {str(rid): eq for rid, eq in teams.equipos.items()}
print(f"Pase completo: {len(datos_blender['frames'])} frames procesados")

# ═══════════════════════════════════════════════════════════════════════════
#  1. JSON PARA BLENDER (con interpolación de huecos)
# ═══════════════════════════════════════════════════════════════════════════
def interpolar(datos):
    frames = datos["frames"]
    todos = set()
    for fr in frames: todos.update(fr["robots"].keys())
    for rid in todos:
        idxv = [i for i,fr in enumerate(frames) if rid in fr["robots"]]
        if len(idxv) < 2: continue
        for k in range(len(idxv)-1):
            i0, i1 = idxv[k], idxv[k+1]
            if i1-i0 <= 1: continue
            p0 = frames[i0]["robots"][rid]; p1 = frames[i1]["robots"][rid]
            pos0 = p0["pos"] if isinstance(p0,dict) else p0
            pos1 = p1["pos"] if isinstance(p1,dict) else p1
            size = p0["size"] if isinstance(p0,dict) else None
            for i in range(i0+1, i1):
                t = (i-i0)/(i1-i0)
                e = {"pos":[round(pos0[0]+(pos1[0]-pos0[0])*t,1),
                            round(pos0[1]+(pos1[1]-pos0[1])*t,1)]}
                if size: e["size"] = size
                e["interpolado"] = True
                frames[i]["robots"][rid] = e
    return datos

datos_blender = interpolar(datos_blender)
json_blender = os.path.join(DIR_BLENDER, f"{NOMBRE_SALIDA}.json")
with open(json_blender, "w", encoding="utf-8") as f:
    json.dump(datos_blender, f, ensure_ascii=False, indent=2)
print(f"[1/5] JSON Blender: {json_blender}")

# ═══════════════════════════════════════════════════════════════════════════
#  2. HEATMAP por equipo  (PNG + MP4)
# ═══════════════════════════════════════════════════════════════════════════
def colorize_heat(acc):
    """Convierte un acumulador en imagen de calor coloreada sobre el campo."""
    blur = cv2.GaussianBlur(acc, (0,0), cfg.SIGMA)
    if blur.max() > 0:
        norm = (blur/blur.max()*255).astype(np.uint8)
    else:
        norm = blur.astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    base = campo.nuevo_canvas()
    off = campo.GD
    region = base[off:off+H, 0:W]
    mezcla = cv2.addWeighted(region, 0.45, cmap, 0.55, 0)
    base[off:off+H, 0:W] = mezcla
    return base

# PNG resumen (acumulado total de cada equipo, lado a lado)
heat_azul = colorize_heat(heat["azul"])
heat_amar = colorize_heat(heat["amarillo"])
cv2.putText(heat_azul, "AZUL", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,180,50), 2)
cv2.putText(heat_amar, "AMARILLO", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,255), 2)
heat_png = cv2.hconcat([heat_azul, heat_amar])
cv2.imwrite(os.path.join(DIR_ANALISIS, "heatmap.png"), heat_png)

# MP4 animado (heatmap que se va construyendo con el tiempo)
Hc = cfg.CAMPO_H_TOTAL
vw_heat = cv2.VideoWriter(os.path.join(DIR_ANALISIS, "heatmap.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, Hc))
heat_run = {"azul": np.zeros((H,W),np.float32), "amarillo": np.zeros((H,W),np.float32)}
for snap in historial:
    for rid,(px,py) in snap["pos"].items():
        eq = snap["equipos"].get(rid, "unknown")
        if eq in heat_run and 0<=py<H and 0<=px<W:
            heat_run[eq][py,px] += 1
    combo = heat_run["azul"] + heat_run["amarillo"]
    vw_heat.write(colorize_heat(combo))
vw_heat.release()
print(f"[2/5] Heatmap: heatmap.png + heatmap.mp4")

# ─── Secuencia PNG transparente del heatmap (para proyectar en césped Blender) ───
def heatmap_png_transparente(acc):
    """Heatmap coloreado SOLO en el área de juego (546×729), con alfa.
       Zonas sin actividad = transparentes. Sin offset de portería."""
    blur = cv2.GaussianBlur(acc, (0,0), cfg.SIGMA)
    if blur.max() > 0:
        norm = (blur/blur.max()*255).astype(np.uint8)
    else:
        norm = blur.astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)          # BGR
    bgra = cv2.cvtColor(cmap, cv2.COLOR_BGR2BGRA)
    # alfa proporcional a la intensidad: frío = transparente, caliente = opaco
    bgra[:, :, 3] = norm
    return bgra

print("   Exportando secuencia PNG del heatmap...")
heat_seq = {"azul": np.zeros((H,W),np.float32), "amarillo": np.zeros((H,W),np.float32)}
for i, snap in enumerate(historial, start=1):
    for rid,(px,py) in snap["pos"].items():
        eq = snap["equipos"].get(rid, "unknown")
        if eq in heat_seq and 0<=py<H and 0<=px<W:
            heat_seq[eq][py,px] += 1
    combo = heat_seq["azul"] + heat_seq["amarillo"]
    png = heatmap_png_transparente(combo)
    cv2.imwrite(os.path.join(DIR_HEAT_SEQ, f"heat_{i:05d}.png"), png)
print(f"   {len(historial)} PNG -> {DIR_HEAT_SEQ}")

# ═══════════════════════════════════════════════════════════════════════════
#  3. MAPA DE POSESIÓN  (PNG + MP4)
# ═══════════════════════════════════════════════════════════════════════════
n_total = len([p for p in posesion_frames if p])
pos_azul = posesion_frames.count("azul")
pos_amar = posesion_frames.count("amarillo")
pct_azul = 100*pos_azul/max(n_total,1)
pct_amar = 100*pos_amar/max(n_total,1)

def barra_posesion(canvas, pa, pm):
    h = 40; y0 = 10
    wa = int(W * pa/100)
    cv2.rectangle(canvas, (0,y0), (wa,y0+h), (255,180,50), -1)
    cv2.rectangle(canvas, (wa,y0), (W,y0+h), (50,220,255), -1)
    cv2.putText(canvas, f"AZUL {pa:.0f}%", (10,y0+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(canvas, f"{pm:.0f}% AMARILLO", (W-180,y0+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# PNG resumen
pose_png = campo.nuevo_canvas()
barra_posesion(pose_png, pct_azul, pct_amar)
cv2.imwrite(os.path.join(DIR_ANALISIS, "posesion.png"), pose_png)

# MP4 animado (barra de posesión acumulada en el tiempo)
vw_pose = cv2.VideoWriter(os.path.join(DIR_ANALISIS, "posesion.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, Hc))
ca = cm_ = 0
for i, snap in enumerate(historial):
    if snap["posesion"] == "azul": ca += 1
    elif snap["posesion"] == "amarillo": cm_ += 1
    tot = max(ca+cm_, 1)
    frame = campo.nuevo_canvas()
    barra_posesion(frame, 100*ca/tot, 100*cm_/tot)
    # dibujar robots y balón
    for rid,(px,py) in snap["pos"].items():
        col = COL.get(snap["equipos"].get(rid,"unknown"))
        cv2.circle(frame, (px, campo.y_canvas(py)), 8, col, -1)
    if snap["ball"]:
        bx,by = snap["ball"]
        cv2.circle(frame, (bx, campo.y_canvas(by)), 6, COL["ball"], -1)
    vw_pose.write(frame)
vw_pose.release()
print(f"[3/5] Posesión: posesion.png + posesion.mp4  (Azul {pct_azul:.0f}% / Amarillo {pct_amar:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
#  4. DIAGRAMA DE VORONOI  (PNG + MP4)
# ═══════════════════════════════════════════════════════════════════════════
def dibujar_voronoi(canvas, pos, equipos):
    """Colorea el territorio dominado por cada equipo (Voronoi por celdas)."""
    pts, cols = [], []
    for rid,(px,py) in pos.items():
        pts.append([px, py])
        eq = equipos.get(rid, "unknown")
        cols.append(COL.get(eq, COL["unknown"]))
    if len(pts) < 2:
        return canvas
    # Para Voronoi con pocos puntos: pintar cada pixel según robot más cercano
    pts_arr = np.array(pts)
    off = campo.GD
    # malla reducida para velocidad
    step = 6
    overlay = canvas.copy()
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            d = np.hypot(pts_arr[:,0]-xx, pts_arr[:,1]-yy)
            idx = int(np.argmin(d))
            cv2.rectangle(overlay, (xx, yy+off), (xx+step, yy+off+step),
                          cols[idx], -1)
    out = cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0)
    campo.dibujar(out)
    for (px,py), c in zip(pts, cols):
        cv2.circle(out, (px, campo.y_canvas(py)), 7, (255,255,255), -1)
        cv2.circle(out, (px, campo.y_canvas(py)), 7, c, 2)
    return out

# PNG resumen: Voronoi en el frame de mayor actividad (más robots visibles)
mejor = max(historial, key=lambda s: len(s["pos"]))
voro_png = dibujar_voronoi(campo.nuevo_canvas(), mejor["pos"], mejor["equipos"])
cv2.imwrite(os.path.join(DIR_ANALISIS, "voronoi.png"), voro_png)

# MP4 animado
vw_voro = cv2.VideoWriter(os.path.join(DIR_ANALISIS, "voronoi.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, Hc))
for snap in historial:
    vw_voro.write(dibujar_voronoi(campo.nuevo_canvas(), snap["pos"], snap["equipos"]))
vw_voro.release()
print(f"[4/5] Voronoi: voronoi.png + voronoi.mp4")

# ─── Secuencia PNG transparente del Voronoi (para proyectar en césped Blender) ───
def voronoi_png_transparente(pos, equipos):
    """Territorio dominado SOLO en el área de juego (546×729), con alfa.
       Sin robots ni líneas (eso ya está en 3D). Semi-transparente uniforme."""
    bgra = np.zeros((H, W, 4), np.uint8)
    pts, cols = [], []
    for rid,(px,py) in pos.items():
        pts.append([px, py])
        eq = equipos.get(rid, "unknown")
        cols.append(COL.get(eq, COL["unknown"]))
    if len(pts) < 2:
        return bgra   # todo transparente
    pts_arr = np.array(pts)
    step = 6
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            d = np.hypot(pts_arr[:,0]-xx, pts_arr[:,1]-yy)
            idx = int(np.argmin(d))
            c = cols[idx]
            cv2.rectangle(bgra, (xx, yy), (xx+step, yy+step),
                          (c[0], c[1], c[2], 130), -1)   # alfa fijo semi-transparente
    return bgra

print("   Exportando secuencia PNG del Voronoi...")
for i, snap in enumerate(historial, start=1):
    png = voronoi_png_transparente(snap["pos"], snap["equipos"])
    cv2.imwrite(os.path.join(DIR_VORO_SEQ, f"voro_{i:05d}.png"), png)
print(f"   {len(historial)} PNG -> {DIR_VORO_SEQ}")

# ═══════════════════════════════════════════════════════════════════════════
#  5. GRAFO DE INTERACCIÓN  (PNG + MP4 + JSON)
# ═══════════════════════════════════════════════════════════════════════════
# JSON de eventos (tu core.py ya lo arma)
grafo.guardar_json(os.path.join(DIR_ANALISIS, "grafo_eventos.json"))

# MP4 animado de aristas (pases/colisiones/disparos/posesión)
vw_graf = cv2.VideoWriter(os.path.join(DIR_ANALISIS, "grafo.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, Hc))
ult_frame_graf = campo.nuevo_canvas()
for snap in historial:
    frame = campo.nuevo_canvas()
    # robots
    for rid,(px,py) in snap["pos"].items():
        col = COL.get(snap["equipos"].get(rid,"unknown"))
        cv2.circle(frame, (px, campo.y_canvas(py)), 8, col, -1)
        cv2.putText(frame, str(rid), (px-5, campo.y_canvas(py)+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    if snap["ball"]:
        bx,by = snap["ball"]
        cv2.circle(frame, (bx, campo.y_canvas(by)), 6, COL["ball"], -1)
    # aristas del frame
    grafo.dibujar(frame, snap["aristas"], snap["pos"], snap["ball"])
    # marcador
    s = snap["score"]
    cv2.putText(frame, f"{s['azul']} - {s['amarillo']}", (W//2-40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    vw_graf.write(frame)
    ult_frame_graf = frame
vw_graf.release()
cv2.imwrite(os.path.join(DIR_ANALISIS, "grafo.png"), ult_frame_graf)
print(f"[5/5] Grafo: grafo.png + grafo.mp4 + grafo_eventos.json")

# ═══════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════
metricas.resumen()
print("\n" + "="*60)
print(f"PARTIDO '{NOMBRE_SALIDA}' COMPLETO")
print("="*60)
print(f"   Marcador final: Azul {score['azul']} - {score['amarillo']} Amarillo")
print(f"   Posesión: Azul {pct_azul:.0f}% / Amarillo {pct_amar:.0f}%")
print(f"\n   Blender: {json_blender}")
print(f"   Análisis: {DIR_ANALISIS}")
print(f"      heatmap.png/mp4 · posesion.png/mp4 · voronoi.png/mp4 · grafo.png/mp4")
print(f"   Secuencias para césped 3D:")
print(f"      {DIR_HEAT_SEQ}  ({len(historial)} PNG)")
print(f"      {DIR_VORO_SEQ}  ({len(historial)} PNG)")
print("="*60)
print(f"\nTODO LISTO (local)")
print(f"   Abre Blender en esta laptop y carga el partido '{NOMBRE_SALIDA}'.")
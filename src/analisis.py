"""
src/analisis.py — Generación de análisis tácticos
==================================================
A partir del historial de frames (posiciones, equipos, posesión, eventos),
genera todas las salidas visuales:
    - Heatmap por equipo            (PNG + MP4 + secuencia PNG para Blender)
    - Mapa de posesión              (PNG + MP4)
    - Diagrama de Voronoi           (PNG + MP4 + secuencia PNG para Blender)
    - Grafo de interacción          (PNG + MP4 + JSON)

Cada función recibe el campo, el historial y las rutas de salida.
"""

import os
import cv2
import numpy as np

from .config import Config, COL


# ═══════════════════════════════════════════════════════════════════════════
#  HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def _colorize_heat(cfg, campo, acc, H, W):
    """Convierte un acumulador en imagen de calor coloreada sobre el campo."""
    blur = cv2.GaussianBlur(acc, (0,0), cfg.SIGMA)
    norm = (blur/blur.max()*255).astype(np.uint8) if blur.max() > 0 else blur.astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    base = campo.nuevo_canvas()
    off = campo.GD
    region = base[off:off+H, 0:W]
    mezcla = cv2.addWeighted(region, 0.45, cmap, 0.55, 0)
    base[off:off+H, 0:W] = mezcla
    return base


def _heatmap_png_transparente(cfg, acc, H, W):
    """Heatmap con alfa proporcional a la intensidad (para proyectar en césped)."""
    blur = cv2.GaussianBlur(acc, (0,0), cfg.SIGMA)
    norm = (blur/blur.max()*255).astype(np.uint8) if blur.max() > 0 else blur.astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    bgra = cv2.cvtColor(cmap, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = norm
    return bgra


def generar_heatmap(cfg, campo, heat, historial, fps, dir_analisis, dir_heat_seq):
    H, W = cfg.CAMPO_H, cfg.CAMPO_W
    Hc = cfg.CAMPO_H_TOTAL

    # PNG resumen (acumulado total por equipo, lado a lado)
    heat_azul = _colorize_heat(cfg, campo, heat["azul"], H, W)
    heat_amar = _colorize_heat(cfg, campo, heat["amarillo"], H, W)
    cv2.putText(heat_azul, "AZUL", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,180,50), 2)
    cv2.putText(heat_amar, "AMARILLO", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,255), 2)
    cv2.imwrite(os.path.join(dir_analisis, "heatmap.png"), cv2.hconcat([heat_azul, heat_amar]))

    # MP4 animado (heatmap acumulándose en el tiempo)
    vw = cv2.VideoWriter(os.path.join(dir_analisis, "heatmap.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hc))
    run = {"azul": np.zeros((H,W),np.float32), "amarillo": np.zeros((H,W),np.float32)}
    for snap in historial:
        for rid,(px,py) in snap["pos"].items():
            eq = snap["equipos"].get(rid, "unknown")
            if eq in run and 0<=py<H and 0<=px<W:
                run[eq][py,px] += 1
        vw.write(_colorize_heat(cfg, campo, run["azul"]+run["amarillo"], H, W))
    vw.release()

    # Secuencia PNG transparente (para proyectar en césped Blender)
    seq = {"azul": np.zeros((H,W),np.float32), "amarillo": np.zeros((H,W),np.float32)}
    for i, snap in enumerate(historial, start=1):
        for rid,(px,py) in snap["pos"].items():
            eq = snap["equipos"].get(rid, "unknown")
            if eq in seq and 0<=py<H and 0<=px<W:
                seq[eq][py,px] += 1
        png = _heatmap_png_transparente(cfg, seq["azul"]+seq["amarillo"], H, W)
        cv2.imwrite(os.path.join(dir_heat_seq, f"heat_{i:05d}.png"), png)

    print(f"[2/5] Heatmap: heatmap.png + heatmap.mp4 + {len(historial)} PNG seq")


# ═══════════════════════════════════════════════════════════════════════════
#  POSESIÓN
# ═══════════════════════════════════════════════════════════════════════════
def _barra_posesion(canvas, pa, pm, W):
    h = 40; y0 = 10
    wa = int(W * pa/100)
    cv2.rectangle(canvas, (0,y0), (wa,y0+h), (255,180,50), -1)
    cv2.rectangle(canvas, (wa,y0), (W,y0+h), (50,220,255), -1)
    cv2.putText(canvas, f"AZUL {pa:.0f}%", (10,y0+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(canvas, f"{pm:.0f}% AMARILLO", (W-180,y0+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


def generar_posesion(cfg, campo, historial, posesion_frames, fps, dir_analisis):
    H, W = cfg.CAMPO_H, cfg.CAMPO_W
    Hc = cfg.CAMPO_H_TOTAL

    n_total = len([p for p in posesion_frames if p])
    pos_azul = posesion_frames.count("azul")
    pos_amar = posesion_frames.count("amarillo")
    pct_azul = 100*pos_azul/max(n_total,1)
    pct_amar = 100*pos_amar/max(n_total,1)

    pose_png = campo.nuevo_canvas()
    _barra_posesion(pose_png, pct_azul, pct_amar, W)
    cv2.imwrite(os.path.join(dir_analisis, "posesion.png"), pose_png)

    vw = cv2.VideoWriter(os.path.join(dir_analisis, "posesion.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hc))
    ca = cm_ = 0
    for snap in historial:
        if snap["posesion"] == "azul": ca += 1
        elif snap["posesion"] == "amarillo": cm_ += 1
        tot = max(ca+cm_, 1)
        frame = campo.nuevo_canvas()
        _barra_posesion(frame, 100*ca/tot, 100*cm_/tot, W)
        for rid,(px,py) in snap["pos"].items():
            col = COL.get(snap["equipos"].get(rid,"unknown"))
            cv2.circle(frame, (px, campo.y_canvas(py)), 8, col, -1)
        if snap["ball"]:
            bx,by = snap["ball"]
            cv2.circle(frame, (bx, campo.y_canvas(by)), 6, COL["ball"], -1)
        vw.write(frame)
    vw.release()
    print(f"[3/5] Posesión: posesion.png + posesion.mp4 (Azul {pct_azul:.0f}% / Amarillo {pct_amar:.0f}%)")
    return pct_azul, pct_amar


# ═══════════════════════════════════════════════════════════════════════════
#  VORONOI
# ═══════════════════════════════════════════════════════════════════════════
def _dibujar_voronoi(cfg, campo, canvas, pos, equipos, H, W):
    pts, cols = [], []
    for rid,(px,py) in pos.items():
        pts.append([px, py])
        cols.append(COL.get(equipos.get(rid, "unknown"), COL["unknown"]))
    if len(pts) < 2:
        return canvas
    pts_arr = np.array(pts)
    off = campo.GD
    step = 6
    overlay = canvas.copy()
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            d = np.hypot(pts_arr[:,0]-xx, pts_arr[:,1]-yy)
            idx = int(np.argmin(d))
            cv2.rectangle(overlay, (xx, yy+off), (xx+step, yy+off+step), cols[idx], -1)
    out = cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0)
    campo.dibujar(out)
    for (px,py), c in zip(pts, cols):
        cv2.circle(out, (px, campo.y_canvas(py)), 7, (255,255,255), -1)
        cv2.circle(out, (px, campo.y_canvas(py)), 7, c, 2)
    return out


def _voronoi_png_transparente(cfg, pos, equipos, H, W):
    bgra = np.zeros((H, W, 4), np.uint8)
    pts, cols = [], []
    for rid,(px,py) in pos.items():
        pts.append([px, py])
        cols.append(COL.get(equipos.get(rid, "unknown"), COL["unknown"]))
    if len(pts) < 2:
        return bgra
    pts_arr = np.array(pts)
    step = 6
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            d = np.hypot(pts_arr[:,0]-xx, pts_arr[:,1]-yy)
            idx = int(np.argmin(d))
            c = cols[idx]
            cv2.rectangle(bgra, (xx, yy), (xx+step, yy+step), (c[0], c[1], c[2], 130), -1)
    return bgra


def generar_voronoi(cfg, campo, historial, fps, dir_analisis, dir_voro_seq):
    H, W = cfg.CAMPO_H, cfg.CAMPO_W
    Hc = cfg.CAMPO_H_TOTAL

    mejor = max(historial, key=lambda s: len(s["pos"]))
    voro_png = _dibujar_voronoi(cfg, campo, campo.nuevo_canvas(), mejor["pos"], mejor["equipos"], H, W)
    cv2.imwrite(os.path.join(dir_analisis, "voronoi.png"), voro_png)

    vw = cv2.VideoWriter(os.path.join(dir_analisis, "voronoi.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hc))
    for snap in historial:
        vw.write(_dibujar_voronoi(cfg, campo, campo.nuevo_canvas(), snap["pos"], snap["equipos"], H, W))
    vw.release()

    for i, snap in enumerate(historial, start=1):
        png = _voronoi_png_transparente(cfg, snap["pos"], snap["equipos"], H, W)
        cv2.imwrite(os.path.join(dir_voro_seq, f"voro_{i:05d}.png"), png)

    print(f"[4/5] Voronoi: voronoi.png + voronoi.mp4 + {len(historial)} PNG seq")


# ═══════════════════════════════════════════════════════════════════════════
#  GRAFO (video de eventos)
# ═══════════════════════════════════════════════════════════════════════════
def generar_grafo(cfg, campo, grafo, historial, fps, dir_analisis):
    H, W = cfg.CAMPO_H, cfg.CAMPO_W
    Hc = cfg.CAMPO_H_TOTAL

    grafo.guardar_json(os.path.join(dir_analisis, "grafo_eventos.json"))

    vw = cv2.VideoWriter(os.path.join(dir_analisis, "grafo.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hc))
    ult = campo.nuevo_canvas()
    for snap in historial:
        frame = campo.nuevo_canvas()
        for rid,(px,py) in snap["pos"].items():
            col = COL.get(snap["equipos"].get(rid,"unknown"))
            cv2.circle(frame, (px, campo.y_canvas(py)), 8, col, -1)
            cv2.putText(frame, str(rid), (px-5, campo.y_canvas(py)+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        if snap["ball"]:
            bx,by = snap["ball"]
            cv2.circle(frame, (bx, campo.y_canvas(by)), 6, COL["ball"], -1)
        grafo.dibujar(frame, snap["aristas"], snap["pos"], snap["ball"])
        s = snap["score"]
        cv2.putText(frame, f"{s['azul']} - {s['amarillo']}", (W//2-40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        vw.write(frame)
        ult = frame
    vw.release()
    cv2.imwrite(os.path.join(dir_analisis, "grafo.png"), ult)
    print(f"[5/5] Grafo: grafo.png + grafo.mp4 + grafo_eventos.json")
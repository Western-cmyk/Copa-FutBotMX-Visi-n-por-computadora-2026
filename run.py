"""
run.py — Punto de entrada del análisis de fútbol robótico (Copa FutBotMX 2026)
==============================================================================
Flujo LINEAL del pipeline, paso a paso:

    1. Configuración y rutas
    2. Inicialización (modelo, campo, detectores)
    3. Pase único por el video (detección + tracking + análisis en vivo)
    4. Generación del JSON para Blender (con interpolación)
    5. Generación de los análisis tácticos (heatmap, posesión, voronoi, grafo)
    6. Resumen final

Uso:
    python run.py --video datos/ejemplo/partido.mp4 --nombre partido_final

Requiere los módulos en src/ y la config de tracking en config/.
"""

import os
import json
import argparse
from pathlib import Path

from src.config import Config, COL
from src.campo import Campo
from src.deteccion import Detector
from src.equipos import TeamDetector
from src.metricas import Metricas
from src.grafo import GrafoInteraccion
from src import analisis

RAIZ = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
#  INTERPOLACIÓN DE HUECOS (para el JSON de Blender)
# ═══════════════════════════════════════════════════════════════════════════
def interpolar(datos):
    frames = datos["frames"]
    todos = set()
    for fr in frames:
        todos.update(fr["robots"].keys())
    for rid in todos:
        idxv = [i for i, fr in enumerate(frames) if rid in fr["robots"]]
        if len(idxv) < 2:
            continue
        for k in range(len(idxv)-1):
            i0, i1 = idxv[k], idxv[k+1]
            if i1-i0 <= 1:
                continue
            p0 = frames[i0]["robots"][rid]; p1 = frames[i1]["robots"][rid]
            pos0 = p0["pos"] if isinstance(p0, dict) else p0
            pos1 = p1["pos"] if isinstance(p1, dict) else p1
            size = p0["size"] if isinstance(p0, dict) else None
            for i in range(i0+1, i1):
                t = (i-i0)/(i1-i0)
                e = {"pos": [round(pos0[0]+(pos1[0]-pos0[0])*t, 1),
                            round(pos0[1]+(pos1[1]-pos0[1])*t, 1)]}
                if size:
                    e["size"] = size
                e["interpolado"] = True
                frames[i]["robots"][rid] = e
    return datos


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Configuración y rutas ──
    parser = argparse.ArgumentParser(description="Análisis de fútbol robótico")
    parser.add_argument("--video", default=str(RAIZ / "datos" / "ejemplo" / "partido.mp4"),
                        help="Ruta al video del partido")
    parser.add_argument("--nombre", default="partido_final",
                        help="Nombre de salida (sin extensión, sin espacios)")
    parser.add_argument("--out", default=str(RAIZ / "resultados" / "renders"),
                        help="Directorio de salida")
    args = parser.parse_args()

    VIDEO, NOMBRE, OUT_DIR = args.video, args.nombre, args.out

    if not os.path.exists(VIDEO):
        raise FileNotFoundError(
            f"No se encontró el video: {VIDEO}\n"
            f"Pasa la ruta con --video o coloca el archivo en datos/ejemplo/partido.mp4"
        )

    dir_blender  = os.path.join(OUT_DIR, "blender")
    dir_analisis = os.path.join(OUT_DIR, "analisis", NOMBRE)
    dir_heat_seq = os.path.join(dir_analisis, "heatmap_seq")
    dir_voro_seq = os.path.join(dir_analisis, "voronoi_seq")
    for d in (dir_blender, dir_analisis, dir_heat_seq, dir_voro_seq):
        os.makedirs(d, exist_ok=True)

    # ── 2. Inicialización ──
    cfg      = Config()
    campo    = Campo(cfg)
    detector = Detector(cfg)
    teams    = TeamDetector(cfg)
    grafo    = GrafoInteraccion(cfg, campo)

    total, FPS = detector.contar_frames(VIDEO)
    metricas   = Metricas(cfg, FPS)

    W, H = cfg.CAMPO_W, cfg.CAMPO_H
    score = {"azul": 0, "amarillo": 0}
    ball_prev = None
    ultimo_gol = -cfg.COOLDOWN_GOL

    # Acumuladores
    datos_blender = {"fps": FPS, "campo_cm": [cfg.CAMPO_CM_W, cfg.CAMPO_CM_H],
                     "escala": cfg.ESCALA, "frames": []}
    heat = {"azul": __import__("numpy").zeros((H, W), "float32"),
            "amarillo": __import__("numpy").zeros((H, W), "float32")}
    posesion_frames = []
    historial = []

    # ── 3. Pase único por el video ──
    print(f"Procesando '{NOMBRE}' — {total} frames (un solo pase)...")
    for fidx, dets in detector.rastrear(VIDEO):
        if fidx % 100 == 0:
            print(f"  {fidx}/{total}")

        robots_bl = {}
        pos_canon = {}
        ball = None
        ball_canon = None

        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = d["centro"]
            tid = d["id"]
            cx_c, cy_c = campo.proyectar(cx, cy)

            if d["clase"] == "Robot" and tid >= 1:
                pos_canon[tid] = (cx_c, cy_c)
                metricas.actualizar(tid, cx_c, cy_c)
                x_cm, y_cm = cx_c/cfg.ESCALA, cy_c/cfg.ESCALA
                ex1, ey1 = campo.proyectar(x1, y1)
                ex2, ey2 = campo.proyectar(x2, y2)
                robots_bl[str(tid)] = {
                    "pos":  [round(x_cm, 1), round(y_cm, 1)],
                    "size": [round(abs(ex2-ex1)/cfg.ESCALA, 1), round(abs(ey2-ey1)/cfg.ESCALA, 1)]
                }
            elif d["clase"] == "Ball":
                bx_c, by_c = campo.proyectar(cx, cy)
                ball_canon = (bx_c, by_c)
                ball = [round(bx_c/cfg.ESCALA, 1), round(by_c/cfg.ESCALA, 1)]

        teams.actualizar(pos_canon)

        # Heatmap acumulado
        for rid, (px, py) in pos_canon.items():
            eq = teams.get_equipo(rid)
            if eq in heat and 0 <= py < H and 0 <= px < W:
                heat[eq][py, px] += 1

        # Grafo de interacción
        aristas = grafo.actualizar(fidx, pos_canon, ball_canon, score)

        # Posesión
        eq_posesion = None
        if grafo.posesion is not None:
            eq_posesion = teams.get_equipo(grafo.posesion)
        posesion_frames.append(eq_posesion)

        # Goles
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

        # Guardar para Blender
        datos_blender["frames"].append({
            "frame": fidx, "robots": robots_bl, "ball": ball, "score": dict(score)
        })

        # Snapshot para los MP4
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

    # ── 4. JSON para Blender ──
    datos_blender = interpolar(datos_blender)
    json_blender = os.path.join(dir_blender, f"{NOMBRE}.json")
    with open(json_blender, "w", encoding="utf-8") as f:
        json.dump(datos_blender, f, ensure_ascii=False, indent=2)
    print(f"[1/5] JSON Blender: {json_blender}")

    # ── 5. Análisis tácticos ──
    analisis.generar_heatmap(cfg, campo, heat, historial, FPS, dir_analisis, dir_heat_seq)
    pct_azul, pct_amar = analisis.generar_posesion(cfg, campo, historial, posesion_frames, FPS, dir_analisis)
    analisis.generar_voronoi(cfg, campo, historial, FPS, dir_analisis, dir_voro_seq)
    analisis.generar_grafo(cfg, campo, grafo, historial, FPS, dir_analisis)

    # ── 6. Resumen final ──
    metricas.resumen()
    print("\n" + "="*60)
    print(f"PARTIDO '{NOMBRE}' COMPLETO")
    print("="*60)
    print(f"   Marcador final: Azul {score['azul']} - {score['amarillo']} Amarillo")
    print(f"   Posesión: Azul {pct_azul:.0f}% / Amarillo {pct_amar:.0f}%")
    print(f"   Blender JSON: {json_blender}")
    print(f"   Análisis: {dir_analisis}")
    print("="*60)


if __name__ == "__main__":
    main()
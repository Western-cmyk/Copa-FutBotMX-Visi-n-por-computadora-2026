"""
core.py — Módulo central del sistema de análisis de fútbol de robots
====================================================================
Contiene toda la lógica compartida: configuración, homografía, dibujo
del campo, detección de equipos, grafos de interacción y métricas.

Importar desde los scripts de análisis:
    from core import Config, Campo, TeamDetector, GrafoInteraccion, Metricas
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# Directorio base = ubicación de este archivo (raíz del repo)
BASE_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Rutas (relativas al repo) — se cargan con best.pt portable, NO el .engine
    MODELO: str = str(BASE_DIR / "modelo" / "best.pt")
    H_PATH: str = str(BASE_DIR / "modelo" / "homografia.npy")

    # Campo real (cm) y escala
    CAMPO_CM_W: int = 182
    CAMPO_CM_H: int = 243
    ESCALA:     int = 3          # px/cm

    # Detección
    CONF:       float = 0.4
    IMG_SZ:     int   = 640

    # Umbrales de eventos (cm)
    DIST_POSESION: float = 20.0
    DIST_COLISION: float = 12.0
    DIST_DISPARO:  float = 40.0

    # Métricas
    VEL_MAX_FISICA: float = 250.0   # cm/s — filtro anti-salto
    UMBRAL_MOV:     float = 1.5      # cm mínimo movimiento real

    # Cooldowns (frames)
    COOLDOWN_GOL:    int = 60
    COOLDOWN_EVENTO: int = 45

    # Visual
    TRAIL_LEN: int = 30
    SIGMA:     int = 14

    # Narrador (opcional; servidor local de TTS)
    NARRADOR_URL: str = "http://127.0.0.1:8765/narrar"

    @property
    def CAMPO_W(self) -> int:
        return self.CAMPO_CM_W * self.ESCALA   # 546

    @property
    def CAMPO_H(self) -> int:
        return self.CAMPO_CM_H * self.ESCALA   # 729

    @property
    def GOAL_D(self) -> int:
        return int(self.CAMPO_H * 0.035)

    @property
    def CAMPO_H_TOTAL(self) -> int:
        return self.CAMPO_H + self.GOAL_D * 2


# Colores compartidos (BGR)
COL = {
    "ball":     (50, 220, 255),
    "white":    (255, 255, 255),
    "azul":     (255, 100,  50),
    "amarillo": (50,  220, 255),
    "unknown":  (128, 128, 128),
}
COLORES_ARISTA = {
    "posesion": (50, 220, 255),
    "pase":     (80, 255, 120),
    "disparo":  (80,  80, 255),
    "colision": (0,   80, 255),
}


# ═══════════════════════════════════════════════════════════════════════════
#  CAMPO — proyección y dibujo
# ═══════════════════════════════════════════════════════════════════════════
class Campo:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        h_path = Path(cfg.H_PATH)
        if not h_path.exists():
            raise FileNotFoundError(
                f"No se encontró la homografía en: {h_path}\n"
                f"Coloca 'homografia.npy' en la carpeta 'modelo/'."
            )
        self.H = np.load(str(h_path))

        self.W   = cfg.CAMPO_W
        self.H_  = cfg.CAMPO_H
        self.GD  = cfg.GOAL_D

        # Geometría de porterías
        self.goal_w  = int(self.W * 0.35)
        self.goal_x1 = (self.W - self.goal_w) // 2
        self.goal_x2 = self.goal_x1 + self.goal_w
        self.goal_top = int(self.H_ * 0.08)
        self.goal_bot = int(self.H_ * 0.92)

    def proyectar(self, px, py):
        """Proyecta un punto del video al campo canónico."""
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self.H)
        x, y = dst[0][0]
        return int(np.clip(x, 0, self.W-1)), int(np.clip(y, 0, self.H_-1))

    def nuevo_canvas(self, color=(30, 100, 30)):
        """Crea un canvas del campo (con espacio para porterías)."""
        c = np.full((self.cfg.CAMPO_H_TOTAL, self.W, 3), color, dtype=np.uint8)
        self.dibujar(c)
        return c

    def dibujar(self, c):
        """Dibuja líneas del campo y porterías sobre un canvas."""
        gd, W, H_ = self.GD, self.W, self.H_
        cv2.rectangle(c, (0, gd), (W-1, gd+H_-1), COL["white"], 1)
        cv2.line(c, (0, gd+H_//2), (W, gd+H_//2), COL["white"], 1)
        cv2.circle(c, (W//2, gd+H_//2), int(W*0.12), COL["white"], 1)
        cv2.circle(c, (W//2, gd+H_//2), 2, COL["white"], -1)
        # Portería amarilla (arriba, fuera del campo)
        cv2.rectangle(c, (self.goal_x1, 0), (self.goal_x2, gd), (50,220,255), 2)
        # Portería azul (abajo, fuera del campo)
        cv2.rectangle(c, (self.goal_x1, gd+H_), (self.goal_x2, gd+H_+gd), (255,180,50), 2)

    def y_canvas(self, cy):
        """Convierte y canónico a y de canvas (con offset de portería)."""
        return cy + self.GD

    def cruzo_gol(self, by_prev, by_curr, bx):
        """Detecta gol SOLO por cruce direccional de línea (con movimiento)."""
        if bx is None or by_prev is None:
            return None
        # La pelota debe estar en el ancho de la portería
        if not (self.goal_x1 < bx < self.goal_x2):
            return None
        # Cruce direccional: la pelota debe MOVERSE a través de la línea
        if by_prev < self.goal_bot and by_curr >= self.goal_bot:
            return "azul"
        if by_prev > self.goal_top and by_curr <= self.goal_top:
            return "amarillo"
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE EQUIPOS (por posición inicial)
# ═══════════════════════════════════════════════════════════════════════════
class TeamDetector:
    def __init__(self, cfg: Config, frames_cal=30):
        self.cfg         = cfg
        self.frames_cal  = frames_cal
        self.posiciones  = defaultdict(list)
        self.equipos     = {}
        self.calibrado   = False
        self.frame_count = 0

    def actualizar(self, pos_robots):
        self.frame_count += 1
        if not self.calibrado:
            for rid, (cx, cy) in pos_robots.items():
                self.posiciones[rid].append(cy)
            if self.frame_count >= self.frames_cal:
                self._asignar()

    def _asignar(self):
        medias = {rid: np.mean(cys) for rid, cys in self.posiciones.items()}
        if not medias:
            return
        mitad = self.cfg.CAMPO_H / 2
        for rid, cy in medias.items():
            self.equipos[rid] = "amarillo" if cy < mitad else "azul"
        self.calibrado = True
        print("Equipos detectados:")
        for rid, eq in sorted(self.equipos.items()):
            print(f"   Robot {rid} -> {eq.upper()}")

    def get_equipo(self, rid):
        return self.equipos.get(rid, "unknown")

    def get_color(self, rid):
        return COL[self.get_equipo(rid)]


# ═══════════════════════════════════════════════════════════════════════════
#  MÉTRICAS (distancia, velocidad)
# ═══════════════════════════════════════════════════════════════════════════
class Metricas:
    def __init__(self, cfg: Config, fps: float):
        self.cfg  = cfg
        self.fps  = fps
        self.data = {}

    def actualizar(self, rid, px, py):
        if rid < 1:
            return
        cfg = self.cfg
        pos_cm = (px / cfg.ESCALA, py / cfg.ESCALA)
        if rid not in self.data:
            self.data[rid] = {"pos_ant": pos_cm, "distancia": 0.0,
                              "vel": 0.0, "vel_max": 0.0}
            return
        m   = self.data[rid]
        dcm = np.hypot(pos_cm[0]-m["pos_ant"][0], pos_cm[1]-m["pos_ant"][1])
        vel = dcm * self.fps
        if vel > cfg.VEL_MAX_FISICA:
            m["pos_ant"] = pos_cm
            m["vel"] = 0.0
            return
        if dcm > cfg.UMBRAL_MOV:
            m["distancia"] += dcm
            m["vel"]        = vel
            m["vel_max"]    = max(m["vel_max"], vel)
        else:
            m["vel"] = 0.0
        m["pos_ant"] = pos_cm

    def resumen(self):
        print("\nMétricas finales:")
        for rid, m in sorted(self.data.items()):
            print(f"  R{rid}: {m['distancia']/100:.1f}m  |  "
                  f"vel máx {m['vel_max']:.0f} cm/s")


# ═══════════════════════════════════════════════════════════════════════════
#  GRAFO DE INTERACCIÓN
# ═══════════════════════════════════════════════════════════════════════════
class GrafoInteraccion:
    def __init__(self, cfg: Config, campo: Campo, narrar_fn=None):
        self.cfg      = cfg
        self.campo    = campo
        self.narrar   = narrar_fn or (lambda txt: None)
        self.posesion = None
        self.ultimo   = defaultdict(lambda: -cfg.COOLDOWN_EVENTO)
        self.colis    = set()
        self.log      = []

    def _dist_cm(self, p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) / self.cfg.ESCALA

    def _cercano(self, ball, robots):
        if not robots or ball is None:
            return None, 999
        d = {rid: self._dist_cm(ball, p) for rid, p in robots.items()}
        rid = min(d, key=d.get)
        return rid, d[rid]

    def _cerca_porteria(self, ball):
        if ball is None:
            return None
        bx, by = ball
        c = self.campo
        if c.goal_x1 < bx < c.goal_x2:
            d = int(self.cfg.DIST_DISPARO * self.cfg.ESCALA)
            if by >= c.goal_bot - d: return "azul"
            if by <= c.goal_top + d: return "amarillo"
        return None

    def actualizar(self, fidx, pos_robots, ball, score):
        cfg     = self.cfg
        aristas = []

        rid, dist = self._cercano(ball, pos_robots)
        if dist <= cfg.DIST_POSESION and rid is not None:
            if self.posesion != rid:
                if (self.posesion is not None and
                        fidx - self.ultimo["pase"] > cfg.COOLDOWN_EVENTO):
                    aristas.append((self.posesion, rid, "pase"))
                    self.ultimo["pase"] = fidx
                    self.narrar(f"Pase del robot {self.posesion} al robot {rid}.")
                elif fidx - self.ultimo["posesion"] > cfg.COOLDOWN_EVENTO:
                    self.narrar(f"Robot {rid} controla el balón.")
                    self.ultimo["posesion"] = fidx
                self.posesion = rid
            aristas.append((rid, "ball", "posesion"))

        porteria = self._cerca_porteria(ball)
        if porteria and fidx - self.ultimo["disparo"] > cfg.COOLDOWN_EVENTO:
            self.narrar(f"¡Disparo a la portería {porteria}!")
            self.ultimo["disparo"] = fidx
            aristas.append((self.posesion or -1, f"porteria_{porteria}", "disparo"))

        ids = list(pos_robots.keys())
        nuevas = set()
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                r1, r2 = ids[i], ids[j]
                if self._dist_cm(pos_robots[r1], pos_robots[r2]) <= cfg.DIST_COLISION:
                    par = (min(r1,r2), max(r1,r2))
                    nuevas.add(par)
                    aristas.append((r1, r2, "colision"))
                    if (par not in self.colis and
                            fidx - self.ultimo[f"col_{par}"] > cfg.COOLDOWN_EVENTO):
                        self.narrar(f"¡Choque entre robot {r1} y robot {r2}!")
                        self.ultimo[f"col_{par}"] = fidx
        self.colis = nuevas

        if aristas:
            self.log.append({
                "frame": fidx, "score": dict(score),
                "eventos": [(str(a), str(b), t) for a,b,t in aristas]
            })
        return aristas

    def dibujar(self, campo_canvas, aristas, pos_robots, ball):
        c = self.campo
        for r1, r2, tipo in aristas:
            col = COLORES_ARISTA.get(tipo, (200,200,200))
            p1  = pos_robots.get(r1) if isinstance(r1, int) else None
            if r2 == "ball":
                p2 = ball
            elif isinstance(r2, str) and "porteria" in r2:
                p2 = (c.W//2, c.goal_bot if "azul" in r2 else c.goal_top)
            else:
                p2 = pos_robots.get(r2) if isinstance(r2, int) else None
            if p1 and p2:
                p1c = (p1[0], c.y_canvas(p1[1]))
                p2c = (p2[0], c.y_canvas(p2[1]))
                cv2.line(campo_canvas, p1c, p2c, col, 2, cv2.LINE_AA)

    def guardar_json(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2)
        print(f"Grafo JSON: {path}  ({len(self.log)} eventos)")


# ═══════════════════════════════════════════════════════════════════════════
#  NARRADOR (thread async vía servidor TTS local)
# ═══════════════════════════════════════════════════════════════════════════
class Narrador:
    def __init__(self, cfg: Config):
        import threading, queue
        self.cfg   = cfg
        self.queue = queue.Queue(maxsize=3)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        import requests, tempfile, subprocess
        while True:
            texto = self.queue.get()
            if texto is None:
                break
            try:
                r = requests.post(self.cfg.NARRADOR_URL,
                                  json={"texto": texto, "velocidad": 1.15},
                                  timeout=8)
                if r.status_code == 200:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.write(r.content); tmp.close()
                    subprocess.Popen(
                        ["powershell", "-c",
                         f"(New-Object Media.SoundPlayer '{tmp.name}').PlaySync()"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[Narrador] Error: {e}")
            self.queue.task_done()

    def narrar(self, texto):
        if not self.queue.full():
            self.queue.put(texto)
            print(f"[Narrador] -> {texto}")
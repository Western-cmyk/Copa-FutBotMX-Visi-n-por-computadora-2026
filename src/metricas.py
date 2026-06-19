"""
src/metricas.py — Métricas por robot (distancia y velocidad)
============================================================
Acumula distancia recorrida y velocidad de cada robot en cm y cm/s,
con filtro anti-salto (descarta velocidades físicamente imposibles).
"""

import numpy as np

from .config import Config


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
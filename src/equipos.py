"""
src/equipos.py — Detección de equipos por posición inicial
==========================================================
Asigna cada robot a un equipo (azul/amarillo) según en qué mitad del campo
pasa sus primeros frames. Calibra una sola vez al inicio del partido.
"""

import numpy as np
from collections import defaultdict

from .config import Config, COL


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
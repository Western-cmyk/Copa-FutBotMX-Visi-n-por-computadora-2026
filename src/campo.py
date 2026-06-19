"""
src/campo.py — Proyección (homografía) y dibujo del campo
=========================================================
Convierte coordenadas del video al plano real del campo y dibuja el campo
canónico (líneas, círculo central, porterías) sobre un canvas.
"""

import cv2
import numpy as np
from pathlib import Path

from .config import Config, COL


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

        self.W  = cfg.CAMPO_W
        self.H_ = cfg.CAMPO_H
        self.GD = cfg.GOAL_D

        # Geometría de porterías
        self.goal_w  = int(self.W * 0.35)
        self.goal_x1 = (self.W - self.goal_w) // 2
        self.goal_x2 = self.goal_x1 + self.goal_w
        self.goal_top = int(self.H_ * 0.08)
        self.goal_bot = int(self.H_ * 0.92)

    def proyectar(self, px, py):
        """Proyecta un punto del video al campo canónico (px reales del campo)."""
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
        """Detecta gol por cruce direccional de línea (con movimiento)."""
        if bx is None or by_prev is None:
            return None
        if not (self.goal_x1 < bx < self.goal_x2):
            return None
        if by_prev < self.goal_bot and by_curr >= self.goal_bot:
            return "azul"
        if by_prev > self.goal_top and by_curr <= self.goal_top:
            return "amarillo"
        return None
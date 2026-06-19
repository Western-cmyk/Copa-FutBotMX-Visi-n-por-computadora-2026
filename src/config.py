"""
src/config.py — Configuración global del sistema
=================================================
Parámetros centrales: rutas, dimensiones del campo, umbrales de detección,
eventos y métricas. Todas las rutas son relativas a la raíz del repo.
"""

from pathlib import Path
from dataclasses import dataclass

# Raíz del repo = dos niveles arriba de este archivo (src/ -> raíz)
RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ── Rutas (relativas al repo) ──
    MODELO: str = str(RAIZ / "modelo" / "best.pt")          # YOLOv8 entrenado, portable
    H_PATH: str = str(RAIZ / "modelo" / "homografia.npy")   # matriz de homografía
    TRACKER: str = str(RAIZ / "config" / "bytetrack_custom.yaml")  # config de tracking

    # ── Campo real (cm) y escala ──
    CAMPO_CM_W: int = 182
    CAMPO_CM_H: int = 243
    ESCALA:     int = 3          # px/cm

    # ── Detección ──
    CONF:   float = 0.4
    IMG_SZ: int   = 640

    # ── Umbrales de eventos (cm) ──
    DIST_POSESION: float = 20.0
    DIST_COLISION: float = 12.0
    DIST_DISPARO:  float = 40.0

    # ── Métricas ──
    VEL_MAX_FISICA: float = 250.0   # cm/s — filtro anti-salto
    UMBRAL_MOV:     float = 1.5     # cm mínimo de movimiento real

    # ── Cooldowns (frames) ──
    COOLDOWN_GOL:    int = 60
    COOLDOWN_EVENTO: int = 45

    # ── Visual ──
    TRAIL_LEN: int = 30
    SIGMA:     int = 14

    # ── Narrador (opcional) ──
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
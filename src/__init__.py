"""
src — Paquete del sistema de análisis de fútbol robótico (Copa FutBotMX 2026)
=============================================================================
Expone los módulos principales para importarlos de forma limpia desde run.py.
"""

from .config import Config, COL, COLORES_ARISTA
from .campo import Campo
from .deteccion import Detector
from .equipos import TeamDetector
from .metricas import Metricas
from .grafo import GrafoInteraccion

__all__ = [
    "Config", "COL", "COLORES_ARISTA",
    "Campo", "Detector", "TeamDetector", "Metricas", "GrafoInteraccion",
]
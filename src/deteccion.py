"""
src/deteccion.py — Detección y tracking de robots y balón
==========================================================
Encapsula la carga del modelo YOLOv8 y el tracking con ByteTrack
(usando la config ajustada de config/bytetrack_custom.yaml).

La clase Detector expone un generador `rastrear(video)` que produce, por cada
frame, una lista limpia de detecciones ya parseadas:
    [{"clase": "Robot"|"Ball", "id": int, "bbox": (x1,y1,x2,y2), "centro": (cx,cy)}, ...]

Así el resto del pipeline no toca la API cruda de Ultralytics.
"""

import os
from pathlib import Path
from ultralytics import YOLO

from .config import Config


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Verificar que el modelo exista
        if not os.path.exists(cfg.MODELO):
            raise FileNotFoundError(
                f"No se encontró el modelo en: {cfg.MODELO}\n"
                f"Coloca 'best.pt' en la carpeta 'modelo/'."
            )
        self.model = YOLO(cfg.MODELO)

        # Resolver el tracker: usar la config custom si existe, si no la default
        if os.path.exists(cfg.TRACKER):
            self.tracker = cfg.TRACKER
            print(f"[Detector] Tracker custom: {cfg.TRACKER}")
        else:
            self.tracker = "bytetrack.yaml"
            print(f"[Detector] Tracker custom no encontrado, usando bytetrack.yaml por defecto")

    def rastrear(self, video):
        """
        Generador. Por cada frame del video, hace yield de:
            (fidx, detecciones)
        donde detecciones es una lista de dicts ya parseados.
        """
        if not os.path.exists(video):
            raise FileNotFoundError(f"No se encontró el video: {video}")

        stream = self.model.track(
            source=video,
            conf=self.cfg.CONF,
            imgsz=self.cfg.IMG_SZ,
            tracker=self.tracker,
            stream=True,
            persist=True,
            verbose=False,
        )

        for fidx, result in enumerate(stream, start=1):
            detecciones = self._parsear(result)
            yield fidx, detecciones

    def _parsear(self, result):
        """Convierte el resultado crudo de Ultralytics en una lista limpia."""
        dets = []
        for box in result.boxes:
            clase = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = int(box.id[0]) if box.id is not None else -1
            dets.append({
                "clase":  clase,
                "id":     tid,
                "bbox":   (x1, y1, x2, y2),
                "centro": (cx, cy),
            })
        return dets

    def contar_frames(self, video):
        """Devuelve el total de frames y FPS del video (para barras de progreso)."""
        import cv2
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total, fps
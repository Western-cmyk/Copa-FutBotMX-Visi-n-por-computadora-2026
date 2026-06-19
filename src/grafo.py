"""
src/grafo.py — Grafo de interacción (eventos del partido)
=========================================================
Detecta y registra eventos entre robots y balón: posesión, pases, disparos
a portería y colisiones. Dibuja las aristas sobre el campo y exporta el log
de eventos a JSON.
"""

import numpy as np
from collections import defaultdict

from .config import Config, COLORES_ARISTA
from .campo import Campo


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


import cv2  # usado en dibujar()
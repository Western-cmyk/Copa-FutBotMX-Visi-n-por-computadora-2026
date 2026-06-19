Video del partido
        |
        v
YOLOv8 + ByteTrack  ->  Deteccion y tracking (Robot, Ball)
        |
        v
Homografia  ->  Coordenadas reales del campo (cm)
        |
        +--> Metricas (distancia, velocidad)
        +--> Eventos (posesion, pases, disparos, colisiones, goles)
        |
        v
Analisis tacticos (PNG + MP4):
   heatmap - posesion - Voronoi - grafo de interaccion
        |
        v
JSON para Blender + secuencias PNG
        |
        v
Reconstruccion 3D (Blender) + narrador
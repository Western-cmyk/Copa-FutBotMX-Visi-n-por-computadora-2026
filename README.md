\# Copa FutBotMX 2026 — Análisis de Fútbol Robótico por Visión por Computadora



Sistema completo de análisis de partidos de fútbol robótico mediante visión por

computadora, con reconstrucción y visualización 3D en Blender. Procesa el video

de un partido en un solo pase y genera métricas, mapas tácticos (heatmap,

posesión, Voronoi, grafo de interacción) y una recreación 3D animada del partido

con narrador.



\*\*Categoría:\*\* Amateur · \*\*Plataforma:\*\* Local (Windows, GPU NVIDIA)



\---



\## ¿Qué hace?



A partir del video de un partido (cámara cenital), el sistema:



1\. \*\*Detecta y rastrea\*\* robots y balón con YOLOv8 + ByteTrack.

2\. \*\*Proyecta\*\* las posiciones del video al plano real del campo (182 × 243 cm)

&#x20;  mediante homografía, obteniendo coordenadas en centímetros.

3\. \*\*Calcula métricas\*\* por robot: distancia recorrida y velocidad (cm, cm/s),

&#x20;  con filtro anti-salto.

4\. \*\*Detecta eventos\*\*: posesión, pases, disparos a portería, colisiones y goles.

5\. \*\*Genera análisis tácticos\*\*, cada uno como PNG y MP4:

&#x20;  - Heatmap de posiciones por equipo

&#x20;  - Mapa de posesión

&#x20;  - Diagrama de Voronoi (territorio dominado)

&#x20;  - Grafo de interacción (eventos del partido)

6\. \*\*Reconstruye el partido en 3D\*\* (Blender 4.3.2, EEVEE Next): estadio, robots

&#x20;  animados, balón, marcador dinámico, público, sistema multi-cámara y proyección

&#x20;  del heatmap/Voronoi sobre el césped (con toggles). Incluye un narrador "pato"

&#x20;  con lip-sync.



\---



\## Pipeline



```

Video del partido

&#x20;     │

&#x20;     ▼

YOLOv8 + ByteTrack ──► Detección y tracking (Robot, Ball)

&#x20;     │

&#x20;     ▼

Homografía ──► Coordenadas reales del campo (cm)

&#x20;     │

&#x20;     ├──► Métricas (distancia, velocidad)

&#x20;     ├──► Eventos (posesión, pases, disparos, colisiones, goles)

&#x20;     │

&#x20;     ▼

Análisis tácticos (PNG + MP4):

&#x20;  heatmap · posesión · Voronoi · grafo de interacción

&#x20;     │

&#x20;     ▼

JSON para Blender + secuencias PNG

&#x20;     │

&#x20;     ▼

Reconstrucción 3D (Blender) + narrador

```



\### Nota sobre SAM 3



SAM 3 se utilizó \*\*únicamente en la preparación del dataset\*\*, para la anotación

poligonal de las dos clases (Robot y Ball) dentro de Roboflow. \*\*No forma parte

del pipeline de inferencia\*\*: el sistema en tiempo de ejecución detecta con el

modelo YOLOv8 entrenado (`modelo/best.pt`). Se documenta así por transparencia.



\---



\## Estructura del repositorio



```

.

├── README.md

├── LICENSE

├── requirements.txt

├── .gitignore

├── core.py                  # Lógica central: Config, Campo, equipos, grafo, métricas

├── procesar\_partido.py      # Motor de análisis (un pase, genera todo)

├── blender/

│   ├── motor\_3d\_partido.py  # Reconstrucción 3D + selector de partidos + proyección

│   ├── merlin\_ui.py         # Narrador "pato" con lip-sync (addon de Blender)

│   └── modelos/             # (vacío) Aquí van los .blend — ver "Assets en Drive"

├── modelo/

│   ├── best.pt              # Modelo YOLOv8 entrenado (detección Robot/Ball)

│   └── homografia.npy       # Matriz de homografía del campo

├── datos/

│   └── ejemplo/             # Video de ejemplo para probar

└── resultados/              # Capturas de ejemplo (heatmap, voronoi, grafo)

```



\---



\## Assets pesados (Google Drive)



Los modelos 3D y los sprites del narrador no se incluyen en el repositorio por su

tamaño. Se descargan desde Google Drive:



\*\*\[➜ Carpeta de assets en Google Drive](https://drive.google.com/drive/folders/1eKLdj8UJ7WWgIY1YFM9nV6OMA2md66Mm?usp=sharing)\*\*



Contenido de la carpeta:



```

Copa-FutBotMX - Modelos 3D/

├── Robot1.blend          → renombrar a robot\_azul.blend

├── Robot2.blend          → renombrar a robot\_amarillo.blend

└── Merlin Caras/         → sprites del pato + "Fondo estadio.png"

```



\*\*Para usar los modelos 3D detallados:\*\*

Descarga `Robot1.blend` y `Robot2.blend`, renómbralos a `robot\_azul.blend` y

`robot\_amarillo.blend`, y colócalos en `blender/modelos/`.



> Si no se incluyen los `.blend`, el motor 3D genera robots de respaldo por código

> automáticamente, así que el sistema funciona igual sin ellos.



\*\*Para el narrador:\*\*

Descarga la carpeta `Merlin Caras/` y, en el panel del addon del pato, selecciona

esa carpeta en el campo "Carpeta".



\---



\## Requisitos



\- Python 3.10+

\- GPU NVIDIA con CUDA (probado en RTX 3050, CUDA 12.1)

\- Blender 4.3.2 (para la parte 3D)



Instala las dependencias de Python:



```bash

pip install -r requirements.txt

```



> \*\*Nota sobre PyTorch + CUDA:\*\* este proyecto usa `torch==2.5.1+cu121`. No

> actualices torch con `pip install -U torch`, ya que puede reemplazarlo por la

> versión CPU y romper la aceleración por GPU. Si necesitas reinstalar, usa el

> índice de CUDA correspondiente.



\---



\## Uso



\### 1. Análisis del partido



Desde la raíz del repositorio:



```bash

python procesar\_partido.py --video datos/ejemplo/partido.mp4 --nombre partido\_final

```



Argumentos:

\- `--video`: ruta al video del partido (por defecto `datos/ejemplo/partido.mp4`)

\- `--nombre`: nombre de salida, sin espacios (por defecto `partido\_final`)

\- `--out`: directorio de salida (por defecto `resultados/renders`)



Esto genera, en `resultados/renders/`:

\- `blender/<nombre>.json` — datos del partido para Blender

\- `analisis/<nombre>/` — heatmap, posesión, Voronoi y grafo (PNG + MP4),

&#x20; más las secuencias PNG para proyectar en el césped 3D.



\### 2. Reconstrucción 3D (Blender)



1\. Abre Blender 4.3.2.

2\. Pestaña \*\*Scripting\*\* → abre `blender/motor\_3d\_partido.py` → \*\*Run Script\*\* (Alt+P).

3\. En el visor 3D: tecla \*\*N\*\* → pestaña \*\*"Partido"\*\* → selecciona el partido →

&#x20;  \*\*"Cargar partido"\*\*.

4\. Usa los toggles de \*\*Heatmap\*\* y \*\*Voronoi\*\* para proyectarlos sobre el césped.

5\. Cambia de cámara con los botones del panel. Reproduce con la barra espaciadora.



\### 3. Narrador (opcional)



1\. En Blender: \*\*Scripting\*\* → abre `blender/merlin\_ui.py` → \*\*Run Script\*\*.

2\. Visor 3D: \*\*N\*\* → pestaña \*\*"Pato"\*\*.

3\. Selecciona la carpeta `Merlin Caras` (descargada de Drive) en "Carpeta".

4\. Pulsa \*\*"Crear estación narrador"\*\*, carga un WAV de narración y usa

&#x20;  \*\*"Hornear animación desde WAV"\*\* → \*\*"Renderizar video"\*\*.



\---



\## Tecnologías



\- \*\*Detección/Tracking:\*\* YOLOv8, ByteTrack (Ultralytics)

\- \*\*Anotación de dataset:\*\* SAM 3 vía Roboflow (solo preparación)

\- \*\*Visión:\*\* OpenCV, homografía

\- \*\*Análisis:\*\* NumPy, SciPy (Voronoi)

\- \*\*3D:\*\* Blender 4.3.2 (EEVEE Next), Python `bpy`

\- \*\*Audio narrador:\*\* sounddevice, soundfile



\---



\## Licencia



MIT — ver \[LICENSE](LICENSE).



\## Autor



Rodrigo (Western) — ITESM Campus Cuernavaca


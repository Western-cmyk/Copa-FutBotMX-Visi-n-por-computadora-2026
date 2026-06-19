bl_info = {
    "name": "Pato Narrador Lip-Sync",
    "author": "Rodrigo",
    "version": (5, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Pato",
    "description": "Pato narrador overlay tipo streamer con lip-sync y re-anclaje a cámara activa",
    "category": "Animation",
}

import bpy
import threading
import math
import os
import random
import shutil
import tempfile
import numpy as np

# ─────────────────────────────────────────────
#  ESTADO GLOBAL
# ─────────────────────────────────────────────
_estado = {
    "activo":         False,
    "volumen_actual": 0.0,
    "volumen_suave":  0.0,
    "cara_actual":    0,
    "caras_cargadas": [],
    "horneado_frames": 0,
    "horneado_seg":    0.0,
}

# ─────────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────────
def _bucle_microfono(device_index):
    try:
        import sounddevice as sd
        def callback(indata, frames, time, status):
            rms = math.sqrt(np.mean(indata ** 2)) * 32768
            _estado["volumen_suave"] = _estado["volumen_suave"] * 0.6 + rms * 0.4
            _estado["volumen_actual"] = _estado["volumen_suave"]
        kwargs = dict(samplerate=44100, channels=1, dtype="float32",
                      callback=callback, blocksize=1024)
        if device_index is not None:
            kwargs["device"] = device_index
        with sd.InputStream(**kwargs):
            while _estado["activo"]:
                sd.sleep(30)
    except Exception as e:
        print(f"Error audio: {e}")
    finally:
        _estado["activo"] = False


def _bucle_wav(wav_path):
    """
    Reproduce el WAV en un stream de salida CONTINUO (sin cortes), y de paso
    calcula el volumen de cada bloque para mover la boca del pato en sync.
    """
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        canales = data.shape[1]
        total_frames = data.shape[0]
        pos = {"i": 0}

        def callback(outdata, frames, time_info, status):
            if status:
                print(status)
            inicio = pos["i"]
            fin = inicio + frames
            bloque = data[inicio:fin]

            if len(bloque) < frames:
                # Se acabó el archivo: completa con silencio y reinicia (loop)
                faltan = frames - len(bloque)
                relleno = np.zeros((faltan, canales), dtype="float32")
                bloque = np.vstack([bloque, relleno])
                pos["i"] = 0
            else:
                pos["i"] = fin

            outdata[:] = bloque

            rms = math.sqrt(np.mean(bloque ** 2)) * 32768
            _estado["volumen_suave"] = _estado["volumen_suave"] * 0.6 + rms * 0.4
            _estado["volumen_actual"] = _estado["volumen_suave"]

        with sd.OutputStream(samplerate=sr, channels=canales,
                              callback=callback, blocksize=1024):
            while _estado["activo"]:
                sd.sleep(30)

    except Exception as e:
        print(f"Error WAV: {e}")
    finally:
        _estado["activo"] = False


# ─────────────────────────────────────────────
#  HELPERS DE IMAGEN
# ─────────────────────────────────────────────
def _nombre_interno(numero):
    return f"Pato_Cara_{numero}"

def _listar_pngs(carpeta):
    if not os.path.isdir(carpeta):
        return []
    pngs = [f for f in os.listdir(carpeta) if f.lower().endswith(".png")]
    pngs.sort()
    return pngs

def _cargar_imagen_directo(carpeta, archivo, numero):
    ruta = os.path.join(carpeta, archivo)
    if not os.path.isfile(ruta):
        return None
    nombre = _nombre_interno(numero)
    if nombre in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[nombre])
    img = bpy.data.images.load(ruta)
    img.name = nombre
    img.alpha_mode = "PREMUL"
    return img

def _crear_material():
    nombre = "Pato_Mat"
    if nombre in bpy.data.materials:
        mat = bpy.data.materials[nombre]
    else:
        mat = bpy.data.materials.new(nombre)
    mat.use_nodes = True
    # blend_method ya no existe en Blender 4.3+ (EEVEE Next); ignorar si falta
    try:
        mat.blend_method = "BLEND"
    except (AttributeError, TypeError):
        pass
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "NONE"
        except (AttributeError, TypeError):
            pass
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out    = nodes.new("ShaderNodeOutputMaterial")
    emis   = nodes.new("ShaderNodeEmission")
    tex    = nodes.new("ShaderNodeTexImage")
    tex.name = "PatoTex"
    transp = nodes.new("ShaderNodeBsdfTransparent")
    mix    = nodes.new("ShaderNodeMixShader")
    links.new(tex.outputs["Color"], emis.inputs["Color"])
    links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(emis.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    out.location=(400,0); mix.location=(200,0); emis.location=(0,-100)
    transp.location=(0,100); tex.location=(-300,0)
    return mat

# Offsets de cada esquina (compartidos por crear y re-anclar)
_OFFSETS_ESQUINA = {
    "INF_DER": ( 0.16, -0.10, -0.5),
    "INF_IZQ": (-0.16, -0.10, -0.5),
    "SUP_DER": ( 0.16,  0.10, -0.5),
    "SUP_IZQ": (-0.16,  0.10, -0.5),
}

def _crear_estacion_narrador(carpeta, tamano=1.0):
    """
    Crea la 'estación del narrador' lejos del estadio:
      - un plano vertical con 'Fondo estadio.png' detrás
      - el pato (Pato_Overlay) de frente
      - una cámara 'Narrador' que enfoca la escena
    Todo en X=-50 (zona despejada lejos del campo).
    """
    BASE_X = -50.0   # lejos del estadio (el campo está en X positivo, cerca de 0..2)

    # ── Plano de fondo con la imagen ──
    fondo_path = os.path.join(carpeta, "Fondo estadio.png")
    nombre_fondo = "Narrador_Fondo"
    if nombre_fondo in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nombre_fondo], do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(BASE_X, 0, 1.5))
    fondo = bpy.context.active_object
    fondo.name = nombre_fondo
    # Parar el plano vertical (mirando hacia +X, hacia la cámara)
    fondo.rotation_euler = (math.radians(90), 0, math.radians(90))
    fondo.scale = (3.5, 2.0, 1)   # ancho x alto del fondo

    if os.path.isfile(fondo_path):
        img_fondo = bpy.data.images.load(fondo_path)
        img_fondo.name = "Narrador_Fondo_IMG"
        mat_f = bpy.data.materials.new("Mat_Narrador_Fondo")
        mat_f.use_nodes = True
        nt = mat_f.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emis = nt.nodes.new("ShaderNodeEmission")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img_fondo
        emis.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], emis.inputs["Color"])
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
        fondo.data.materials.append(mat_f)
        print(f"✅ Fondo cargado: {fondo_path}")
    else:
        print(f"⚠️ No se encontró 'Fondo estadio.png' en: {carpeta}")

    # ── Pato de frente, delante del fondo (posición fijada manualmente) ──
    obj = bpy.data.objects.get("Pato_Overlay")
    if obj:
        # Quitar parent de cualquier cámara y soltar constraints
        if obj.parent:
            obj.parent = None
        for c in list(obj.constraints):
            obj.constraints.remove(c)
        obj.location = (-49.0, 0.74981, 1.1209)
        obj.rotation_euler = (math.radians(90), 0, math.radians(90))
        obj.scale = (0.680, 0.680, 0.680)

    # ── Cámara del narrador ──
    nombre_cam = "Narrador"
    if nombre_cam in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nombre_cam], do_unlink=True)
    cd = bpy.data.cameras.new(nombre_cam)
    cam = bpy.data.objects.new(nombre_cam, cd)
    bpy.context.collection.objects.link(cam)
    # La cámara mira al pato desde +X
    cam.location = (BASE_X + 4.0, 0, 1.4)
    cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    cd.lens = 50
    print("✅ Estación del narrador creada (cámara 'Narrador')")
    return cam


def _crear_plano_overlay(esquina="INF_DER", tamano=0.35):
    """Crea el pato como objeto (sin anclarlo a cámara; va al set del narrador)."""
    nombre = "Pato_Overlay"
    if nombre in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nombre], do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1)
    obj = bpy.context.active_object
    obj.name = nombre
    bpy.ops.object.transform_apply(rotation=True)

    mat = _crear_material()
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def _swapear_cara(numero):
    nombre = _nombre_interno(numero)
    if nombre not in bpy.data.images:
        return
    mat = bpy.data.materials.get("Pato_Mat")
    if not mat or not mat.use_nodes:
        return
    nodo = mat.node_tree.nodes.get("PatoTex")
    if nodo:
        nodo.image = bpy.data.images[nombre]


# ─────────────────────────────────────────────
#  HORNEAR AUDIO → SECUENCIA DE FRAMES (para renderizar)
# ─────────────────────────────────────────────
def _calcular_secuencia_caras(wav_path, fps, umbral, velocidad_ms, caras_disponibles):
    """
    Analiza el WAV completo y devuelve una lista con el número de cara
    que corresponde a CADA frame del video (frame 1, 2, 3, ...).
    """
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mezcla a mono solo para analizar volumen

    duracion = len(data) / sr
    total_frames = max(1, math.ceil(duracion * fps))
    muestras_por_frame = max(1, int(sr / fps))

    reposo = caras_disponibles[0]
    habla = caras_disponibles[1:] if len(caras_disponibles) > 1 else [reposo]

    min_gap_frames = max(1, round((velocidad_ms / 1000.0) * fps))

    secuencia = []
    cara_actual = reposo
    frames_desde_cambio = 0

    for i in range(total_frames):
        ini = i * muestras_por_frame
        fin = ini + muestras_por_frame
        bloque = data[ini:fin]
        rms = math.sqrt(np.mean(bloque ** 2)) * 32768 if len(bloque) else 0.0

        if rms > umbral:
            if cara_actual == reposo or frames_desde_cambio >= min_gap_frames:
                cara_actual = random.choice(habla)
                frames_desde_cambio = 0
            else:
                frames_desde_cambio += 1
        else:
            cara_actual = reposo
            frames_desde_cambio = 0

        secuencia.append(cara_actual)

    return secuencia, total_frames, duracion


# ─────────────────────────────────────────────
#  ANCLAJE A CÁMARA ACTIVA (función reutilizable)
# ─────────────────────────────────────────────
def anclar_pato_a_camara_activa(escala=None):
    """
    Ancla el pato overlay a la cámara ACTIVA de la escena, en su esquina
    configurada. La llama el botón 'Anclar' del pato y también el motor 3D
    cuando cambias de cámara. Si no hay pato/cámara, no hace nada.
    """
    obj = bpy.data.objects.get("Pato_Overlay")
    cam = bpy.context.scene.camera
    if not obj or not cam:
        return False
    props = bpy.context.scene.pato_props
    esquina = props.esquina
    tamano = escala if escala is not None else props.tamano

    obj.parent = cam
    obj.matrix_parent_inverse = cam.matrix_world.inverted()
    obj.location = _OFFSETS_ESQUINA.get(esquina, _OFFSETS_ESQUINA["INF_DER"])
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (tamano*0.4, tamano*0.4, tamano*0.4)

    con = None
    for c in obj.constraints:
        if c.type == 'TRACK_TO':
            con = c
            break
    if con is None:
        con = obj.constraints.new('TRACK_TO')
    con.target = cam
    con.track_axis = 'TRACK_Z'
    con.up_axis = 'UP_Y'
    return True


# ─────────────────────────────────────────────
#  PROPIEDADES
# ─────────────────────────────────────────────
def _get_devices(self, context):
    items = [("DEFAULT", "Dispositivo por defecto", "", 0)]
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                items.append((str(i), d["name"], f"Índice {i}", i + 1))
    except Exception:
        pass
    return items

class PatoProps(bpy.types.PropertyGroup):
    fuente: bpy.props.EnumProperty(
        name="Fuente",
        items=[("MIC","Micrófono","En vivo","PLAY_SOUND",0),
               ("FILE","Archivo WAV","Audio","FILE_SOUND",1)],
        default="MIC")
    dispositivo: bpy.props.EnumProperty(name="Entrada", items=_get_devices)
    wav_path: bpy.props.StringProperty(name="WAV", subtype="FILE_PATH", default="")
    carpeta_caras: bpy.props.StringProperty(name="Carpeta", subtype="DIR_PATH", default="")
    esquina: bpy.props.EnumProperty(
        name="Esquina",
        items=[("INF_DER","Inferior derecha","",0),("INF_IZQ","Inferior izquierda","",1),
               ("SUP_DER","Superior derecha","",2),("SUP_IZQ","Superior izquierda","",3)],
        default="INF_DER")
    tamano: bpy.props.FloatProperty(name="Tamaño", default=0.35, min=0.1, max=1.0)
    umbral: bpy.props.FloatProperty(name="Umbral", default=300.0, min=0.0, max=3000.0)
    velocidad_ms: bpy.props.IntProperty(name="Velocidad (ms)", default=200, min=50, max=1000)
    corriendo: bpy.props.BoolProperty(default=False)
    visible: bpy.props.BoolProperty(name="Mostrar pato", default=True)
    calidad_video: bpy.props.EnumProperty(
        name="Calidad de video",
        items=[
            ("COMPATIBLE", "Alta calidad (recomendado)",
             "Casi sin pérdida visible, se abre en cualquier reproductor de Windows y Mac", 0),
            ("LOSSLESS", "Sin pérdida real (lossless)",
             "Calidad perfecta, pero puede no abrir en el reproductor por defecto: usa VLC", 1),
        ],
        default="COMPATIBLE",
    )


# ─────────────────────────────────────────────
#  OPERADORES
# ─────────────────────────────────────────────
class PATO_OT_preparar(bpy.types.Operator):
    bl_idname = "pato.preparar"
    bl_label = "Crear estación narrador"
    def execute(self, context):
        props = context.scene.pato_props
        carpeta = bpy.path.abspath(props.carpeta_caras)
        if not os.path.isdir(carpeta):
            self.report({"ERROR"}, "Selecciona una carpeta válida")
            return {"CANCELLED"}
        pngs = _listar_pngs(carpeta)
        # Excluir el fondo de la lista de caras del pato
        pngs = [p for p in pngs if p.lower() != "fondo estadio.png"]
        if not pngs:
            self.report({"ERROR"}, f"No hay PNG de caras en: {carpeta}")
            return {"CANCELLED"}
        cargadas = []
        for i, archivo in enumerate(pngs):
            img = _cargar_imagen_directo(carpeta, archivo, i)
            if img:
                cargadas.append(i)
                print(f"  Cargada cara {i}: {archivo}")
        if not cargadas:
            self.report({"ERROR"}, "No se pudieron cargar los PNG")
            return {"CANCELLED"}
        _estado["caras_cargadas"] = cargadas
        _crear_plano_overlay(props.esquina, props.tamano)   # crea el pato
        _swapear_cara(cargadas[0])
        _crear_estacion_narrador(carpeta, props.tamano)      # fondo + cámara
        self.report({"INFO"}, f"Estación creada con {len(cargadas)} caras")
        return {"FINISHED"}

class PATO_OT_toggle_visible(bpy.types.Operator):
    bl_idname = "pato.toggle_visible"
    bl_label = "Mostrar/Ocultar pato"
    def execute(self, context):
        props = context.scene.pato_props
        props.visible = not props.visible
        obj = bpy.data.objects.get("Pato_Overlay")
        if obj:
            obj.hide_viewport = not props.visible
            obj.hide_render = not props.visible
        return {"FINISHED"}

class PATO_OT_modo_libre(bpy.types.Operator):
    bl_idname = "pato.modo_libre"
    bl_label = "Modo posicionar"
    bl_description = "Libera el pato para moverlo desde la vista de cámara"
    def execute(self, context):
        obj = bpy.data.objects.get("Pato_Overlay")
        if not obj:
            self.report({"ERROR"}, "Primero crea el pato")
            return {"CANCELLED"}
        if obj.parent:
            mw = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = mw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({"INFO"}, "Mueve el pato (G). Luego pulsa 'Anclar'")
        return {"FINISHED"}

class PATO_OT_ver_narrador(bpy.types.Operator):
    bl_idname = "pato.ver_narrador"
    bl_label = "Ver cámara narrador"
    bl_description = "Cambia a la cámara del narrador (pato + fondo estadio)"
    def execute(self, context):
        cam = bpy.data.objects.get("Narrador")
        if not cam:
            self.report({"ERROR"}, "Primero crea la estación del narrador")
            return {"CANCELLED"}
        context.scene.camera = cam
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
        return {"FINISHED"}

class PATO_OT_iniciar(bpy.types.Operator):
    bl_idname = "pato.iniciar"
    bl_label = "Iniciar Lip-Sync"
    _timer = None
    _ultimo_cambio = 0
    def modal(self, context, event):
        props = context.scene.pato_props
        if not _estado["activo"]:
            self._limpiar(context)
            props.corriendo = False
            return {"CANCELLED"}
        if event.type == "TIMER":
            import time
            ahora = int(time.time()*1000)
            vol = _estado["volumen_actual"]
            caras = _estado["caras_cargadas"]
            if len(caras) < 2:
                return {"PASS_THROUGH"}
            reposo = caras[0]
            habla  = caras[1:]
            if vol > props.umbral:
                if ahora - self._ultimo_cambio > props.velocidad_ms:
                    nueva = random.choice(habla)
                    if nueva != _estado["cara_actual"]:
                        _swapear_cara(nueva)
                        _estado["cara_actual"] = nueva
                    self._ultimo_cambio = ahora
            else:
                if _estado["cara_actual"] != reposo:
                    _swapear_cara(reposo)
                    _estado["cara_actual"] = reposo
            for a in context.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
        return {"PASS_THROUGH"}
    def execute(self, context):
        props = context.scene.pato_props
        try:
            import sounddevice  # noqa
        except ImportError:
            self.report({"ERROR"}, "Instala sounddevice")
            return {"CANCELLED"}
        if not _estado["caras_cargadas"]:
            self.report({"ERROR"}, "Primero crea el pato")
            return {"CANCELLED"}
        _estado["activo"] = True
        _estado["volumen_suave"] = 0.0
        _estado["cara_actual"] = _estado["caras_cargadas"][0]
        self._ultimo_cambio = 0
        if props.fuente == "MIC":
            dev = None if props.dispositivo=="DEFAULT" else int(props.dispositivo)
            hilo = threading.Thread(target=_bucle_microfono, args=(dev,), daemon=True)
        else:
            wav = bpy.path.abspath(props.wav_path)
            if not os.path.isfile(wav):
                self.report({"ERROR"}, "WAV no encontrado")
                _estado["activo"] = False
                return {"CANCELLED"}
            hilo = threading.Thread(target=_bucle_wav, args=(wav,), daemon=True)
        hilo.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)
        props.corriendo = True
        return {"RUNNING_MODAL"}
    def _limpiar(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
    def cancel(self, context):
        self._limpiar(context)

class PATO_OT_detener(bpy.types.Operator):
    bl_idname = "pato.detener"
    bl_label = "Detener"
    def execute(self, context):
        _estado["activo"] = False
        context.scene.pato_props.corriendo = False
        return {"FINISHED"}


class PATO_OT_hornear(bpy.types.Operator):
    bl_idname = "pato.hornear"
    bl_label = "Hornear animación desde WAV"
    bl_description = ("Analiza el WAV completo y genera una secuencia de imágenes "
                       "sincronizada para renderizar un video real con Render Animation")

    def execute(self, context):
        props = context.scene.pato_props
        wav = bpy.path.abspath(props.wav_path)

        if not os.path.isfile(wav):
            self.report({"ERROR"}, "Selecciona un archivo WAV válido en 'Archivo WAV'")
            return {"CANCELLED"}
        if not _estado["caras_cargadas"]:
            self.report({"ERROR"}, "Primero pulsa 'Crear estación narrador'")
            return {"CANCELLED"}

        try:
            import soundfile as sf  # noqa
        except ImportError:
            self.report({"ERROR"}, "Instala soundfile: pip install soundfile")
            return {"CANCELLED"}

        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base

        secuencia, total_frames, duracion = _calcular_secuencia_caras(
            wav, fps, props.umbral, props.velocidad_ms, _estado["caras_cargadas"]
        )

        # ── Carpeta donde se guardan los frames de la secuencia ──
        if bpy.data.filepath:
            base_dir = os.path.dirname(bpy.path.abspath(bpy.data.filepath))
        else:
            base_dir = tempfile.gettempdir()
        carpeta_seq = os.path.join(base_dir, "pato_secuencia_frames")
        if os.path.isdir(carpeta_seq):
            shutil.rmtree(carpeta_seq)
        os.makedirs(carpeta_seq, exist_ok=True)

        carpeta_caras = bpy.path.abspath(props.carpeta_caras)
        pngs = [p for p in _listar_pngs(carpeta_caras) if p.lower() != "fondo estadio.png"]

        ancho = max(6, len(str(total_frames)))
        for i, num_cara in enumerate(secuencia, start=1):
            origen = os.path.join(carpeta_caras, pngs[num_cara])
            destino = os.path.join(carpeta_seq, f"pato_seq_{i:0{ancho}d}.png")
            try:
                # Hardlink: no ocupa espacio extra en disco (mismo volumen)
                os.link(origen, destino)
            except OSError:
                shutil.copyfile(origen, destino)

        # ── Cargar como secuencia de imágenes en el nodo del material ──
        nombre_img = "Pato_Secuencia_IMG"
        if nombre_img in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[nombre_img])
        primer_frame = os.path.join(carpeta_seq, f"pato_seq_{1:0{ancho}d}.png")
        img_seq = bpy.data.images.load(primer_frame)
        img_seq.name = nombre_img
        img_seq.source = 'SEQUENCE'
        img_seq.alpha_mode = "PREMUL"

        mat = bpy.data.materials.get("Pato_Mat")
        nodo = mat.node_tree.nodes.get("PatoTex") if mat else None
        if nodo:
            nodo.image = img_seq
            nodo.image_user.frame_start = 1
            nodo.image_user.frame_duration = total_frames
            nodo.image_user.frame_offset = 0
            nodo.image_user.use_auto_refresh = True

        # ── Rango de tiempo de la escena ──
        scene.frame_start = 1
        scene.frame_end = total_frames
        scene.frame_current = 1

        # ── Cámara del narrador como cámara activa ──
        cam = bpy.data.objects.get("Narrador")
        if cam:
            scene.camera = cam

        # ── Audio en la línea de tiempo (VSE) ──
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        seq = scene.sequence_editor
        for s in list(seq.sequences_all):
            if s.name == "Pato_Audio":
                seq.sequences.remove(s)
        seq.sequences.new_sound(name="Pato_Audio", filepath=wav, channel=1, frame_start=1)

        # ── Salida de render: mp4 con audio, calidad alta y compatible ──
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
        scene.render.ffmpeg.audio_codec = 'AAC'
        scene.render.ffmpeg.audio_bitrate = 192

        if props.calidad_video == 'LOSSLESS':
            scene.render.ffmpeg.constant_rate_factor = 'LOSSLESS'
        else:
            scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'

        nombre_salida = "pato_video.mp4"
        scene.render.filepath = os.path.join(base_dir, nombre_salida)

        _estado["horneado_frames"] = total_frames
        _estado["horneado_seg"] = duracion

        self.report(
            {"INFO"},
            f"Horneado: {total_frames} frames (~{duracion:.1f}s). "
            f"Ahora pulsa Render > Render Animation (Ctrl+F12)"
        )
        return {"FINISHED"}


class PATO_OT_renderizar(bpy.types.Operator):
    bl_idname = "pato.renderizar"
    bl_label = "Renderizar video (Ctrl+F12)"
    bl_description = "Renderiza la animación horneada como archivo de video con audio"

    def execute(self, context):
        if not _estado["horneado_frames"]:
            self.report({"ERROR"}, "Primero pulsa 'Hornear animación desde WAV'")
            return {"CANCELLED"}
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)
        return {"FINISHED"}


# ─────────────────────────────────────────────
#  PANEL
# ─────────────────────────────────────────────
class PATO_PT_panel(bpy.types.Panel):
    bl_label = "Pato Narrador"
    bl_idname = "PATO_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pato"
    def draw(self, context):
        layout = self.layout
        props = context.scene.pato_props
        box = layout.box()
        box.label(text="Imágenes del pato", icon="IMAGE_DATA")
        box.prop(props, "carpeta_caras", text="Carpeta")
        box.prop(props, "esquina", text="Esquina")
        box.prop(props, "tamano", slider=True)
        if _estado["caras_cargadas"]:
            box.label(text=f"Caras: {len(_estado['caras_cargadas'])}", icon="CHECKMARK")
        box.operator("pato.preparar", icon="MESH_PLANE")
        box.operator("pato.ver_narrador", text="Ver cámara narrador", icon="CAMERA_DATA")
        ico = "HIDE_OFF" if props.visible else "HIDE_ON"
        box.operator("pato.toggle_visible",
                     text="Ocultar pato" if props.visible else "Mostrar pato", icon=ico)

        box2 = layout.box()
        box2.label(text="Audio", icon="PLAY_SOUND")
        box2.prop(props, "fuente", expand=True)
        if props.fuente == "MIC":
            box2.prop(props, "dispositivo", text="")
        else:
            box2.prop(props, "wav_path", text="")

        box3 = layout.box()
        box3.label(text="Ajustes", icon="PREFERENCES")
        box3.prop(props, "umbral", slider=True)
        box3.prop(props, "velocidad_ms", slider=True)

        if props.corriendo:
            box4 = layout.box()
            vol = min(_estado["volumen_actual"]/1500.0, 1.0)
            box4.progress(factor=vol, type="BAR",
                          text=f"{_estado['volumen_actual']:.0f} RMS | Cara: {_estado['cara_actual']}")

        layout.separator()
        if not props.corriendo:
            layout.operator("pato.iniciar", text="Iniciar (vista previa en vivo)", icon="PLAY")
        else:
            layout.operator("pato.detener", text="Detener", icon="PAUSE")

        # ── Renderizar video real desde el WAV ──────────────────────
        box5 = layout.box()
        box5.label(text="Renderizar video desde audio", icon="RENDER_ANIMATION")
        box5.label(text="Usa el WAV seleccionado arriba ↑", icon="INFO")
        box5.prop(props, "calidad_video", text="Calidad")
        box5.operator("pato.hornear", text="1. Hornear animación desde WAV", icon="MOD_TIME")
        if _estado["horneado_frames"]:
            box5.label(
                text=f"Listo: {_estado['horneado_frames']} frames (~{_estado['horneado_seg']:.1f}s)",
                icon="CHECKMARK",
            )
        box5.operator("pato.renderizar", text="2. Renderizar video (Ctrl+F12)", icon="RENDER_RESULT")


# ─────────────────────────────────────────────
#  REGISTRO
# ─────────────────────────────────────────────
_clases = (PatoProps, PATO_OT_preparar, PATO_OT_toggle_visible,
           PATO_OT_modo_libre, PATO_OT_ver_narrador,
           PATO_OT_iniciar, PATO_OT_detener,
           PATO_OT_hornear, PATO_OT_renderizar,
           PATO_PT_panel)

def register():
    for cls in _clases:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pato_props = bpy.props.PointerProperty(type=PatoProps)

def unregister():
    _estado["activo"] = False
    for cls in reversed(_clases):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.pato_props

if __name__ == "__main__":
    register()
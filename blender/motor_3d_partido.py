"""
motor_3d_partido.py — Motor 3D + selector de partidos + proyección en césped
=============================================================================
VERSIÓN OPTIMIZADA:
  - Público DESACTIVADO (era el mayor costo de framerate). ACTIVAR_PUBLICO=True para volver.
  - Capas de proyección SIN auto-refresh (no leen PNG del disco en cada frame).

Corre en Blender: Scripting → abrir este archivo → Run Script (Alt+P).
Luego: N → pestaña "Partido" → elige partido → "Cargar partido".

═══════════════════════════════════════════════════════════════════════════
 IMPORTANTE — RUTAS:
   Edita BASE_DIR_MANUAL abajo con la ruta del repo EN TU PC. Es necesario
   porque Blender no detecta la ubicación del script de forma fiable cuando
   se corre desde el editor interno. Cada equipo pone aquí su propia ruta.
═══════════════════════════════════════════════════════════════════════════
"""

import bpy
import json
import os
import math
import random
import glob
from pathlib import Path
from mathutils import Vector

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — EDITA ESTO con la ruta del repo en tu PC
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR_MANUAL = r"C:\Users\rodri\Copa-FutBotMX"   # <-- CAMBIA según tu máquina

def _resolver_base_dir():
    if BASE_DIR_MANUAL:
        return Path(BASE_DIR_MANUAL)
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        pass
    env = os.environ.get("COPA_FUTBOT_DIR")
    if env and os.path.isdir(env):
        return Path(env)
    if bpy.data.filepath:
        return Path(bpy.path.abspath("//")).resolve()
    return Path(os.getcwd())

BASE_DIR = _resolver_base_dir()
print(f"[motor_3d] BASE_DIR = {BASE_DIR}")

PROY             = str(BASE_DIR / "resultados" / "renders")
CARPETA_JSON     = os.path.join(PROY, "blender")
CARPETA_ANALISIS = os.path.join(PROY, "analisis")

MODELO_AZUL     = str(BASE_DIR / "blender" / "modelos" / "robot_azul.blend")
MODELO_AMARILLO = str(BASE_DIR / "blender" / "modelos" / "robot_amarillo.blend")

# ── Opciones de rendimiento ──
ACTIVAR_PUBLICO    = False   # True para volver a generar el público
CAPAS_AUTO_REFRESH = False   # True para que heatmap/voronoi se animen (más lento)

ESCALA_BLENDER = 0.01
RADIO_BALON    = 0.03
ALTURA_SEGUIMIENTO = 2.0

def cm(v):
    return v * ESCALA_BLENDER

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS DE CONSTRUCCIÓN
# ═══════════════════════════════════════════════════════════════════════════
def _material(nombre, color, emision=0.0, rough=0.5, metal=0.0):
    mat = bpy.data.materials.new(nombre)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emision > 0:
        b.inputs["Emission Color"].default_value = color
        b.inputs["Emission Strength"].default_value = emision
    return mat

def _crear_linea(p1, p2, grosor, z, mat, nombre):
    cx = (p1[0]+p2[0])/2
    cy = (p1[1]+p2[1])/2
    largo = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    ang   = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z))
    obj = bpy.context.active_object
    obj.name = nombre
    obj.scale = (largo/2 + grosor/2, grosor/2, 0.01)
    obj.rotation_euler = (0, 0, ang)
    obj.data.materials.append(mat)
    return obj

def _crear_circulo(centro, radio, grosor, z, mat, nombre, seg=48):
    bpy.ops.mesh.primitive_torus_add(
        location=(centro[0], centro[1], z),
        major_radius=radio, minor_radius=grosor/2,
        major_segments=seg, minor_segments=6)
    obj = bpy.context.active_object
    obj.name = nombre
    obj.scale.z = 0.1
    obj.data.materials.append(mat)
    return obj

# ═══════════════════════════════════════════════════════════════════════════
#  ESTADIO
# ═══════════════════════════════════════════════════════════════════════════
def construir_estadio(CAMPO_W, CAMPO_H):
    W, H = cm(CAMPO_W), cm(CAMPO_H)
    GL = cm(3)

    mat_cesped  = _material("Mat_Cesped",  (0.05, 0.4, 0.15, 1))
    mat_blanco  = _material("Mat_Blanco",  (0.95, 0.95, 0.95, 1))
    mat_amarillo= _material("Mat_Amarillo",(1.0, 0.85, 0.0, 1), emision=0.3)
    mat_azul    = _material("Mat_Azul",    (0.1, 0.4, 1.0, 1), emision=0.3)
    mat_pared   = _material("Mat_Pared",   (0.12, 0.12, 0.16, 1))
    mat_piso    = _material("Mat_Piso",    (0.2, 0.2, 0.22, 1))
    mat_grada   = _material("Mat_Grada",   (0.3, 0.35, 0.45, 1), rough=0.4, metal=0.3)

    bpy.ops.mesh.primitive_plane_add(size=2, location=(W/2, H/2, 0))
    c = bpy.context.active_object
    c.name = "Cesped"
    c.scale = (W/2 + cm(10), H/2 + cm(10), 1)
    c.data.materials.append(mat_cesped)

    z = 0.01
    _crear_linea((0,0),(W,0), GL, z, mat_blanco, "L_inf")
    _crear_linea((0,H),(W,H), GL, z, mat_blanco, "L_sup")
    _crear_linea((0,0),(0,H), GL, z, mat_blanco, "L_izq")
    _crear_linea((W,0),(W,H), GL, z, mat_blanco, "L_der")
    _crear_linea((0,H/2),(W,H/2), GL, z, mat_blanco, "L_media")
    _crear_circulo((W/2,H/2), cm(CAMPO_W*0.12), GL, z, mat_blanco, "Circulo")
    bpy.ops.mesh.primitive_cylinder_add(radius=cm(2), depth=0.005, location=(W/2,H/2,z))
    bpy.context.active_object.name = "Punto_central"
    bpy.context.active_object.data.materials.append(mat_blanco)

    aw = W*0.35; ad = cm(CAMPO_H*0.12)
    ax1 = (W-aw)/2; ax2 = ax1+aw
    _crear_linea((ax1,H),(ax1,H-ad), GL, z, mat_blanco, "Area_si")
    _crear_linea((ax2,H),(ax2,H-ad), GL, z, mat_blanco, "Area_sd")
    _crear_linea((ax1,H-ad),(ax2,H-ad), GL, z, mat_blanco, "Area_sf")
    _crear_linea((ax1,0),(ax1,ad), GL, z, mat_blanco, "Area_ii")
    _crear_linea((ax2,0),(ax2,ad), GL, z, mat_blanco, "Area_id")
    _crear_linea((ax1,ad),(ax2,ad), GL, z, mat_blanco, "Area_if")

    gw = W*0.30; gd = cm(20); gh = cm(25)
    gx1 = (W-gw)/2
    def porteria(y_base, signo, mat, nombre):
        y_fondo = y_base + signo*gd
        cz = gh/2
        bpy.ops.mesh.primitive_cube_add(size=1, location=(gx1+gw/2, y_fondo, cz))
        p = bpy.context.active_object
        p.name = f"{nombre}_fondo"; p.scale = (gw, cm(1), gh)
        p.data.materials.append(mat)
        for lado, gx in [("izq",gx1),("der",gx1+gw)]:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(gx,(y_base+y_fondo)/2,cz))
            pl = bpy.context.active_object
            pl.name = f"{nombre}_{lado}"; pl.scale = (cm(1), gd, gh)
            pl.data.materials.append(mat)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(gx1+gw/2,(y_base+y_fondo)/2,gh))
        pt = bpy.context.active_object
        pt.name = f"{nombre}_techo"; pt.scale = (gw, gd, cm(1))
        pt.data.materials.append(mat)
    porteria(H, +1, mat_amarillo, "Porteria_Amarilla")
    porteria(0, -1, mat_azul, "Porteria_Azul")

    mg = cm(30); ph = cm(60)
    px1, py1 = -mg, -mg
    px2, py2 = W+mg, H+mg
    gm_piso = cm(70)
    bpy.ops.mesh.primitive_plane_add(size=2, location=(W/2, H/2, -0.005))
    piso = bpy.context.active_object
    piso.name = "Piso_Exterior"
    piso.scale = (W/2 + gm_piso, H/2 + gm_piso, 1)
    piso.data.materials.append(mat_piso)

    for i, ((mx,my),(sx,sy)) in enumerate([
        ((W/2,py1),(px2-px1,cm(2))), ((W/2,py2),(px2-px1,cm(2))),
        ((px1,H/2),(cm(2),py2-py1)), ((px2,H/2),(cm(2),py2-py1)),
    ]):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(mx,my,ph/2))
        pared = bpy.context.active_object
        pared.name = f"Pared_{i}"; pared.scale = (sx, sy, ph)
        pared.data.materials.append(mat_pared)

    gm = cm(32); n_esc = 5; ae = cm(18); pe = cm(28)
    for lado, (bx, by, dx, dy, largo) in {
        "sup": (W/2, py2+gm, 0, 1, px2-px1),
        "inf": (W/2, py1-gm, 0, -1, px2-px1),
        "izq": (px1-gm, H/2, -1, 0, py2-py1),
        "der": (px2+gm, H/2, 1, 0, py2-py1),
    }.items():
        for e in range(n_esc):
            off = e*pe; zz = ae/2 + e*ae
            mx = bx + dx*off; my = by + dy*off
            if dy != 0: sx, sy = largo, pe
            else: sx, sy = pe, largo
            bpy.ops.mesh.primitive_cube_add(size=1, location=(mx,my,zz))
            g = bpy.context.active_object
            g.name = f"Grada_{lado}_{e}"; g.scale = (sx, sy, ae)
            g.data.materials.append(mat_grada)

    print("Estadio construido")
    return W, H

# ═══════════════════════════════════════════════════════════════════════════
#  PROYECCIÓN EN CÉSPED
# ═══════════════════════════════════════════════════════════════════════════
def crear_capa_proyeccion(nombre, carpeta_seq, prefijo, W, H, z_offset, n_frames):
    if not os.path.isdir(carpeta_seq):
        print(f"[aviso] No hay secuencia para {nombre}: {carpeta_seq}")
        return None

    pngs = sorted(glob.glob(os.path.join(carpeta_seq, f"{prefijo}_*.png")))
    if not pngs:
        print(f"[aviso] Sin PNGs en {carpeta_seq}")
        return None

    try:
        bpy.ops.mesh.primitive_plane_add(size=1, location=(W/2, H/2, z_offset))
        plano = bpy.context.active_object
        plano.name = nombre
        plano.scale = (W, H, 1)
        plano.rotation_euler = (0, 0, 0)

        mat = bpy.data.materials.new(f"Mat_{nombre}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emis = nt.nodes.new("ShaderNodeEmission")
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        tex = nt.nodes.new("ShaderNodeTexImage")

        img = bpy.data.images.load(pngs[0])

        if CAPAS_AUTO_REFRESH:
            img.source = 'SEQUENCE'
            tex.image = img
            tex.image_user.frame_duration = n_frames
            tex.image_user.frame_start = 1
            tex.image_user.frame_offset = 0
            tex.image_user.use_auto_refresh = True
        else:
            img.source = 'FILE'
            tex.image = img

        tex.interpolation = 'Linear'
        emis.inputs["Strength"].default_value = 1.5

        nt.links.new(tex.outputs["Color"], emis.inputs["Color"])
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(emis.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

        try:
            mat.blend_method = 'BLEND'
        except (AttributeError, TypeError):
            pass

        plano.data.materials.append(mat)
        plano.hide_viewport = True
        plano.hide_render = True
        modo = "animada" if CAPAS_AUTO_REFRESH else "fija"
        print(f"Capa '{nombre}': {len(pngs)} frames (textura {modo}, oculta por defecto)")
        return plano

    except Exception as e:
        print(f"[ERROR] creando capa '{nombre}': {type(e).__name__}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  LUCES / PÚBLICO / MARCADOR / MODELOS
# ═══════════════════════════════════════════════════════════════════════════
def crear_luces_estadio(CAMPO_W, CAMPO_H):
    W, H = cm(CAMPO_W), cm(CAMPO_H)
    bpy.ops.object.light_add(type='AREA', location=(W/2, H/2, cm(350)))
    luz = bpy.context.active_object
    luz.name = "Luz_Cenital"
    luz.data.energy = 350
    luz.data.size = cm(300)
    print("Luz cenital creada")

def crear_publico(CAMPO_W, CAMPO_H):
    if not ACTIVAR_PUBLICO:
        print("Público desactivado (ACTIVAR_PUBLICO = False)")
        return
    W, H = cm(CAMPO_W), cm(CAMPO_H)
    mg = cm(32)
    px1, py1 = -mg, -mg
    px2, py2 = W+mg, H+mg
    colores = [(1,0.2,0.2,1),(0.2,0.4,1,1),(1,0.85,0,1),(0.2,1,0.3,1),
               (1,0.5,0,1),(0.8,0.2,1,1),(1,1,1,1)]
    mats = [_material(f"Mat_Fan_{i}", col, emision=0.2) for i, col in enumerate(colores)]
    spacing = cm(12); altura_fan = cm(14); n_filas = 5
    fans = []
    zonas = {
        "sup": (px1, py2+cm(45), 1, 0, 0, 1, px2-px1),
        "inf": (px1, py1-cm(45), 1, 0, 0, -1, px2-px1),
        "izq": (px1-cm(45), py1, 0, 1, -1, 0, py2-py1),
        "der": (px2+cm(45), py1, 0, 1, 1, 0, py2-py1),
    }
    for lado, (bx, by, dx, dy, ex, ey, largo) in zonas.items():
        n_cols = max(1, int(largo / spacing))
        for fila in range(n_filas):
            for col in range(n_cols):
                x = bx + dx*col*spacing + ex*fila*spacing
                y = by + dy*col*spacing + ey*fila*spacing
                zz = altura_fan/2 + fila*cm(10) + cm(20)
                bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, zz))
                fan = bpy.context.active_object
                fan.name = f"Fan_{lado}_{fila}_{col}"
                fan.scale = (cm(5), cm(5), altura_fan)
                fan.data.materials.append(random.choice(mats))
                fans.append(fan)
    scene = bpy.context.scene
    for fan in fans:
        base_z = fan.location.z
        offset = random.randint(0, 19)
        for f in range(1, scene.frame_end+1, 20):
            fan.location.z = base_z + (cm(4) if (f//20 + offset) % 2 == 0 else 0)
            fan.keyframe_insert(data_path="location", index=2, frame=f+offset%20)
    print(f"Público: {len(fans)} aficionados")

def crear_marcador_dinamico(CAMPO_W, CAMPO_H, frames):
    W, H = cm(CAMPO_W), cm(CAMPO_H)
    def crear_texto(nombre, loc, rot, tam):
        bpy.ops.object.text_add(location=loc)
        txt = bpy.context.active_object
        txt.name = nombre
        txt.data.body = "0 - 0"
        txt.data.align_x = 'CENTER'
        txt.data.size = tam
        txt.rotation_euler = rot
        mat = _material(f"Mat_{nombre}", (1,1,1,1), emision=2.0)
        txt.data.materials.append(mat)
        return txt
    crear_texto("Marcador_Flotante", (W/2, H/2, cm(150)), (math.radians(90),0,0), cm(30))
    crear_texto("Marcador_Pared", (W/2, H+cm(28), cm(70)), (math.radians(90),0,0), cm(25))

    scores = []
    for fr in frames:
        s = fr.get("score", {"azul":0,"amarillo":0})
        scores.append(f"{s['azul']} - {s['amarillo']}")
    bpy.app.driver_namespace["scores_partido"] = scores

    def actualizar_marcador(scene):
        sc = bpy.app.driver_namespace.get("scores_partido", [])
        if not sc: return
        idx = max(0, min(scene.frame_current-1, len(sc)-1))
        for n in ["Marcador_Flotante","Marcador_Pared"]:
            o = bpy.data.objects.get(n)
            if o and o.type == 'FONT':
                o.data.body = sc[idx]

    otros = [h for h in bpy.app.handlers.frame_change_post
             if h.__name__ != "actualizar_marcador"]
    bpy.app.handlers.frame_change_post.clear()
    for h in otros:
        bpy.app.handlers.frame_change_post.append(h)
    bpy.app.handlers.frame_change_post.append(actualizar_marcador)
    print("Marcador dinámico creado")

def crear_robot_respaldo(nombre, color):
    bpy.ops.mesh.primitive_cylinder_add(radius=cm(7), depth=cm(12), location=(0, 0, cm(6)))
    cuerpo = bpy.context.active_object
    cuerpo.name = nombre
    mat = _material(f"Mat_{nombre}", (*color, 1), emision=0.2, metal=0.4, rough=0.3)
    cuerpo.data.materials.append(mat)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, cm(5), cm(12)))
    cab = bpy.context.active_object
    cab.scale = (cm(4), cm(3), cm(3))
    cab.data.materials.append(mat)
    bpy.ops.object.select_all(action='DESELECT')
    cuerpo.select_set(True); cab.select_set(True)
    bpy.context.view_layer.objects.active = cuerpo
    bpy.ops.object.join()
    m = bpy.context.active_object
    m.name = nombre
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return m

def importar_modelo(blend_path, nombre, color_respaldo=(0.2, 0.4, 1.0)):
    if not os.path.exists(blend_path):
        print(f"[aviso] No existe modelo {blend_path} — usando robot de respaldo")
        return crear_robot_respaldo(nombre, color_respaldo)
    with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
        dt.objects = [o for o in df.objects]
    imp = [o for o in dt.objects if o and o.type == 'MESH']
    for o in imp:
        bpy.context.collection.objects.link(o)
    if not imp:
        return crear_robot_respaldo(nombre, color_respaldo)
    bpy.ops.object.select_all(action='DESELECT')
    for o in imp: o.select_set(True)
    bpy.context.view_layer.objects.active = imp[0]
    if len(imp) > 1: bpy.ops.object.join()
    m = bpy.context.active_object
    m.name = nombre
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    d = m.dimensions; mx = max(d.x, d.y, d.z)
    if mx > 0:
        fac = cm(15)/mx
        m.scale = (fac, fac, fac)
        bpy.ops.object.transform_apply(scale=True)
    return m

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTRUIR PARTIDO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════
def construir_partido(json_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for h in list(bpy.app.handlers.frame_change_post):
        if h.__name__ == "actualizar_marcador":
            bpy.app.handlers.frame_change_post.remove(h)

    with open(json_path, "r", encoding="utf-8") as f:
        datos = json.load(f)

    FPS     = datos["fps"]
    CAMPO_W = datos["campo_cm"][0]
    CAMPO_H = datos["campo_cm"][1]
    frames  = datos["frames"]
    equipos = datos.get("equipos", {})
    n_frames = len(frames)
    print(f"\nCargando: {os.path.basename(json_path)} — {n_frames} frames")

    scene = bpy.context.scene
    scene.render.fps = int(FPS)
    scene.frame_start = 1
    scene.frame_end = n_frames

    W, H = construir_estadio(CAMPO_W, CAMPO_H)
    crear_luces_estadio(CAMPO_W, CAMPO_H)
    crear_publico(CAMPO_W, CAMPO_H)
    crear_marcador_dinamico(CAMPO_W, CAMPO_H, frames)

    nombre_partido = os.path.splitext(os.path.basename(json_path))[0]
    dir_seq = os.path.join(CARPETA_ANALISIS, nombre_partido)
    crear_capa_proyeccion("Capa_Heatmap",
                          os.path.join(dir_seq, "heatmap_seq"), "heat",
                          W, H, z_offset=0.02, n_frames=n_frames)
    crear_capa_proyeccion("Capa_Voronoi",
                          os.path.join(dir_seq, "voronoi_seq"), "voro",
                          W, H, z_offset=0.015, n_frames=n_frames)
    for n in ["Capa_Heatmap", "Capa_Voronoi"]:
        o = bpy.data.objects.get(n)
        if o:
            o.hide_set(True)
            o.hide_viewport = True
            o.hide_render = True

    print("Importando modelos...")
    base_azul = importar_modelo(MODELO_AZUL, "Base_Azul", color_respaldo=(0.1, 0.4, 1.0))
    base_amar = importar_modelo(MODELO_AMARILLO, "Base_Amarillo", color_respaldo=(1.0, 0.85, 0.0))

    ids = sorted({rid for fr in frames for rid in fr["robots"].keys()}, key=lambda x: int(x))
    robots_obj = {}
    for rid in ids:
        eq = equipos.get(rid, "azul")
        base = base_azul if eq == "azul" else base_amar
        if base is None: continue
        inst = base.copy()
        inst.data = base.data.copy()
        inst.name = f"Robot_{rid}_{eq}"
        bpy.context.collection.objects.link(inst)
        min_z = min((inst.matrix_world @ v.co).z for v in inst.data.vertices)
        inst.location.z -= min_z
        robots_obj[rid] = inst
    if base_azul: base_azul.hide_render = base_azul.hide_viewport = True
    if base_amar: base_amar.hide_render = base_amar.hide_viewport = True

    bpy.ops.mesh.primitive_uv_sphere_add(radius=RADIO_BALON, location=(0,0,RADIO_BALON))
    balon = bpy.context.active_object
    balon.name = "Balon"
    balon.data.materials.append(_material("Mat_Balon", (1,0.3,0,1), emision=0.5))

    def get_pos(rd):
        return rd["pos"] if isinstance(rd, dict) else rd
    print("Animando...")
    for i, fr in enumerate(frames):
        fb = i + 1
        scene.frame_set(fb)
        for rid, obj in robots_obj.items():
            if rid in fr["robots"]:
                x, y = get_pos(fr["robots"][rid])
                obj.location = (cm(x), cm(CAMPO_H - y), obj.location.z)
                obj.keyframe_insert(data_path="location", frame=fb)
        if fr["ball"]:
            bx, by = fr["ball"]
            balon.location = (cm(bx), cm(CAMPO_H - by), RADIO_BALON)
            balon.keyframe_insert(data_path="location", frame=fb)

    camaras = {}
    def crear_camara(nombre, loc, rot=None, lente=None):
        cd = bpy.data.cameras.new(nombre)
        c = bpy.data.objects.new(nombre, cd)
        bpy.context.collection.objects.link(c)
        c.location = loc
        if rot: c.rotation_euler = rot
        if lente: cd.lens = lente
        camaras[nombre] = c
        return c
    crear_camara("Fija_Cenital", (cm(91), cm(122), cm(400)),
                 (math.radians(0), math.radians(0), math.radians(90)), lente=40)
    crear_camara("Fija_Lateral", (cm(91), cm(-150), cm(165)),
                 (math.radians(60), math.radians(0), math.radians(0)), lente=50)
    crear_camara("Fija_Esquina", (cm(-80), cm(-80), cm(180)),
                 (math.radians(55), math.radians(0), math.radians(-45)), lente=50)
    for rid, obj in robots_obj.items():
        eq = equipos.get(rid, "azul")
        camf = crear_camara(f"Sigue_R{rid}_{eq}", (0,0,cm(ALTURA_SEGUIMIENTO*100)))
        cl = camf.constraints.new('COPY_LOCATION')
        cl.target = obj; cl.use_z = False
        ct = camf.constraints.new('TRACK_TO')
        ct.target = obj; ct.track_axis = 'TRACK_NEGATIVE_Z'; ct.up_axis = 'UP_Y'
    scene.camera = camaras["Fija_Cenital"]

    motores = {e.identifier for e in
               bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    if 'BLENDER_EEVEE_NEXT' in motores:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    elif 'BLENDER_EEVEE' in motores:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.filepath = os.path.join(PROY, "render_")
    scene.frame_set(1)

    print(f"Partido cargado: {len(robots_obj)} robots + balón + estadio")
    return len(robots_obj), n_frames

# ═══════════════════════════════════════════════════════════════════════════
#  ESCANEAR PARTIDOS
# ═══════════════════════════════════════════════════════════════════════════
def escanear_partidos(self, context):
    items = []
    if not os.path.isdir(CARPETA_JSON):
        return [("NONE", "(carpeta no encontrada)", "")]
    archivos = sorted([f for f in os.listdir(CARPETA_JSON) if f.lower().endswith(".json")])
    for nombre in archivos:
        ruta = os.path.join(CARPETA_JSON, nombre)
        simbolo = "[--]"; info = "sin marcador"
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("frames") and "score" in d["frames"][0]:
                simbolo = "[OK]"; info = f"{len(d['frames'])} frames - con marcador"
            else:
                info = f"{len(d.get('frames', []))} frames - sin marcador"
        except Exception:
            simbolo = "[X]"; info = "JSON invalido"
        items.append((ruta, f"{simbolo} {nombre}", info))
    if not items:
        return [("NONE", "(no hay partidos .json)", "")]
    return items

# ═══════════════════════════════════════════════════════════════════════════
#  OPERADORES Y PANEL
# ═══════════════════════════════════════════════════════════════════════════
class PARTIDO_OT_cargar(bpy.types.Operator):
    bl_idname = "partido.cargar"
    bl_label = "Cargar partido"
    bl_description = "Reconstruye la escena 3D con el partido seleccionado"
    def execute(self, context):
        ruta = context.scene.partido_seleccionado
        if ruta in ("NONE", "") or not os.path.exists(ruta):
            self.report({'ERROR'}, "Selecciona un partido válido")
            return {'CANCELLED'}
        try:
            n_robots, n_frames = construir_partido(ruta)
            self.report({'INFO'}, f"Cargado: {n_robots} robots, {n_frames} frames")
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

class PARTIDO_OT_toggle_capa(bpy.types.Operator):
    bl_idname = "partido.toggle_capa"
    bl_label = "Toggle capa"
    capa: bpy.props.StringProperty()
    def execute(self, context):
        o = bpy.data.objects.get(self.capa)
        if o:
            visible = o.hide_viewport
            nuevo_oculto = not visible
            o.hide_set(nuevo_oculto)
            o.hide_viewport = nuevo_oculto
            o.hide_render = nuevo_oculto
        return {'FINISHED'}

class PARTIDO_OT_set_camera(bpy.types.Operator):
    bl_idname = "partido.set_camera"
    bl_label = "Cambiar Cámara"
    cam_name: bpy.props.StringProperty()
    def execute(self, context):
        if self.cam_name in bpy.data.objects:
            context.scene.camera = bpy.data.objects[self.cam_name]
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces[0].region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}

class PARTIDO_PT_panel(bpy.types.Panel):
    bl_label = "Partido"
    bl_idname = "PARTIDO_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Partido"
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Seleccionar partido:", icon='FILE_FOLDER')
        box.prop(context.scene, "partido_seleccionado", text="")
        box.operator("partido.cargar", text="Cargar partido", icon='IMPORT')
        box.label(text="[OK] con marcador   [--] sin marcador")

        cap_h = bpy.data.objects.get("Capa_Heatmap")
        cap_v = bpy.data.objects.get("Capa_Voronoi")
        if cap_h or cap_v:
            layout.separator()
            box2 = layout.box()
            box2.label(text="Datos en el césped:", icon='IMAGE_DATA')
            if cap_h:
                ic = 'HIDE_OFF' if not cap_h.hide_viewport else 'HIDE_ON'
                txt = "Heatmap: ON" if not cap_h.hide_viewport else "Heatmap: OFF"
                box2.operator("partido.toggle_capa", text=txt, icon=ic).capa = "Capa_Heatmap"
            if cap_v:
                ic = 'HIDE_OFF' if not cap_v.hide_viewport else 'HIDE_ON'
                txt = "Voronoi: ON" if not cap_v.hide_viewport else "Voronoi: OFF"
                box2.operator("partido.toggle_capa", text=txt, icon=ic).capa = "Capa_Voronoi"

        if any(o.type == 'CAMERA' for o in bpy.data.objects):
            layout.separator()
            layout.label(text="Cámaras Fijas:")
            for n in ["Fija_Cenital","Fija_Lateral","Fija_Esquina"]:
                if n in bpy.data.objects:
                    layout.operator("partido.set_camera", text=n.replace("Fija_","")).cam_name = n
            layout.label(text="Seguimiento:")
            for on in bpy.data.objects.keys():
                if on.startswith("Sigue_"):
                    layout.operator("partido.set_camera", text=on.replace("Sigue_","Robot ")).cam_name = on
            layout.label(text="Reproducir: barra espaciadora")
        else:
            layout.label(text="Carga un partido para ver cámaras", icon='INFO')

# ═══════════════════════════════════════════════════════════════════════════
#  REGISTRO
# ═══════════════════════════════════════════════════════════════════════════
clases = [PARTIDO_OT_cargar, PARTIDO_OT_toggle_capa,
          PARTIDO_OT_set_camera, PARTIDO_PT_panel]
for cls in clases:
    try: bpy.utils.unregister_class(cls)
    except Exception: pass
    bpy.utils.register_class(cls)

bpy.types.Scene.partido_seleccionado = bpy.props.EnumProperty(
    name="Partido",
    description="Partidos disponibles",
    items=escanear_partidos,
)

print("\nPANEL 'Partido' LISTO")
print(f"   JSON: {CARPETA_JSON}")
print(f"   Análisis: {CARPETA_ANALISIS}")
print("\nUSO: N -> 'Partido' -> elige -> 'Cargar partido'")
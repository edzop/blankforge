"""Blender embedded script — runs inside blender --background --python this_script.py"""
import sys
import math

import bpy


def main():
    argv = sys.argv
    try:
        sep = argv.index("--")
    except ValueError:
        print("Usage: blender -b --python script.py -- obj_path output_path view width height")
        return

    args = argv[sep + 1:]
    obj_path, output_path, view = args[0], args[1], args[2]
    width = int(args[3]) if len(args) > 3 else 1920
    height = int(args[4]) if len(args) > 4 else 1080

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    board_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

    # Material
    mat = bpy.data.materials.new(name="BoardMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.82, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    if board_obj:
        board_obj.data.materials.append(mat)

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position camera based on view
    if board_obj:
        bbox = [board_obj.matrix_world @ v.co for v in board_obj.data.vertices]
        xs = [v.x for v in bbox]
        ys = [v.y for v in bbox]
        zs = [v.z for v in bbox]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        dist = span * 1.4
    else:
        cx, cy, cz, dist = 0, 0, 0, 3000

    if view == "top":
        cam_obj.location = (cx, cy, cz + dist)
        cam_obj.rotation_euler = (0, 0, 0)
    elif view == "side":
        cam_obj.location = (cx, cy - dist, cz)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)
    elif view == "profile":
        cam_obj.location = (cx - dist, cy, cz)
        cam_obj.rotation_euler = (math.radians(90), 0, math.radians(-90))
    else:
        cam_obj.location = (cx + dist * 0.6, cy - dist * 0.8, cz + dist * 0.4)
        cam_obj.rotation_euler = (math.radians(65), 0, math.radians(35))

    # Lights
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(-30))
    sun.energy = 3.0

    fill = bpy.data.lights.new("Fill", type="SUN")
    fill_obj = bpy.data.objects.new("Fill", fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(60), 0, math.radians(120))
    fill.energy = 1.0

    # Render settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.cycles.samples = 128

    bpy.ops.render.render(write_still=True)


main()

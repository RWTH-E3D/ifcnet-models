import bpy
import math
import sys
from pathlib import Path

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']

# cameras: a list of camera positions
# a camera position is defined by two parameters: (theta, phi),
# where we fix the "r" of (r, theta, phi) in spherical coordinate system.

# 5 orientations: front, right, back, left, top
# cameras = [
#     (60, 0), (60, 90), (60, 180), (60, 270),
#     (0, 0), (180, 0)
# ]

# 12 orientations around the object with 30-deg elevation
cameras = [(60, i) for i in range(0, 360, 30)]

render_setting = scene.render

# output image size = (W, H)
w = 448
h = 448
render_setting.resolution_x = w
render_setting.resolution_y = h


def init_camera():
    cam = D.objects['Camera']
    # select the camera object
    cam.select_set(True)


def fix_camera_to_origin():
    origin_name = 'Origin'

    # create origin
    try:
        origin = D.objects[origin_name]
    except KeyError:
        bpy.ops.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]

    origin.location = (0, 0, 0)

    cam = D.objects['Camera']
    cam.select_set(True)

    if 'Track To' not in cam.constraints:
        bpy.ops.object.constraint_add(type='TRACK_TO')

    cam.constraints['Track To'].target = origin
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def do_model(path, target_dir):
    name = load_model(path)
    center_model(name)
    normalize_model(name)
    target_path = target_dir.joinpath(*path.parts[-3:-1])
    target_path.mkdir(exist_ok=True, parents=True)
    for i, c in enumerate(cameras):
        move_camera(c)
        render()
        save(str(target_path / f"{name}.{i}.png"))

    delete_model(name)


def load_model(path):
    bpy.ops.import_scene.obj(filepath=str(path))
    return path.stem


def delete_model(name):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith(name):
            ob.select_set(True)
        else:
            ob.select_set(False)
    bpy.ops.object.delete()


def center_model(name):
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
    obj = D.objects[name]
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)


def normalize_model(name):
    obj = D.objects[name]
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim * 1.5

    # Get material
    mat = bpy.data.materials.get("Material")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")

    mat.diffuse_color = [0.2, 0.2, 0.2, 0]

    # Assign it to object
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)

    print('new dim:' + str(dim))


def move_camera(coord):
    def deg2rad(deg):
        return deg * math.pi / 180.

    r = 3.
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)

    D.objects['Camera'].location = (loc_x, loc_y, loc_z)


def render():
    bpy.ops.render.render()


def save(path):
    D.images['Render Result'].save_render(filepath=path)
    print('save to ' + path)


def make_dataset():
    src_path = Path("../../data/raw/IFCNetCore")
    dst_path = Path("../../data/processed/MVCNN/IFCNetCore")
    files = src_path.glob("**/**/*.obj")

    init_camera()
    fix_camera_to_origin()

    for f in files:
        do_model(f, dst_path)


if __name__ == "__main__":
    make_dataset()

import sys
import pymesh
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


def write_ply(filename, points=None, mesh=None, as_text=False, comments=None):
    """

    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    comments: list of string

    Returns
    -------
    boolean
        True if no problems

    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if comments:
            for comment in comments:
                header.append('comment ' + comment)

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def triangle_area_multi(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def sample_point_cloud(vertices, faces, n=2048):
    v1 = vertices[faces[:,0]]
    v2 = vertices[faces[:,1]]
    v3 = vertices[faces[:,2]]
    areas = triangle_area_multi(v1, v2, v3)
    probs = areas / areas.sum()
    face_indices = np.random.choice(range(len(faces)), size=n, p=probs)    

    sampled_v1 = v1[face_indices]
    sampled_v2 = v2[face_indices]
    sampled_v3 = v3[face_indices]

    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u+v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    w = 1 - (u + v)

    sampled_points = (sampled_v1 * u) + (sampled_v2 * v) + (sampled_v3 * w)
    sampled_points = sampled_points.astype(np.float32)

    point_cloud = pd.DataFrame({
        "x": sampled_points[:, 0],
        "y": sampled_points[:, 1],
        "z": sampled_points[:, 2]
    })

    return point_cloud


def extract_vertices_and_faces(fn):
    print(f"Processing {fn}")
    mesh = pymesh.load_mesh(str(fn))

    # clean up
    mesh, _ = pymesh.remove_isolated_vertices(mesh)

    # get elements
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # move to center
    center = np.mean(vertices, axis=0)
    vertices -= center

    # normalize
    max_len = np.max(np.sum(vertices**2, axis=1))
    vertices /= np.sqrt(max_len)

    return vertices, faces


def main(files, dst_path):

    for fn in files:
        output_path = dst_path.joinpath(*fn.parts[-3:-1])
        output_path.mkdir(parents=True, exist_ok=True)
        output_path /= Path(fn).stem + ".ply"

        vertices, faces = extract_vertices_and_faces(fn)
        point_cloud = sample_point_cloud(vertices, faces)

        write_ply(str(output_path), point_cloud, as_text=True)


def make_dataset():
    src_path = Path("../../data/raw/IFCNetCore")
    dst_path = Path("../../data/processed/DGCNN/IFCNetCore")
    files = src_path.glob("**/**/*.obj")

    main(files, dst_path)


if __name__ == "__main__":
    make_dataset()

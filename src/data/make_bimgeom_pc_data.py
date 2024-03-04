import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


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
    face_indices = np.random.choice(range(len(faces)), size=n)#, p=probs)    

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

    return sampled_points


def extract_vertices_and_faces(fn):
    geometry = PyntCloud.from_file(str(fn))
    
    vertices = np.array(geometry.points)
    faces = np.array(geometry.mesh)
    
    # move to center
    center = np.mean(vertices, axis=0)
    vertices -= center

    return vertices, faces


def main(files, dst_path):
    resolution_info = {}
    for fn in files:
        output_path = dst_path.joinpath(*fn.parts[-3:-1])
        output_path.mkdir(parents=True, exist_ok=True)
        output_path /= Path(fn).stem + ".ply"
        
        if output_path.exists():
            print(f"Skipping {fn}. Already exists.")
            continue
            
        print(f"Processing {fn}")

        vertices, faces = extract_vertices_and_faces(fn)
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        diff = max_bound - min_bound
        resolution = 0.01

        if (diff < 2*resolution).any():
            resolution = 0.001 #  Change resolution to mm

        resolution_info[str(fn)] = resolution
        x_len, y_len, z_len = diff // resolution

        num_points = int(x_len*y_len*z_len)
        
        print(x_len, y_len, z_len)
        print(f"Sampling {num_points} points")
        point_cloud = sample_point_cloud(vertices, faces, n=4096)

        print(f"{point_cloud.shape[0]} points remaining after quantization")

        point_cloud = pd.DataFrame({
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2]
        })

        write_ply(str(output_path), point_cloud, as_text=True)
        with (dst_path/"resolution_info_bimgeom.json").open("w") as f:
            json.dump(resolution_info, f, indent=2)
            


def make_dataset():
    src_path = Path("../../data/raw/BIMGEOM")
    dst_path = Path("../../data/processed/IFCGeomUniform/BIMGEOM")
    files = sorted(src_path.glob("**/**/*.ply"))

    main(files, dst_path)


if __name__ == "__main__":
    make_dataset()

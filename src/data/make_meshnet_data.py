import glob as glob
import numpy as np
import os
import shutil
from pathlib import Path


def simplify_mesh(path, output_path, num_faces=2048):
    import meshlabxml as mlx
    topology = mlx.files.measure_topology(str(path))
    if topology["face_num"] <= num_faces:
        shutil.copy(path, output_path)
        return
    
    script = mlx.FilterScript(file_in=str(path), file_out=str(output_path))
    mlx.remesh.simplify(script, texture=False, faces=num_faces, preserve_topology=False)
    script.run_script()


def simplify_models():
    src_path = Path("../../data/raw/IFCNetCore")
    dst_path = Path("../../data/interim/MeshNet/IFCNetCore")
    files = src_path.glob("**/**/*.obj")

    for f in files:
        print(f"Simplifying: {str(f)}")
        target_path = dst_path.joinpath(*f.parts[-3:-1])
        target_path.mkdir(parents=True, exist_ok=True)
        target_path /= f.name
        simplify_mesh(f, target_path)


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


def make_dataset():
    import pymesh
    src_path = Path("../../data/interim/MeshNet/IFCNetCore")
    dst_path = Path("../../data/processed/MeshNet/IFCNetCore")
    files = src_path.glob("**/**/*.obj")

    for f in files:
        print(str(f))
        # load mesh
        mesh = pymesh.load_mesh(str(f))

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

        # get normal vector
        mesh.add_attribute('face_normal')
        face_normal = mesh.get_face_attribute('face_normal')

        # get neighbors
        faces_contain_this_vertex = []
        for i in range(len(vertices)):
            faces_contain_this_vertex.append(set([]))
        centers = []
        corners = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        centers = np.array(centers)
        corners = np.array(corners)
        faces = np.concatenate([centers, corners, face_normal], axis=1)
        neighbors = np.array(neighbors)

        target_path = dst_path.joinpath(*f.parts[-3:-1])
        target_path.mkdir(parents=True, exist_ok=True)
        target_path /= (f.stem + ".npz")
        np.savez(target_path, faces=faces, neighbors=neighbors)
        print(str(target_path))


if __name__ == "__main__":
    #simplify_models()
    make_dataset()

import torch
import json
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from src.data.util import read_ply
import MinkowskiEngine as ME


class IFCNetPly(Dataset):

    def __init__(self, data_root, class_names, partition="train", transform=None):
        self.transform = transform
        self.data_root = data_root
        self.class_names = class_names
        self.partition = partition
        self.files = sorted(data_root.glob(f"**/{partition}/*.ply"))

        self.cache = {}

    def __getitem__(self, idx):
        if idx in self.cache:
            pointcloud, label = self.cache[idx]
        else:
            f = self.files[idx]
            df = read_ply(f)
            pointcloud = df["points"].to_numpy()
            class_name = f.parts[-3]
            label = self.class_names.index(class_name)
            self.cache[idx] = (pointcloud, label)

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud, label

    def __len__(self):
        return len(self.files)

    
class IFCNetPlySparse(Dataset):

    def __init__(self, data_root, class_names, partition="train", transform=None):
        self.transform = transform
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.partition = partition
        self.files = sorted(data_root.glob(f"**/{partition}/*.ply"))
        self.quantization_info = None
        
        with (self.data_root / "quantization_info.json").open("r") as f:
            self.quantization_info = json.load(f)

        self.cache = {}

    def __getitem__(self, idx):
        if idx in self.cache:
            pointcloud, label, quantization_size = self.cache[idx]
        else:
            f = self.files[idx]
            df = read_ply(f)
            pointcloud = np.ascontiguousarray(df["points"].to_numpy())
            class_name = f.parts[-3]
            quantization_size = self.quantization_info[class_name][f.stem + ".ifc"]
            label = self.class_names.index(class_name)
            self.cache[idx] = (pointcloud, label, quantization_size)

        if self.transform:
            pointcloud = self.transform(pointcloud)
            
        #coords, feats = ME.utils.sparse_quantize(
        #    coordinates=pointcloud,
        #    features=pointcloud,
        #    quantization_size=quantization_size
        #)
        coords = pointcloud / quantization_size
        feats = pointcloud
        
        # normalize
        max_len = np.max(np.sum(feats**2, axis=1))
        feats /= np.sqrt(max_len)

        return coords, feats, label

    def __len__(self):
        return len(self.files)
    

class IFCNetNumpy(Dataset):

    def __init__(self, data_root, max_faces, class_names, partition='train'):
        self.data_root = data_root
        self.max_faces = max_faces
        self.partition = partition
        self.files = sorted(data_root.glob(f"**/{partition}/*.npz"))
        self.class_names = class_names

    def __getitem__(self, idx):
        path = self.files[idx]
        class_name = path.parts[-3]
        label = self.class_names.index(class_name)
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face)
        neighbor_index = torch.from_numpy(neighbor_index)
        target = torch.tensor(label, dtype=torch.long)
        data = torch.cat([face, neighbor_index], dim=1)

        return data, target

    def __len__(self):
        return len(self.files)


class MultiviewImgDataset(Dataset):

    def __init__(self, root_dir, classnames, num_views, partition="train", transform=None):
        self.classnames = classnames
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views

        self.filepaths = sorted(root_dir.glob(f"**/{partition}/*.png"))
        self.filepaths = np.array(self.filepaths).reshape(-1, self.num_views)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        paths = self.filepaths[idx]
        class_name = paths[0].parts[-3]
        label = self.classnames.index(class_name)

        imgs = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        return torch.stack(imgs), label


class SingleImgDataset(Dataset):

    def __init__(self, root_dir, classnames, partition="train", transform=None):
        self.classnames = classnames
        self.transform = transform
        self.root_dir = root_dir

        self.filepaths = sorted(root_dir.glob(f"**/{partition}/*.png"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.parts[-3]
        label = self.classnames.index(class_name)

        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label
        
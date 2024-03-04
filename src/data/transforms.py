import numpy as np

class TranslatePointCloud:

    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
            
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud


class ShufflePointCloud:

    def __call__(self, pointcloud):
        copy = pointcloud.copy()
        np.random.shuffle(copy)
        return copy

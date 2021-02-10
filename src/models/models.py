import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
import numpy as np


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def save(self, path, model_name, epoch):
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), path/f"{model_name}-{epoch}.pth")
        
    def load(self, path):
        self.load_state_dict(torch.load(path))


class SVCNN(Model):

    def __init__(self, nclasses, pretrained=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__()

        self.nclasses = nclasses
        self.pretrained = pretrained
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretrained)
                self.net.fc = nn.Linear(512, nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretrained)
                self.net.fc = nn.Linear(512, nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretrained)
                self.net.fc = nn.Linear(2048, nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretrained).features
                self.net_2 = models.alexnet(pretrained=self.pretrained).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretrained).features
                self.net_2 = models.vgg11(pretrained=self.pretrained).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretrained).features
                self.net_2 = models.vgg16(pretrained=self.pretrained).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096, nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, model, nclasses, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__()

        self.nclasses = nclasses
        self.num_views = num_views

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views), self.num_views, *y.shape[-3:]))
        y = torch.max(y, 1)[0]
        y = y.view(y.shape[0], -1)
        y = self.net_2(y)
        return y


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(Model):
    def __init__(self, dropout, k, emb_dims, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class FaceRotateConvolution(nn.Module):

    def __init__(self):
        super(FaceRotateConvolution, self).__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, corners):

        fea = (self.rotate_mlp(corners[:, :6]) +
               self.rotate_mlp(corners[:, 3:9]) +
               self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3

        return self.fusion_mlp(fea)


class FaceKernelCorrelation(nn.Module):

    def __init__(self, num_kernel=64, sigma=0.2):
        super(FaceKernelCorrelation, self).__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU()

    def forward(self, normals, neighbor_index):

        b, _, n = normals.size()

        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(normals.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1))
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1)

        fea = torch.cat([center, neighbor], 4)
        fea = fea.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)
        weight = torch.cat([torch.sin(self.weight_alpha) * torch.cos(self.weight_beta),
                            torch.sin(self.weight_alpha) * torch.sin(self.weight_beta),
                            torch.cos(self.weight_alpha)], 0)
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1)

        dist = torch.sum((fea - weight)**2, 1)
        fea = torch.sum(torch.sum(np.e**(dist / (-2 * self.sigma**2)), 4), 3) / 16

        return self.relu(self.bn(fea))


class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)


class StructuralDescriptor(nn.Module):

    def __init__(self, num_kernel, sigma):
        super(StructuralDescriptor, self).__init__()

        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(num_kernel, sigma)
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64 + 3 + num_kernel, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

    def forward(self, corners, normals, neighbor_index):
        structural_fea1 = self.FRC(corners)
        structural_fea2 = self.FKC(normals, neighbor_index)

        return self.structural_mlp(torch.cat([structural_fea1, structural_fea2, normals], 1))


class MeshConvolution(nn.Module):

    def __init__(self, aggregation_method, spatial_in_channel,
        structural_in_channel, spatial_out_channel, structural_out_channel):
        super(MeshConvolution, self).__init__()

        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel

        assert aggregation_method in ['Concat', 'Max', 'Average']
        self.aggregation_method = aggregation_method

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in_channel + self.structural_in_channel, self.spatial_out_channel, 1),
            nn.BatchNorm1d(self.spatial_out_channel),
            nn.ReLU(),
        )

        if self.aggregation_method == 'Concat':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 2, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def forward(self, spatial_fea, structural_fea, neighbor_index):
        b, _, n = spatial_fea.size()

        # Combination
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea], 1))

        # Aggregation
        if self.aggregation_method == 'Concat':
            structural_fea = torch.cat([structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 1)
            structural_fea = self.concat_mlp(structural_fea)
            structural_fea = torch.max(structural_fea, 3)[0]

        elif self.aggregation_method == 'Max':
            structural_fea = torch.cat([structural_fea.unsqueeze(3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 3)
            structural_fea = torch.max(structural_fea, 3)[0]

        elif self.aggregation_method == 'Average':
            structural_fea = torch.cat([structural_fea.unsqueeze(3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 3)
            structural_fea = torch.sum(structural_fea, dim=3) / 4

        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea

class MeshNet(Model):

    def __init__(self, num_kernel, sigma, aggregation_method, require_fea=True):
        super(MeshNet, self).__init__()
        self.require_fea = require_fea

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(num_kernel, sigma)
        self.mesh_conv1 = MeshConvolution(aggregation_method, 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(aggregation_method, 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        face = x[..., :15].float()
        neighbor_index = x[..., 15:].long()

        # reorganize
        face = face.permute(0, 2, 1).contiguous()
        centers, corners, normals = face[:, :3], face[:, 3:12], face[:, 12:]
        corners = corners - torch.cat([centers, centers, centers], dim=1)

        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        y = self.classifier[-1:](fea)

        if self.require_fea:
            return y, fea / torch.norm(fea)
        else:
            return y
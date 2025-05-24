import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import yaml

class BBox3DDataset(Dataset):
    def __init__(
            self,
            data_root,
            split='train',
            transform=None,
            flip_p = 0.5
            
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.flip_p = flip_p

        # Load splits.yaml data
        with open('./splits.yaml', 'r') as f:
            splits = yaml.load(f, Loader=yaml.FullLoader)

        self.split_dirs = splits[self.split]

        #ToDo : Get the mean and std from the dataset
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=self.flip_p),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        return len(self.split_dirs)
    
    def __getitem__(self, index):
        dir_name = self.split_dirs[index]
        dir_path = osp.join(self.data_root, dir_name)

        data = {}

        # Loading image
        img = Image.open(osp.join(dir_path, 'rgb.jpg'))
        img = self.image_transform(img)
        data['img'] = img

        # Loading 2D bbox
        bbox2D = torch.from_numpy(np.load(osp.join(dir_path,'bbox2d.npy'))).float()
        data['bbox2D'] = bbox2D    

        # Loading the pcd
        pcd_data = np.load(osp.join(dir_path, 'pc.npy'))
        x = pcd_data[0]
        y = pcd_data[1]
        z = pcd_data[2]

        pcd_points = np.vstack((x,y,z)).T
        # pcd_points = np.stack([x,y,z], axis=1).reshape(-1,3)
        data['pcd'] = torch.from_numpy(pcd_points).float()

        # Loading the 3D bbox
        bbox3D = torch.from_numpy(np.load(osp.join(dir_path, 'bbox3d.npy'))).float()
        data['bbox3D'] = bbox3D

        # Loading the segmentation mask
        mask = torch.from_numpy(np.load(osp.join(dir_path, 'mask.npy'))).float()
        data['mask'] = mask

        # Loading calibration data
        K,E = self.load_calib(dir_path)
        data['intrinsics'] = K
        data['extrinsics'] = E

        return data
    
    @staticmethod
    def load_calib(dir_path):
        with open(osp.join(dir_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            intrinsics = list(map(float, lines[0].split()))
            mat_K = torch.from_numpy(np.array(intrinsics)).float().reshape(3,3)
            extrinsics = list(map(float, lines[1].split()))
            mat_E = torch.from_numpy(np.array(extrinsics)).float().reshape(4,4)
        
        return mat_K, mat_E
        

    





        
        
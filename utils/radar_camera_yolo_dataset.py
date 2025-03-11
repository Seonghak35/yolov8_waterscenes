import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class RadarCameraYoloDataset(Dataset):
    def __init__(self, dataset_path, input_shape, num_classes, epoch_length, train=True):
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.train = train

        self.image_dir = os.path.join(dataset_path, 'images')
        self.radar_dir = os.path.join(dataset_path, 'radar')
        self.labels_dir = os.path.join(dataset_path, 'labels')

        self.image_files = sorted(os.listdir(self.image_dir))
        self.radar_files = sorted(os.listdir(self.radar_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))

        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.epoch_length

    def load_radar_data(self, radar_path):
        """Load radar REVP maps."""
        radar_data = pd.read_csv(radar_path)
        return radar_data[['range', 'elevation', 'velocity', 'power']].values

    def project_radar_to_camera(self, radar_data):
        """Projects 3D radar points onto the camera image plane."""
        # Dummy projection function - replace with actual extrinsic matrix
        return np.clip(radar_data[:, :2], 0, 1)

    def __getitem__(self, idx):
        index = idx % len(self.image_files)

        img_path = os.path.join(self.image_dir, self.image_files[index])
        radar_path = os.path.join(self.radar_dir, self.radar_files[index])
        label_path = os.path.join(self.labels_dir, self.label_files[index])

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load radar data
        radar_data = self.load_radar_data(radar_path)
        radar_features = self.project_radar_to_camera(radar_data)

        # Load labels
        labels = np.loadtxt(label_path, delimiter=' ')

        return image, torch.tensor(radar_features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

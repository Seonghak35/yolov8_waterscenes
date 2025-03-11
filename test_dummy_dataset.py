import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class DummyRadarCameraYoloDataset(Dataset):
    def __init__(self, input_shape=(640, 640), num_classes=7, epoch_length=10, train=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.train = train

    def __len__(self):
        return self.epoch_length

    def generate_radar_revp_map(self, num_points=1000):
        """Generates a dummy REVP (Range, Elevation, Velocity, Power) radar map."""
        radar_points = np.random.rand(num_points, 4)  # (N, 4)
        revp_map = np.zeros((4, self.input_shape[0], self.input_shape[1]))  # (4, H, W)

        for i in range(num_points):
            u, v = int(radar_points[i, 0] * (self.input_shape[1] - 1)), int(radar_points[i, 1] * (self.input_shape[0] - 1))
            if 0 <= u < self.input_shape[1] and 0 <= v < self.input_shape[0]:
                revp_map[:, v, u] = radar_points[i]  # Assign REVP values
        
        return torch.tensor(revp_map, dtype=torch.float32)

    def __getitem__(self, idx):
        # Generate random RGB image (3, H, W)
        image = torch.rand((3, *self.input_shape))

        # Generate radar REVP map (4, H, W)
        radar_revp = self.generate_radar_revp_map()

        # Generate random bounding boxes (M, 5) [class, x, y, w, h]
        M = np.random.randint(1, 10)
        labels = torch.cat((torch.randint(0, self.num_classes, (M, 1)), torch.rand((M, 4))), dim=1)

        return image, radar_revp, labels

# Create dataset instance
dataset = DummyRadarCameraYoloDataset(input_shape=(640, 640), num_classes=7, epoch_length=10, train=True)

# Fetch first sample
image, radar_revp, labels = dataset[0]

# Print tensor shapes
print("✅ Image shape:", image.shape)  # Expected: (3, 640, 640)
print("✅ Radar REVP shape:", radar_revp.shape)  # Expected: (4, 640, 640)
print("✅ Labels shape:", labels.shape)  # Expected: (M, 5)

# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.title("Sample Image from Dummy Dataset")
plt.show()

# Print radar REVP map stats
print("📊 Radar REVP Map Sample:", radar_revp[:, :5, :5])

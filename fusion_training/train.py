import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.ops import DeformConv2d

# ✅ CUDA 강제 비활성화 (GPU 사용 금지)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ 강제 CPU 모드
device = torch.device("cpu")
print("⚠️ Running on CPU mode only")

# ✅ Shuffle Attention 정의
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=8):
        super(ShuffleAttention, self).__init__()
        self.groups = 2
        self.group_channels = channels // self.groups
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x, groups):
        B, C, H, W = x.size()
        x = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)
        return x

    def forward(self, x):
        x = self.channel_shuffle(x, 2)
        attention = self.channel_attention(x)
        return x * attention

# ✅ WaterScenes와 동일한 차원의 Dummy Dataset 생성
class DummyRadarCameraYoloDataset(Dataset):
    def __init__(self, num_samples=100, input_shape=(640, 640), num_classes=7):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand((3, *self.input_shape))
        radar_revp = torch.rand((4, *self.input_shape))
        fused_input = torch.cat((image, radar_revp), dim=0)
        M = np.random.randint(1, 10)
        labels = torch.cat((torch.randint(0, self.num_classes, (M, 1)), torch.rand((M, 4))), dim=1)
        return fused_input, labels


# ✅ YOLOv8 모델 정의 (Fusion 입력 지원)
class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleYOLO, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3, stride=1, padding=1)  # 7채널 입력
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)  # 출력 클래스 수

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
# ✅ Improved SimpleYOLO with fusion
class FusionYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(FusionYOLO, self).__init__()
        self.num_classes = num_classes

        # Camera stem
        self.camera_stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Radar stem
        self.radar_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.radar_deform_conv = DeformConv2d(4, 64, kernel_size=3, padding=1)
        self.radar_bn = nn.BatchNorm2d(64)
        self.radar_attention = ShuffleAttention(64)

        # Adaptive fusion weight
        self.alpha = nn.Parameter(torch.rand(1))

        # Fusion feature layers
        self.fusion_feature = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, camera, radar):
        F_camera = self.camera_stem(camera)

        radar_pooled = self.radar_pool(radar)
        offset = torch.zeros((radar.size(0), 18, radar.size(2), radar.size(3)), device=radar.device)
        radar_feature = torch.relu(self.radar_deform_conv(radar_pooled, offset))
        radar_feature = self.radar_bn(radar_feature)
        radar_feature = self.radar_attention(radar_feature)

        fusion_feature = torch.relu(F_camera + radar_feature * self.alpha)

        output = self.fusion_feature(fusion_feature)
        return output

# ✅ Dynamic Collate Function (YOLO 바운딩 박스 개수 다름 문제 해결)
def yolo_collate_fn(batch):
    images = []
    labels = []

    for fused_input, label in batch:
        images.append(fused_input)
        labels.append(label)

    # 이미지 데이터는 크기가 동일하므로 그대로 스택 가능
    images = torch.stack(images, dim=0)

    # 바운딩 박스는 개수가 다를 수 있으므로 리스트 유지
    return images, labels

# ✅ 모델, 데이터 로더 설정
num_classes = 7
model = FusionYOLO(num_classes=num_classes).to(device)
dataset = DummyRadarCameraYoloDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=yolo_collate_fn)

# ✅ 손실 함수 및 최적화 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # inputs: [batch, 7, H, W] → 분리
        camera_input = inputs[:, :3, :, :].to(device)
        radar_input = inputs[:, 3:, :, :].to(device)

        outputs = model(camera_input, radar_input)

        # Dummy loss 계산
        target_counts = torch.tensor([label.shape[0] for label in labels], dtype=torch.float32).to(device)
        loss = criterion(outputs.mean(dim=(2, 3)), target_counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("✅ Training Completed!")

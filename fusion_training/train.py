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


# ✅ CSP Block 정의
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, expansion=0.5):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False)        

        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y1 = self.bottlenecks(y1)
        y = torch.cat([y1, y2], dim=1)
        return self.final_conv(y)
    

# ✅ Shuffle Attention 정의
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=2):
        super(ShuffleAttention, self).__init__()
        self.groups = groups

        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W).transpose(1,2).reshape(B,C,H,W)
        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att
    

# ✅ WaterScenes와 동일한 차원의 Dummy Dataset 생성
class DummyRadarCameraYoloDataset(Dataset):
    def __init__(self, num_samples=100, input_shape=(64, 64), num_classes=7):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image data (RGB)
        camera = torch.rand((3, *self.input_shape))

        # Radar REVP Map 생성 (R, E, V, P 4 channel)
        range_map = torch.rand((1, *self.input_shape)) * 100  # 거리 (0~100m)
        elevation_map = torch.rand((1, *self.input_shape)) * 180  # 고도각 (0~180도)
        velocity_map = torch.randn((1, *self.input_shape))  # 속도 (-x~+x)
        power_map = torch.rand((1, *self.input_shape)) * 50  # 반사 신호 강도

        radar_revp = torch.cat((range_map, elevation_map, velocity_map, power_map), dim=0)  # (4, H, W)

        # YOLO-format Bounding Box (M, 5) [class_id, x, y, w, h]
        M = np.random.randint(1, 10)  # 임의의 객체 개수 (1~10)
        labels = torch.zeros((M, 5))  # (M, 5) Tensor

        labels[:, 0] = torch.randint(0, self.num_classes, (M,))  # 클래스 ID (정수)
        labels[:, 1:] = torch.rand((M, 4))  # x_center, y_center, width, height (0~1 범위)

        return camera, radar_revp, labels
    

class RadarCameraYOLO(nn.Module):
    def __init__(self, num_classes=7):
        super(RadarCameraYOLO, self).__init__()

        # Camera Feature Extractor (CSP)
        self.camera_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CSPBlock(64, 128, num_layers=3)
        )

        # Radar Feature Extractor
        self.radar_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.radar_deform_conv = DeformConv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.radar_bn = nn.BatchNorm2d(64)
        self.radar_attention = ShuffleAttention(64)

        # Adaptive fusion weight α
        self.alpha = nn.Parameter(torch.rand(1))

        # channel mathcing
        self.fusion_conv = nn.Sequential(
        nn.Conv2d(128 + 64, 128, kernel_size=1, stride=1, padding=0, bias=False),  
        nn.BatchNorm2d(128),
        nn.SiLU()
        )

        # YOLO Backbone (CSPDarknet)
        self.yolo_backbone = nn.Sequential(
            CSPBlock(128, 256, num_layers=3),
            CSPBlock(256, 512, num_layers=3),
            CSPBlock(512, 1024, num_layers=1)
        )

        # FPN Neck (Feature Pyramid Network)
        self.yolo_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # YOLO Decoupled Head
        self.yolo_head_cls = nn.Conv2d(256, num_classes, kernel_size=1)
        self.yolo_head_reg = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, camera, radar):
        # Camera Feature Extraction
        F_camera = self.camera_stem(camera)

        # Radar Feature Extration
        radar_pooled = self.radar_pool(radar)

        stride_factor = self.radar_deform_conv.stride[0] if isinstance(self.radar_deform_conv.stride, tuple) else self.radar_deform_conv.stride
        offset_h = radar.size(2) // stride_factor
        offset_w = radar.size(3) // stride_factor
        offset = torch.zeros((radar.size(0), 18, offset_h, offset_w), device=radar.device)

        radar_feature = torch.relu(self.radar_deform_conv(radar_pooled, offset))
        radar_feature = self.radar_bn(radar_feature)
        radar_feature = self.radar_attention(radar_feature)

        # Adaptive Fusion
        fusion_feature = torch.cat([F_camera, self.alpha * radar_feature], dim=1)

        # YOLO
        fusion_feature = self.fusion_conv(fusion_feature)
        yolo_feature = self.yolo_backbone(fusion_feature)
        neck_feature = self.yolo_neck(yolo_feature)
        class_output = self.yolo_head_cls(neck_feature)
        bbox_output = self.yolo_head_reg(neck_feature)

        return class_output, bbox_output   


# # ✅ Dynamic Collate Function (YOLO 바운딩 박스 개수 다름 문제 해결)
# def yolo_collate_fn(batch):
#     cameras = []
#     radars = []
#     labels = []

#     for camera, radar, label in batch:  # image, radar가 분리된 상태로 전달됨
#         cameras.append(camera)
#         radars.append(radar)
#         labels.append(label)

#     # ✅ 이미지 및 레이더 데이터를 각각 스택
#     camera = torch.stack(camera, dim=0)
#     radars = torch.stack(radars, dim=0)

#     return camera, radars, labels



# ✅ 모델, 데이터 로더 설정
num_classes = 7
model = RadarCameraYOLO(num_classes=num_classes).to(device)
dataset = DummyRadarCameraYoloDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

# ✅ 손실 함수 및 최적화 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프
num_epochs = 5
print("🚀 Training started!")
for epoch in range(num_epochs):
    for i, (camera, radar, labels) in enumerate(dataloader):
        inputs = torch.cat([camera, radar], dim=1).to(device)
        outputs = model(camera.to(device), radar.to(device))

        # Dummy loss 계산
        target_counts = torch.tensor([label.shape[0] for label in labels], dtype=outputs.dtype).to(device)
        loss = criterion(outputs.mean(dim=(2, 3)).squeeze(), target_counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    print(f"✅ Epoch {epoch+1} completed.")
print("✅ Training Completed!")

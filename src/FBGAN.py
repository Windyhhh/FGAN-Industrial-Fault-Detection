import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os
from FENETFIL import Feature_input_layer
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from config import FEATURE_PARAMS, MODEL_PARAMS, DETECTION_PARAMS
import random

# ========== 固定随机种子 ==========
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ========== 确保 CuDNN 可复现 ==========
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)  # 关闭异常检测以提高速度


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")


class KDEThresholdEstimator:
    def __init__(self, confidence_level=0.90):
        self.confidence_level = confidence_level
        self.kde = None
        self.threshold = None

    def fit(self, scores):
        from scipy import stats
        scores = np.asarray(scores).ravel()
        self.kde = stats.gaussian_kde(scores)

        lo = scores.min() - np.std(scores) * 1.0
        hi = scores.max() + np.std(scores) * 1.0
        grid = np.linspace(lo, hi, 2000)
        cdf = np.array([self.kde.integrate_box_1d(-np.inf, x) for x in grid])
        idx = np.searchsorted(cdf, self.confidence_level, side='left')
        idx = np.clip(idx, 0, len(grid) - 1)
        self.threshold = float(grid[idx])
        if not np.isfinite(self.threshold):
            self.threshold = np.percentile(scores, self.confidence_level * 100)

    def get_threshold(self):
        if self.threshold is None:
            raise RuntimeError("KDE未拟合")
        return self.threshold


def load_and_preprocess_data(file_path, CPV=None, lag_number=None):
    """加载并预处理数据"""
    if CPV is None:
        CPV = FEATURE_PARAMS.get('CPV', 0.95)
    if lag_number is None:
        lag_number = FEATURE_PARAMS.get('lag_number', 3)

    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path, header=None)
    print(f"Original data shape: {data.shape}")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print(f"Scaled data shape: {scaled_data.shape}")

    features = Feature_input_layer(scaled_data, CPV, lag_number)
    print(f"Extracted features shape: {features.shape}")

    samples_lost = scaled_data.shape[0] - features.shape[0]
    aligned_data = scaled_data[samples_lost:]
    print(f"Aligned data shape: {aligned_data.shape}")
    print(f"Final features shape: {features.shape}")

    assert aligned_data.shape[0] == features.shape[0], \
        f"Sample number mismatch: {aligned_data.shape[0]} vs {features.shape[0]}"

    return aligned_data, features, scaler


def get_dimensions(file_path, CPV=None, lag_number=None):
    """获取数据的实际维度"""
    _, features, _ = load_and_preprocess_data(file_path, CPV, lag_number)
    return features.shape[1]


# 获取实际维度
data_file = '../data/d00_train.csv'
try:
    feature_size = get_dimensions(data_file)
except:
    feature_size = 13

data_size = 31

# 从配置文件读取参数
hidden_size = MODEL_PARAMS.get('hidden_size', 256)
batch_size = MODEL_PARAMS.get('batch_size', 32)
num_epochs = MODEL_PARAMS.get('num_epochs', 1500)
lr = MODEL_PARAMS.get('lr', 0.0001)
beta1 = MODEL_PARAMS.get('beta1', 0.5)
dropout_rate = MODEL_PARAMS.get('dropout', 0.05)


class Generator(nn.Module):
    """简化的生成器 - 专注区分度"""

    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()

        # 极简但有效的架构
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 关键：窄瓶颈强制学习紧凑表示
            nn.Linear(hidden_size, hidden_size // 6),  # 更窄的瓶颈
            nn.BatchNorm1d(hidden_size // 6),
            nn.ReLU(),

            nn.Linear(hidden_size // 6, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(hidden_size, output_size)
        )

        # 简单跳跃连接
        self.skip = nn.Linear(input_size, output_size)

    def forward(self, x):
        main_out = self.encoder(x)
        skip_out = self.skip(x)
        return 0.8 * main_out + 0.2 * skip_out




class Discriminator(nn.Module):
    """增强的判别器 - 专门针对难检测故障"""

    def __init__(self, input_size, is_feature_discriminator=False):
        super(Discriminator, self).__init__()

        self.is_feature_discriminator = is_feature_discriminator

        if is_feature_discriminator:
            # Dz：专门强化特征空间的异常检测能力
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),

                # 多尺度特征提取
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.LeakyReLU(0.2),
            )

            # 异常模式检测分支
            self.anomaly_detector = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.BatchNorm1d(hidden_size // 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )

            # 重建质量评估分支
            self.quality_assessor = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.BatchNorm1d(hidden_size // 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )

        else:
            # Dx：保持原有强度
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),

                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.LeakyReLU(0.2),

                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.is_feature_discriminator:
            features = self.feature_extractor(x)
            anomaly_score = self.anomaly_detector(features)
            quality_score = self.quality_assessor(features)

            # 融合两个分支的信息
            combined_score = 0.7 * anomaly_score + 0.3 * quality_score
            return combined_score, (anomaly_score, quality_score)
        else:
            return self.layers(x), None



def augment_data(data, noise_level=0.01):
    """数据增强：添加随机噪声"""
    std_per_feature = data.std(dim=0, keepdim=True)
    noise = torch.randn_like(data) * noise_level * std_per_feature
    augmented_data = data + noise
    return augmented_data


def calculate_anomaly_score_improved(original_data, feature_data, G, H, Dx, Dz, device):
    """增强的异常检测评分"""
    with torch.no_grad():
        original_data = torch.FloatTensor(original_data).to(device)
        feature_data = torch.FloatTensor(feature_data).to(device)

        batch_size = 32
        all_scores = []

        for i in range(0, len(original_data), batch_size):
            batch_x = original_data[i:i + batch_size]
            batch_z = feature_data[i:i + batch_size]

            # 重建误差计算
            recon_features = H(batch_x)
            recon_original = G(batch_z)

            # 直接重建误差 - 加权不同维度
            feature_error = torch.mean(torch.abs(recon_features - batch_z), dim=1)
            original_error = torch.mean(torch.abs(recon_original - batch_x), dim=1)

            # 循环重建误差
            cycle_features = H(recon_original)
            cycle_original = G(recon_features)
            cycle_error_z = torch.mean(torch.abs(cycle_features - batch_z), dim=1)
            cycle_error_x = torch.mean(torch.abs(cycle_original - batch_x), dim=1)

            # 判别器分数 - 重点利用Dz的两个分支
            disc_x_score, _ = Dx(batch_x)
            disc_z_score, (anomaly_score, quality_score) = Dz(batch_z)

            # 多层次异常评分
            # 1. 重建质量分数
            recon_score = (feature_error * 2.0 + original_error * 1.0 +
                           cycle_error_z * 1.5 + cycle_error_x * 0.5) / 5.0

            # 2. 特征空间异常分数（关键改进）
            feature_anomaly = (1 - anomaly_score.squeeze()) * 2.0

            # 3. 数据空间判别分数
            data_anomaly = (1 - disc_x_score.squeeze()) * 1.0

            # 4. 重建质量评估分数
            quality_anomaly = (1 - quality_score.squeeze()) * 1.5

            # 最终融合 - 重点强调特征空间异常
            final_score = (recon_score * 0.4 +
                           feature_anomaly * 0.35 +
                           quality_anomaly * 0.15 +
                           data_anomaly * 0.1)

            all_scores.extend(final_score.cpu().numpy())

        return np.array(all_scores)


# 修改2：添加诊断函数
def diagnose_discriminator(Dx, Dz, real_x, real_z, fake_x, fake_z, epoch):
    """诊断判别器状态"""
    with torch.no_grad():
        real_pred_x = Dx(real_x)[0]
        fake_pred_x = Dx(fake_x)[0]
        real_pred_z = Dz(real_z)[0]
        fake_pred_z = Dz(fake_z)[0]

        print(f"\nEpoch {epoch + 1} 判别器诊断:")
        print(f"Dx - 真实: {real_pred_x.mean():.3f}±{real_pred_x.std():.3f}")
        print(f"Dx - 虚假: {fake_pred_x.mean():.3f}±{fake_pred_x.std():.3f}")
        print(f"Dz - 真实: {real_pred_z.mean():.3f}±{real_pred_z.std():.3f}")
        print(f"Dz - 虚假: {fake_pred_z.mean():.3f}±{fake_pred_z.std():.3f}")

        # 计算判别准确率
        dx_acc = ((real_pred_x > 0.5).float().mean() + (fake_pred_x < 0.5).float().mean()) / 2
        dz_acc = ((real_pred_z > 0.5).float().mean() + (fake_pred_z < 0.5).float().mean()) / 2
        print(f"Dx准确率: {dx_acc:.3f}, Dz准确率: {dz_acc:.3f}")

        if dx_acc < 0.6 and dz_acc < 0.6:
            print("判别器过弱！")
            return "weak"
        elif dx_acc > 0.9 and dz_acc > 0.9:
            print("判别器过强！")
            return "strong"
        else:
            print("判别器状态正常")
            return "normal"


def train_bidirectional_gan():
    # 加载和预处理数据
    original_data, feature_data, scaler = load_and_preprocess_data('d00_train.csv')

    print(f"\nTraining with data dimensions:")
    print(f"Original data: {original_data.shape}")
    print(f"Feature data: {feature_data.shape}")

    # 转换为PyTorch张量
    original_data = torch.FloatTensor(original_data)
    feature_data = torch.FloatTensor(feature_data)

    # 创建数据加载器
    dataset = TensorDataset(original_data, feature_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化增强的模型
    G = Generator(feature_data.shape[1], original_data.shape[1]).to(device)
    H = Generator(original_data.shape[1], feature_data.shape[1]).to(device)
    Dx = Discriminator(original_data.shape[1]).to(device)
    Dz = Discriminator(feature_data.shape[1], is_feature_discriminator=True).to(device)

    print(f"\nEnhanced Model dimensions:")
    print(f"Enhanced Generator G: input {feature_data.shape[1]}, output {original_data.shape[1]}")
    print(f"Enhanced Generator H: input {original_data.shape[1]}, output {feature_data.shape[1]}")

    # 调整优化器（增强生成器需要稍微不同的学习率）
    g_optimizer = optim.Adam(
        list(G.parameters()) + list(H.parameters()),
        lr=lr * 1,  # 增强生成器可以用稍高的学习率
        betas=(0.5, 0.999),
        weight_decay=MODEL_PARAMS.get('l2_penalty', 0.0001)
    )
    dx_optimizer = optim.Adam(
        Dx.parameters(),
        lr=lr * 0.8,  # 保持判别器较低学习率
        betas=(0.5, 0.999),
        weight_decay=MODEL_PARAMS.get('l2_penalty', 0.0001)
    )
    dz_optimizer = optim.Adam(
        Dz.parameters(),
        lr=lr * 0.8,  # 保持判别器较低学习率
        betas=(0.5, 0.999),
        weight_decay=MODEL_PARAMS.get('l2_penalty', 0.0001)
    )

    # 学习率调度器
    g_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        g_optimizer, T_0=100, T_mult=2  # 使用warm restart
    )
    dx_scheduler = optim.lr_scheduler.StepLR(dx_optimizer, step_size=200, gamma=0.8)
    dz_scheduler = optim.lr_scheduler.StepLR(dz_optimizer, step_size=200, gamma=0.8)

    # 启用优化
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    # 训练循环
    # 设置为训练模式
    G.train()
    H.train()
    Dx.train()
    Dz.train()

    print("\n=== 开始训练（重建优化版本）===")
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        epoch_dx_loss = 0
        epoch_dz_loss = 0
        epoch_g_loss = 0
        epoch_h_loss = 0
        epoch_recon_loss = 0
        batch_count = 0

        for i, (real_x, real_z) in enumerate(dataloader):
            real_x = real_x.to(device, non_blocking=True)
            real_z = real_z.to(device, non_blocking=True)

            if np.random.rand() < 0.4:  # 40%的batch进行微扰
                real_x = augment_data(real_x, noise_level=0.03)
                real_z = augment_data(real_z, noise_level=0.03)

            # 平衡的训练频率
            dx_iterations = 1 if i % 2 == 0 else 0
            dz_iterations = 1 if i % 3 == 0 else 0  # 适度削弱Dz
            g_iterations = 2

            # ====== 训练判别器 ======
            # Dx训练（保持适度强度）
            # 训练Dx
            for _ in range(dx_iterations):
                dx_optimizer.zero_grad()

                with torch.no_grad():
                    fake_x = G(real_z)

                real_pred = Dx(real_x)[0]
                fake_pred = Dx(fake_x)[0]

                dx_loss = (nn.BCELoss()(real_pred, torch.ones_like(real_pred) * 0.9) +
                           nn.BCELoss()(fake_pred, torch.zeros_like(fake_pred) + 0.1))

                dx_loss.backward()
                dx_optimizer.step()

            # 训练Dz - 保持适度能力
            for _ in range(dz_iterations):
                dz_optimizer.zero_grad()

                with torch.no_grad():
                    fake_z = H(real_x)

                real_pred = Dz(real_z)[0]
                fake_pred = Dz(fake_z)[0]

                # 轻度标签平滑
                dz_loss = (nn.BCELoss()(real_pred, torch.ones_like(real_pred) * 0.9) +
                           nn.BCELoss()(fake_pred, torch.zeros_like(fake_pred) + 0.)) * 1.2

                dz_loss.backward()
                dz_optimizer.step()

            # ====== 专门针对重建优化的生成器训练 ======
            for _ in range(g_iterations):
                g_optimizer.zero_grad()

                fake_x = G(real_z)
                fake_z = H(real_x)

                # 对抗损失
                g_loss = nn.BCELoss()(Dx(fake_x)[0], torch.ones_like(Dx(fake_x)[0]))
                h_loss = nn.BCELoss()(Dz(fake_z)[0], torch.ones_like(Dz(fake_z)[0]))

                # 简化重建损失
                direct_loss = (nn.L1Loss()(H(real_x), real_z) * 2.0 +  # 强调H方向
                               nn.L1Loss()(G(real_z), real_x) * 1.0)

                cycle_loss = (nn.L1Loss()(H(G(real_z)), real_z) * 1.5 +
                              nn.L1Loss()(G(H(real_x)), real_x) * 0.5)

                # 平衡的损失权重
                total_loss = g_loss * 0.8 + h_loss * 1.0 + direct_loss * 1.5 + cycle_loss * 1.0

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(G.parameters()) + list(H.parameters()), max_norm=1.0)
                g_optimizer.step()

            # 累积损失统计
            epoch_dx_loss += dx_loss
            epoch_dz_loss += dz_loss
            epoch_g_loss += g_loss.item()
            epoch_h_loss += h_loss.item()
            epoch_recon_loss += direct_loss.item()
            batch_count += 1

        # 学习率调度
        g_scheduler.step()
        if epoch % 100 == 99:
            dx_scheduler.step()
            dz_scheduler.step()

        # 输出进度
        if (epoch + 1) % 10 == 0:
            avg_dx = epoch_dx_loss / max(1, batch_count)
            avg_dz = epoch_dz_loss / max(1, batch_count)
            avg_g = epoch_g_loss / batch_count
            avg_h = epoch_h_loss / batch_count
            avg_recon = epoch_recon_loss / batch_count

            tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], '
                       f'D_x: {avg_dx:.4f}, D_z: {avg_dz:.4f}, '
                       f'G: {avg_g:.4f}, H: {avg_h:.4f}, '
                       f'Recon: {avg_recon:.4f}')

        # Dz诊断
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                test_fake_x = G(real_z[:10])
                test_fake_z = H(real_x[:10])

                # 诊断时不添加噪声
                Dz.eval()
                real_pred_z = Dz(real_z[:10])[0]
                fake_pred_z = Dz(test_fake_z)[0]
                Dz.train()

                dz_acc = ((real_pred_z > 0.5).float().mean() + (fake_pred_z < 0.5).float().mean()) / 2

                print(f"\nEpoch {epoch + 1} Dz诊断:")
                print(f"Dz - 真实: {real_pred_z.mean():.3f}±{real_pred_z.std():.3f}")
                print(f"Dz - 虚假: {fake_pred_z.mean():.3f}±{fake_pred_z.std():.3f}")
                print(f"Dz准确率: {dz_acc:.3f}")

                if dz_acc > 0.80:
                    print("[WARNING] Dz still strong")
                elif dz_acc < 0.55:
                    print("[OK] Dz weakened successfully")
                else:
                    print("[INFO] Dz status moderate")

        # GPU内存管理
        if (epoch + 1) % 50 == 0:
            torch.cuda.empty_cache()

        # 保存检查点
        if (epoch + 1) % 100 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'G_state_dict': G.state_dict(),
                'H_state_dict': H.state_dict(),
                'Dx_state_dict': Dx.state_dict(),
                'Dz_state_dict': Dz.state_dict(),
                'epoch': epoch
            }, f'models/recon_optimized_checkpoint_epoch_{epoch + 1}.pth')

    print("\n=== 训练完成 ===")
    Dz.eval()

    # 使用改进的评分方法计算训练数据异常分数
    print("\n计算基于重建质量的异常分数...")
    all_scores = []

    for batch_x, batch_z in dataloader:
        scores = calculate_anomaly_score_improved(batch_x, batch_z, G, H, Dx, Dz, device)
        all_scores.extend(scores)

    all_scores = np.array(all_scores)

    # 使用更保守的阈值
    confidence_level = DETECTION_PARAMS.get('confidence_level', 0.98)
    kde_estimator = KDEThresholdEstimator(confidence_level=confidence_level)
    kde_estimator.fit(all_scores)
    threshold = kde_estimator.get_threshold()

    print(f"基于重建质量的阈值 ({confidence_level * 100}%置信度): {threshold:.4f}")
    print(f"训练数据异常分数统计: 均值={np.mean(all_scores):.4f}, 标准差={np.std(all_scores):.4f}")

    # 保存完整模型
    os.makedirs('models', exist_ok=True)
    save_dict = {
        'G_state_dict': G.state_dict(),
        'H_state_dict': H.state_dict(),
        'Dx_state_dict': Dx.state_dict(),
        'Dz_state_dict': Dz.state_dict(),
        'kde_estimator': kde_estimator,
        'threshold': threshold,
        'feature_dim': feature_data.shape[1],
        'data_dim': original_data.shape[1],
        'scoring_method': 'reconstruction_based',
        'config': {
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'CPV': FEATURE_PARAMS.get('CPV', 0.95),
            'lag_number': FEATURE_PARAMS.get('lag_number', 3),
            'confidence_level': confidence_level,
            'anomaly_detection_method': 'reconstruction_based'
        }
    }

    torch.save(save_dict, 'models/complete_model.pth')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("模型保存完成")
    print(f"特征维度: {feature_data.shape[1]}")
    print(f"原始数据维度: {original_data.shape[1]}")
    print(f"重建质量阈值: {threshold:.4f}")

    return G, H, Dx, Dz, kde_estimator


if __name__ == "__main__":
    print("=== 开始FBGAN训练（修复判别器版本）===")
    print(f"配置参数:")
    print(f"  CPV: {FEATURE_PARAMS.get('CPV', 0.95)}")
    print(f"  Lag number: {FEATURE_PARAMS.get('lag_number', 3)}")
    print(f"  Hidden size: {MODEL_PARAMS.get('hidden_size', 256)}")
    print(f"  Batch size: {MODEL_PARAMS.get('batch_size', 32)}")
    print(f"  Epochs: {MODEL_PARAMS.get('num_epochs', 1500)}")
    print(f"  Learning rate: {MODEL_PARAMS.get('lr', 0.0001)}")
    print(f"  Dropout: {MODEL_PARAMS.get('dropout', 0.05)}")
    print(f"  置信度: {DETECTION_PARAMS.get('confidence_level', 0.90)}")

    train_bidirectional_gan()
    print("\n=== 程序完成 ===")
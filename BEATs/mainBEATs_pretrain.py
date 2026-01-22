import torchaudio.transforms as transforms
import torch.nn.functional as F
import math
from torch.nn import Parameter
from beats.BEATs import BEATs, BEATsConfig
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_mixup_layer import MixupLayer, Mixup
import math
import numpy as np
from tqdm import tqdm
import os
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from scipy.stats import hmean
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score
import pandas as pd
from torchsummary import summary
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import sys
import csv
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from loss import SCAdaCos
from speechbrain.lobes.models import ECAPA_TDNN
from sklearn import manifold
import seaborn as sns
from KNNrescale import compute_knn_scaled_scores, compute_gwrp_scaled_scores
from prepare import spec_aug
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import bisect
import copy
from torch.optim.lr_scheduler import LambdaLR

num_label = 917
num_label = 604


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def to_categorical(labels, num_classes):
    return np.eye(num_classes)[labels]


def adjust_size(wav, new_size):
    reps = int(np.ceil(new_size / wav.shape[0]))
    offset = np.random.randint(low=0, high=int(reps * wav.shape[0] - new_size + 1))
    return np.tile(wav, reps=reps)[offset:offset + new_size]


def apply_specaug(spec, max_time_mask=80, max_freq_mask=80, num_time_masks=1, num_freq_masks=1):
    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)

    augmented_spec = spec.clone()

    # 应用时间遮罩
    time_masking = transforms.TimeMasking(time_mask_param=max_time_mask)
    for _ in range(num_time_masks):
        augmented_spec = time_masking(augmented_spec)

    # 应用频率遮罩
    freq_masking = transforms.FrequencyMasking(freq_mask_param=max_freq_mask)
    for _ in range(num_freq_masks):
        augmented_spec = freq_masking(augmented_spec)

    # 如果原始输入是 2D 张量，去除 batch 维度
    if augmented_spec.shape[0] == 1:
        augmented_spec = augmented_spec.squeeze(0)

    return augmented_spec


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, time_steps, input_dim)
        attention_weights = self.attention(x)  # (batch_size, time_steps, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, time_steps, 1)

        # 加权均值
        mean = torch.sum(attention_weights * x, dim=1)  # (batch_size, input_dim)

        # 加权标准差
        variance = torch.sum(attention_weights * (x - mean.unsqueeze(1)) ** 2, dim=1)  # (batch_size, input_dim)
        std_dev = torch.sqrt(variance + 1e-12)  # 避免开方时的数值不稳定性

        # 连接均值和标准差
        utterance_embedding = torch.cat([mean, std_dev], dim=1)  # (batch_size, 2 * input_dim)

        return utterance_embedding


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        W = self.weight.to(device)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(W))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = label
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine_decay, min_lr / optimizer.defaults['lr'])  # 最小值裁剪

    return LambdaLR(optimizer, lr_lambda)


def freeze_beats_layers(model, num_layers_to_freeze=6):
    encoder_layers = model.encoder.layers  # BEATs encoder transformer 层
    for i in range(num_layers_to_freeze):
        for param in encoder_layers[i].parameters():
            param.requires_grad = False


def progressively_unfreeze(model, current_epoch, total_epochs, start_unfreeze_epoch=5):
    if current_epoch < start_unfreeze_epoch:
        return  # 还未开始解冻

    encoder_layers = model.encoder.layers
    layers_to_unfreeze = (current_epoch - start_unfreeze_epoch + 1)

    for i in range(min(len(encoder_layers), layers_to_unfreeze)):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True


def update_ema_teacher(student_model, teacher_model, ema_decay=0.999):
    with torch.no_grad():
        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)


class BEATsWithNewLayer(BEATs):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.asp_layer = ECAPA_TDNN.AttentiveStatisticsPooling(768,global_context=False)
        self.asp_layer = AttentiveStatisticsPooling(768)
        self.layer1 = nn.Linear(768 * 2, 512, bias=False)  # 添加新的全连接层，假设你需要从527映射到202
        self.layer2 = nn.Linear(512, num_label, bias=False)  # 添加新的全连接层，假设你需要从527映射到202
        self.scadacos = SCAdaCos(n_classes=num_label, n_subclusters=16, trainable=True)

    def forward(self, source, label, padding_mask=None):
        x, _, lprobs = self.extract_features(source, padding_mask, need_weights=True, layer=11)

        # utterance_embedding = self.asp_layer(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        utterance_embedding = self.asp_layer(x)
        logit = self.layer1(utterance_embedding)
        logit = self.layer2(logit)
        # logit = self.scadacos((lprobs, label, label))
        # return logit, lprobs
        return logit, utterance_embedding


loss_function = nn.CrossEntropyLoss()
arcface = ArcMarginProduct(num_label, num_label)
scadacos = SCAdaCos(n_classes=154, n_subclusters=16, trainable=False)


# 模型的训练
def train(model, train_loader, optimizer, epoch):
    total_accuracy = 0
    model.train()
    progressively_unfreeze(model, epoch, epochs)
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label.to(device)
        data = torch.squeeze(data)
        data = data.to(device)
        data = torch.tensor(data, dtype=torch.float32)
        outputs = model(source=data, label=label)
        Arcfaceoutput = arcface(outputs[0], label)

        loss = loss_function(Arcfaceoutput, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_teacher(student_model=model, teacher_model=teacher_model)
        scheduler.step()  # 更新学习率

        if (batch_idx + 1) % 100 == 0:
            # 打印当前学习率
            lr = optimizer.param_groups[0]['lr']
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLearning Rate = {:.7f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data, lr))


def test(model, test_loader, optimizer, epoch):
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for data, label in test_loader:
            label = label.to(device)
            data = torch.squeeze(data)
            data = data.to(device)
            data = torch.tensor(data, dtype=torch.float32)
            outputs = model(source=data, label=label)
            Arcfaceoutput = arcface(outputs[0], label)
            loss = loss_function(Arcfaceoutput, label)

            # 获取预测标签
            _, predicted = torch.max(Arcfaceoutput, 1)  # 获取预测标签（最大概率对应的类）
            _, label = torch.max(label, 1)  # 获取预测标签（最大概率对应的类）
            # 计算正确预测的数量
            correct = (predicted == label).sum().item()

            # 更新总样本数和正确预测数
            total_samples += label.size(0)
            total_accuracy += correct

            # 计算总准确率
        accuracy = total_accuracy / total_samples * 100
        print('Test Loss: Loss: {:.6f}'.format(loss.data))
        print('Test Accuracy: Accuracy: {:.2f}%'.format(accuracy))


class BlockCachedDataset(Dataset):
    def __init__(self, data_paths, label_path, block_size=50000):
        self.data_paths = data_paths
        self.labels = np.load(label_path)
        self.block_size = block_size

        # 使用低内存方式获取每个文件中的样本数量
        self.sample_counts = [self._get_npy_sample_count(p) for p in data_paths]
        self.cumulative_counts = np.cumsum([0] + self.sample_counts)

        # 当前缓存状态
        self.cache_file_idx = None
        self.cache_block_idx = None
        self.cache_data = None

    def _get_npy_sample_count(self, path):
        with open(path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"Unsupported .npy version: {version}")
            return shape[0]

    def __len__(self):
        return len(self.labels)

    def _load_block(self, file_idx, block_idx):
        path = self.data_paths[file_idx]
        data = np.load(path, mmap_mode='r')

        start = block_idx * self.block_size
        end = min(start + self.block_size, data.shape[0])
        return data[start:end]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1
        sample_idx = idx - self.cumulative_counts[file_idx]
        block_idx = sample_idx // self.block_size
        block_offset = sample_idx % self.block_size

        if not (file_idx == self.cache_file_idx and block_idx == self.cache_block_idx):
            self.cache_data = self._load_block(file_idx, block_idx)
            self.cache_file_idx = file_idx
            self.cache_block_idx = block_idx

        raw = self.cache_data[block_offset]  # shape: (192000, 1)
        raw_tensor = torch.tensor(raw, dtype=torch.float32).permute(1, 0)  # [1, 192000]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return raw_tensor, label_tensor


if __name__ == '__main__':
    npydata = "./pretrainNpyData/"
    target_sr = 16000
    data_paths = [
        # "./pretrainNpyData/2019_16000_train_raw.npy",
        # "./pretrainNpyData/2020_16000_train_raw.npy",
        # "./pretrainNpyData/2021_16000_train_raw.npy",
        "./pretrainNpyData/2022_16000_train_raw.npy",
        "./pretrainNpyData/2023_16000_train_raw.npy",
        "./pretrainNpyData/2024_16000_train_raw.npy"
    ]

    label_path = "./pretrainNpyData/y_train_cat_4train.npy"

    dataset = BlockCachedDataset(data_paths=data_paths, label_path=label_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # training parameters
    batch_size = 32
    epochs = 20
    aeons = 1
    alpha = 1
    n_subclusters = 16
    ensemble_size = 100
    idx = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('logs')

    final_results_dev = np.zeros((ensemble_size, 6))
    final_results_eval = np.zeros((ensemble_size, 6))

    for k_ensemble in np.arange(idx, idx + 2):
        checkpoint = torch.load('.\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model2025'])

        model = BEATsWithNewLayer(cfg)
        model.load_state_dict(checkpoint['model2025'], strict=False)

        model = model.to(device)

        teacher_model = BEATsWithNewLayer(cfg)
        teacher_model.load_state_dict(checkpoint['model2025'], strict=False)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # 教师模型不训练
        for p in teacher_model.parameters():
            p.requires_grad = False
        freeze_beats_layers(model, num_layers_to_freeze=6)

        optimizer = optim.AdamW(model.parameters(), lr=1e-5)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * int(dataset.__len__() / batch_size * epochs)),  # 通常 10% 作为 warmup
            num_training_steps=int(dataset.__len__() / batch_size * epochs),
            min_lr=1e-7
        )

        for k in np.arange(aeons):
            print('ensemble iteration: ' + str(k_ensemble + 1))
            print('aeon: ' + str(k + 1))
            # fit model2025
            weight_path = "./pretrainmodel/" + 'wts_' + str(k + 1) + 'k_' + str(target_sr) + '_' + str(
                k_ensemble + 1) + '_new_no-bias.h5'
            if not os.path.isfile(weight_path):
                for idx in range(0, epochs):
                    start = time.time()
                    print('----------------start {} train--- --------------'.format(idx + 1))
                    start1 = time.time()
                    train(model=model, train_loader=dataloader, optimizer=optimizer, epoch=idx)
                    end1 = time.time()
                    print('one epoch spent time:{}'.format(end1 - start1))
                    print('------------------end {} train-----------------'.format(idx + 1))
                    end = time.time()
                    print('Total spent time:{}'.format(end - start))

                torch.save(teacher_model, weight_path)
            else:
                model = torch.load(weight_path)

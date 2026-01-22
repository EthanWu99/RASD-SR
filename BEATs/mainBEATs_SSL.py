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

num_label = 180


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
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

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
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = label
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


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
        # 每 n 次 steps 更新一次参数
        if (batch_idx + 1) % 8 == 0:
            optimizer.step()
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()

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

            _, predicted = torch.max(Arcfaceoutput, 1)
            _, label = torch.max(label, 1)

            correct = (predicted == label).sum().item()

            total_samples += label.size(0)
            total_accuracy += correct

        accuracy = total_accuracy / total_samples * 100
        print('Test Loss: Loss: {:.6f}'.format(loss.data))
        print('Test Accuracy: Accuracy: {:.2f}%'.format(accuracy))


########################################################################################################################
# Load data and compute embeddings
########################################################################################################################
target_sr = 16000

# load train data

print('Loading train data')

dev_data_path = r"F:\DCASE\dcase2023t2\data\dcase2025t2\dev_data\raw"
eval_data_path = r"F:\DCASE\dcase2023t2\data\dcase2025t2\eval_data\raw"
npydata = "./2025npyData/"
load_additional_data = True
process_test = False
categories = os.listdir(dev_data_path)
if os.path.isfile(npydata + str(target_sr) + '_train_raw.npy'):
    train_raw = np.load(npydata + str(target_sr) + '_train_raw.npy')
    train_ids = np.load(npydata + 'train_ids.npy')
    train_files = np.load(npydata + 'train_files.npy')
    train_atts = np.load(npydata + 'train_atts.npy')
    train_domains = np.load(npydata + 'train_domains.npy')
else:
    train_raw = []
    train_ids = []
    train_files = []
    train_atts = []
    train_domains = []
    if load_additional_data == True:
        dicts = [dev_data_path + "/", eval_data_path + "/"]
    else:
        dicts = [dev_data_path + "/"]
    eps = 1e-12
    for dict in dicts:
        for label, category in enumerate(os.listdir(dict)):
            print(category)
            enumPath = dict + category + "/train/"
            for count, file in tqdm(enumerate(os.listdir(enumPath)), total=len(os.listdir(enumPath))):
                if file.endswith('.wav'):
                    # file_path = dict + category + "/train/" + file
                    file_path = enumPath + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    # raw = adjust_size(raw, 192000)
                    raw = librosa.util.pad_center(raw, size=192000)
                    train_raw.append(raw)
                    train_ids.append(category + '_' + file.split('_')[1])
                    train_files.append(file_path)
                    train_domains.append(file.split('_')[2])
                    train_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))

    # reshape arrays and store
    train_ids = np.array(train_ids)
    train_files = np.array(train_files)
    train_raw = np.expand_dims(np.array(train_raw, dtype=np.float16), axis=-1)
    train_atts = np.array(train_atts)
    train_domains = np.array(train_domains)
    np.save(npydata + 'train_ids.npy', train_ids)
    np.save(npydata + 'train_files.npy', train_files)
    np.save(npydata + 'train_atts.npy', train_atts)
    np.save(npydata + 'train_domains.npy', train_domains)
    np.save(npydata + str(target_sr) + '_train_raw.npy', train_raw)

# load evaluation data
# 导入验证集，和训练集位于同一个文件夹内的文件
print('Loading evaluation data')
if os.path.isfile(npydata + str(target_sr) + '_eval_raw.npy'):
    eval_raw = np.load(npydata + str(target_sr) + '_eval_raw.npy')
    eval_ids = np.load(npydata + 'eval_ids.npy')
    eval_normal = np.load(npydata + 'eval_normal.npy')
    eval_files = np.load(npydata + 'eval_files.npy')
    eval_atts = np.load(npydata + 'eval_atts.npy')
    eval_domains = np.load(npydata + 'eval_domains.npy')
else:
    eval_raw = []
    eval_ids = []
    eval_normal = []
    eval_files = []
    eval_atts = []
    eval_domains = []
    eps = 1e-12
    dicts = [dev_data_path + "/"]
    for dict in dicts:
        for label, category in enumerate(os.listdir(dict)):
            print(category)
            enumPath = dict + category + "/test/"
            for count, file in tqdm(enumerate(os.listdir(enumPath)), total=len(os.listdir(enumPath))):
                if file.endswith('.wav'):
                    # file_path = "./dev_data/" + category + "/test/" + file
                    file_path = enumPath + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    # raw = adjust_size(raw, 192000)
                    raw = librosa.util.pad_center(raw, size=192000)
                    eval_raw.append(raw)
                    eval_ids.append(category + '_' + file.split('_')[1])
                    eval_normal.append(file.split('_test_')[1].split('_')[0] == 'normal')
                    eval_files.append(file_path)
                    eval_domains.append(file.split('_')[2])
                    eval_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))

    # reshape arrays and store
    eval_ids = np.array(eval_ids)
    eval_normal = np.array(eval_normal)
    eval_files = np.array(eval_files)
    eval_atts = np.array(eval_atts)
    eval_domains = np.array(eval_domains)
    eval_raw = np.expand_dims(np.array(eval_raw, dtype=np.float16), axis=-1)
    np.save(npydata + 'eval_ids.npy', eval_ids)
    np.save(npydata + 'eval_normal.npy', eval_normal)
    np.save(npydata + 'eval_files.npy', eval_files)
    np.save(npydata + 'eval_atts.npy', eval_atts)
    np.save(npydata + 'eval_domains.npy', eval_domains)
    np.save(npydata + str(target_sr) + '_eval_raw.npy', eval_raw)

# load test data
# 导入测试集，测试集为全新的文件
if load_additional_data == True:
    print('Loading test data')
    categories_test = os.listdir(eval_data_path)
    if os.path.isfile(npydata + str(target_sr) + '_test_raw.npy'):
        test_raw = np.load(npydata + str(target_sr) + '_test_raw.npy')
        test_ids = np.load(npydata + 'test_ids.npy')
        test_files = np.load(npydata + 'test_files.npy')
    else:
        test_raw = []
        test_ids = []
        test_files = []
        eps = 1e-12
        for label, category in enumerate(os.listdir(eval_data_path)):
            print(category)
            data_type = category.split('_')
            for count, file in tqdm(
                    enumerate(os.listdir(
                        eval_data_path + "/" + category + "/test")),
                    total=len(os.listdir(
                        eval_data_path + "/" + category + "/test"))):
                if file.endswith('.wav'):
                    file_path = eval_data_path + "/" + category + "/test/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    raw = librosa.util.pad_center(raw, size=192000)
                    # raw = adjust_size(raw, 192000)  # 288000 or 192000
                    test_raw.append(raw)
                    test_ids.append(category + '_' + file.split('_')[1])
                    test_files.append(file_path)
        # reshape arrays and store
        test_ids = np.array(test_ids)
        test_files = np.array(test_files)
        test_raw = np.expand_dims(np.array(test_raw, dtype=np.float16), axis=-1)
        np.save(npydata + 'test_ids.npy', test_ids)
        np.save(npydata + 'test_files.npy', test_files)
        np.save(npydata + str(target_sr) + '_test_raw.npy', test_raw)

# encode ids as labels
le_4train = LabelEncoder()

source_train = np.array([os.path.basename(file).split('_')[2] == 'source' for file in train_files.tolist()])
source_eval = np.array([os.path.basename(file).split('_')[2] == 'source' for file in eval_files.tolist()])
train_ids_4train = np.array(
    ['###'.join([train_ids[k], train_atts[k], str(source_train[k])]) for k in np.arange(train_ids.shape[0])])
eval_ids_4train = np.array(
    ['###'.join([eval_ids[k], eval_atts[k], str(source_eval[k])]) for k in np.arange(eval_ids.shape[0])])
le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
num_classes_4train = len(np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0)))
train_labels_4train = le_4train.transform(train_ids_4train)
eval_labels_4train = le_4train.transform(eval_ids_4train)

le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
if load_additional_data == True:
    test_labels = le.transform(test_ids)
num_classes = len(np.unique(train_labels))

# distinguish between normal and anomalous samples on development set
all_eval_raw = eval_raw
all_eval_labels = eval_labels
all_eval_ids = eval_ids
# 测试集中的异常信号
unknown_raw = eval_raw[~eval_normal]
# 测试集中异常信号的标签，类别（并不是异常与否）
unknown_labels = eval_labels[~eval_normal]
# 加入属性后测试集中异常信号的标签值
unknown_labels_4train = eval_labels_4train[~eval_normal]
# 测试集中异常信号的文件地址
unknown_files = eval_files[~eval_normal]
# 测试集中异常信号的设备类别原型
unknown_ids = eval_ids[~eval_normal]
# 测试集中异常信号的属于源域还是目标域
unknown_domains = eval_domains[~eval_normal]
# 测试集中是否属于源域的部分的异常信号
source_unknown = source_eval[~eval_normal]
# 测试集信号中的正常信号
eval_raw = eval_raw[eval_normal]
# 测试集中正常信号的标签值
eval_labels = eval_labels[eval_normal]
# 测试集中正常信号的添加属性的标签值
eval_labels_4train = eval_labels_4train[eval_normal]
eval_files = eval_files[eval_normal]
eval_ids = eval_ids[eval_normal]
eval_domains = eval_domains[eval_normal]
source_eval = source_eval[eval_normal]

# training parameters
batch_size = 32
epochs = 20
aeons = 1
alpha = 1
n_subclusters = 16
ensemble_size = 100
idx = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('logs')

final_results_dev = np.zeros((ensemble_size, 6))
final_results_eval = np.zeros((ensemble_size, 6))

pred_eval = np.zeros((eval_raw.shape[0], np.unique(train_labels).shape[0]))
pred_unknown = np.zeros((unknown_raw.shape[0], np.unique(train_labels).shape[0]))
if load_additional_data == True:
    pred_test = np.zeros((test_raw.shape[0], np.unique(train_labels).shape[0]))
pred_train = np.zeros((train_labels.shape[0], np.unique(train_labels).shape[0]))

for k_ensemble in np.arange(idx, idx + 5):
    # prepare scores and domain info
    y_train_cat = to_categorical(train_labels, num_classes)
    y_eval_cat = to_categorical(eval_labels, num_classes)
    y_unknown_cat = to_categorical(unknown_labels, num_classes)

    y_train_cat_4train = to_categorical(train_labels_4train, num_classes_4train)
    y_eval_cat_4train = to_categorical(eval_labels_4train, num_classes_4train)
    y_unknown_cat_4train = to_categorical(unknown_labels_4train, num_classes_4train)

    checkpoint = torch.load('.\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

    # 从大模型开始训练
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model2025'])

    model = BEATsWithNewLayer(cfg)
    model.load_state_dict(checkpoint['model2025'], strict=False)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)


    # 学习率调度器：具有 warm-up 的线性学习率下降
    def lr_lambda(current_step):
        if current_step < 120:
            return float(current_step) / float(max(1, 120))
        return max(0.0, float(10000 - current_step) / float(max(1, 10000 - 120)))


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练数据集转为tensor数据后续转为train_loader
    train_raw = torch.tensor(train_raw)
    y_train_cat_4train = torch.tensor(y_train_cat_4train)

    eval_raw = torch.tensor(eval_raw)
    y_eval_cat_4train = torch.tensor(y_eval_cat_4train)

    unknown_raw = torch.tensor(unknown_raw)
    y_unknown_cat_4train = torch.tensor(y_unknown_cat_4train)
    if load_additional_data == True:
        test_raw = torch.tensor(test_raw)
    # 既用于训练，也用于测试
    eval_dataset = Data.TensorDataset(eval_raw, y_eval_cat_4train)
    train_dataset = Data.TensorDataset(train_raw, y_train_cat_4train)
    unknown_dataset = Data.TensorDataset(unknown_raw, y_unknown_cat_4train)
    if load_additional_data == True:
        test_dataset = Data.TensorDataset(test_raw)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    unkown_loader = torch.utils.data.DataLoader(dataset=unknown_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    # print(model2025.summary())
    for k in np.arange(aeons):
        print('ensemble iteration: ' + str(k_ensemble + 1))
        print('aeon: ' + str(k + 1))
        # fit model2025
        weight_path = "./fullmodel/" + 'wts_' + str(k + 1) + 'k_' + str(target_sr) + '_' + str(
            k_ensemble + 1) + '_new_no-bias.h5'
        if not os.path.isfile(weight_path):
            for idx in range(0, epochs):
                start = time.time()
                print('----------------start {} train--- --------------'.format(idx + 1))
                start1 = time.time()
                train(model=model, train_loader=train_loader, optimizer=optimizer, epoch=idx)
                end1 = time.time()
                print('one epoch spent time:{}'.format(end1 - start1))
                print('------------------end {} train-----------------'.format(idx + 1))
                print('----------------start {} test------------------'.format(idx + 1))
                test(model=model, test_loader=eval_loader, optimizer=optimizer, epoch=idx)
                end = time.time()
                print('Total spent time:{}'.format(end - start))
                print('------------------end {} test------------------'.format(idx + 1))

            torch.save(model, weight_path)
        else:
            model = torch.load(weight_path)

        emb_model = model
        emb_model.eval()
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        if load_additional_data == True:
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

        unkown_loader = torch.utils.data.DataLoader(dataset=unknown_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        with torch.no_grad():
            eval_embs = []
            train_embs = []
            test_embs = []
            unknown_embs = []
            label = np.zeros((batch_size, 154))
            label = torch.tensor(label)
            label = label.to(device)
            if load_additional_data == True:
                for inputs in test_loader:
                    inputs = torch.squeeze(inputs[0])
                    inputs = inputs.to(device)
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    _, output = emb_model(source=inputs, label=label)
                    test_embs.append(output.cpu().numpy())

            for inputs, label in train_loader:
                inputs = torch.squeeze(inputs)
                inputs = inputs.to(device)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                _, output = emb_model(source=inputs, label=label)
                train_embs.append(output.cpu().numpy())

            for inputs, label in eval_loader:
                inputs = torch.squeeze(inputs)
                inputs = inputs.to(device)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                _, output = emb_model(source=inputs, label=label)
                eval_embs.append(output.cpu().numpy())

            for inputs, label in unkown_loader:
                inputs = torch.squeeze(inputs)
                inputs = inputs.to(device)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                _, output = emb_model(source=inputs, label=label)
                unknown_embs.append(output.cpu().numpy())

            eval_embs = np.concatenate(eval_embs, axis=0)
            train_embs = np.concatenate(train_embs, axis=0)
            unknown_embs = np.concatenate(unknown_embs, axis=0)
            if load_additional_data == True and process_test == True:
                test_embs = np.concatenate(test_embs, axis=0)
        # test_embs = emb_model.predict([test_raw, np.zeros((test_raw.shape[0], num_classes_4train))],
        #                               batch_size=batch_size)

        # length normalization
        # ######
        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        # X_tsne = tsne.fit_transform(train_embs)
        # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        # converted_array = np.argmax(y_train_cat_4train, axis=1)
        # # converted_array=np.repeat(np.arange(1,17),1000)
        # # plt.figure(figsize=(8, 8), dpi=1000)
        # # for i in range(X_norm.shape[0]):
        # #     plt.text(X_norm[i, 0], X_norm[i, 1], str(converted_array[i]), color=palette[converted_array[i]],
        # #              fontdict={'weight': 'bold', 'size': 2})
        # #
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('fig.jpg', dpi=1000, bbox_inches='tight', format='jpg')
        # # plt.show()
        # # 使用Seaborn的调色板生成154种不同的颜色
        # palette = sns.color_palette('hsv', y_train_cat_4train.shape[1])
        # colors = np.array([palette[label] for label in converted_array])
        #
        # # 创建散点图
        # plt.figure(figsize=(15, 15), dpi=1000)
        # scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=colors, s=10, cmap='hsv')
        #
        # # 计算每个类别的中心点
        # for i in range(y_train_cat_4train.shape[1]):
        #     class_points = X_norm[converted_array == i]
        #     if len(class_points) > 0:
        #         center = class_points.mean(axis=0)
        #         plt.text(center[0], center[1], str(i), fontsize=12, ha='center', va='center', color='black',
        #                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        #
        # # 设置图形标题和轴标签
        # # plt.title('The t-SNE reduced 2D viewable view of embedding')
        # # plt.xlabel('X-axis')
        # # plt.ylabel('Y-axis')
        # plt.xticks([])
        # plt.yticks([])
        # # 保存高分辨率JPG图片
        # plt.savefig('scatter_plot_with_class_centers.jpg', dpi=1000, bbox_inches='tight', format='jpg')
        #
        # # 显示图形
        # plt.show()
        ##
        x_train_ln = length_norm(train_embs)
        x_eval_ln = length_norm(eval_embs)
        if load_additional_data == True:
            x_test_ln = length_norm(test_embs)
        x_unknown_ln = length_norm(unknown_embs)

        for j, lab in tqdm(enumerate(np.unique(train_labels))):
            # 计算均值计算余弦距离异常值得分
            cat = le.inverse_transform([lab])[0]
            means_source_ln = []
            for jj, lablab in enumerate(np.unique(train_labels_4train[source_train * (train_labels == lab)])):
                means_source_ln.append(np.mean(x_train_ln[train_labels_4train == lablab], axis=0))
            means_source_ln2 = np.array(means_source_ln)
            means_target_ln = x_train_ln[~source_train * (train_labels == lab)]
            # means_ln = np.vstack([means_source_ln2, means_target_ln])
            means_ln = x_train_ln[(train_labels == lab)]
            eval_cos = compute_gwrp_scaled_scores(x_eval_ln[eval_labels == lab], means_ln, 0.9, 16).reshape(-1, 1)
            eval_cos_knn = compute_knn_scaled_scores(x_eval_ln[eval_labels == lab], means_ln, 16).reshape(-1,
                                                                                                          1)
            eval_cos = eval_cos + eval_cos_knn
            unknown_cos = compute_gwrp_scaled_scores(x_unknown_ln[eval_labels == lab], means_ln, 0.9, 16).reshape(
                -1, 1)
            unknown_cos_knn = compute_knn_scaled_scores(x_unknown_ln[eval_labels == lab], means_ln,
                                                        16).reshape(-1, 1)
            unknown_cos = unknown_cos + unknown_cos_knn
            if load_additional_data == True:
                test_cos = compute_gwrp_scaled_scores(x_test_ln[test_labels == lab], means_ln, 0.9, 16).reshape(-1,
                                                                                                                1)
                test_cos_knn = compute_knn_scaled_scores(x_test_ln[test_labels == lab], means_ln, 16).reshape(-1, 1)
                test_cos = test_cos + test_cos_knn
            train_cos = compute_gwrp_scaled_scores(x_train_ln[train_labels == lab], means_ln, 0.9, 16).reshape(-1,
                                                                                                               1)
            train_cos_knn = compute_knn_scaled_scores(x_train_ln[train_labels == lab], means_ln, 16).reshape(-1, 1)
            train_cos = train_cos + train_cos_knn
            if np.sum(eval_labels == lab) > 0:
                pred_eval[eval_labels == lab, j] = np.maximum(pred_eval[eval_labels == lab, j],
                                                              np.min(eval_cos, axis=-1))
                pred_unknown[unknown_labels == lab, j] = np.maximum(pred_unknown[unknown_labels == lab, j],
                                                                    np.min(unknown_cos, axis=-1))
            if load_additional_data == True:
                if np.sum(test_labels == lab) > 0:
                    pred_test[test_labels == lab, j] = np.maximum(pred_test[test_labels == lab, j],
                                                                  np.min(test_cos, axis=-1))

            pred_train[train_labels == lab, j] = np.maximum(pred_train[train_labels == lab, j],
                                                            np.min(train_cos, axis=-1))
        # print results for development set
        print(
            '#######################################################################################################')
        print('DEVELOPMENT SET')
        print(
            '#######################################################################################################')
        # 添加result输出
        csv_lines = []
        result_column_dict = {
            "source_target": ["section", "AUC", "AUC (source)", "AUC (target)", "pAUC", "pAUC (source)",
                              "pAUC (target)",
                              "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                              "F1 score (source)", "F1 score (target)"]}
        csv_lines.append(result_column_dict["source_target"])
        aucs = []
        p_aucs = []
        aucs_source = []
        p_aucs_source = []
        aucs_target = []
        p_aucs_target = []
        for j, cat in enumerate(np.unique(eval_ids)):
            y_pred = np.concatenate([pred_eval[eval_labels == le.transform([cat]), le.transform([cat])],
                                     pred_unknown[unknown_labels == le.transform([cat]), le.transform([cat])]],
                                    axis=0)
            y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                                     np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))
            source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])],
                                         source_unknown[unknown_labels == le.transform([cat])]], axis=0)
            auc = roc_auc_score(y_true[source_all], y_pred[source_all])
            p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
            aucs_source.append(auc)
            p_aucs_source.append(p_auc)
            print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            auc_s = auc
            p_auc_s = p_auc
            auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
            p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
            aucs_target.append(auc)
            p_aucs_target.append(p_auc)
            print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            auc_t = auc
            p_auc_t = p_auc
            AUC = hmean([auc_s, auc_t])
            pAUC = hmean([p_auc_t, p_auc_s])
            # shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred[source_all])
            # decision_threshold = scipy.stats.gamma.ppf(q=0.9, a=shape_hat, loc=loc_hat, scale=scale_hat)
            train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
            decision_threshold = np.percentile(train_scores, q=90)
            tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true[source_all],
                                                              [1 if x > decision_threshold else 0 for x in
                                                               y_pred[source_all]]).ravel()
            prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
            recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
            f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)
            # shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred[~source_all])
            # decision_threshold = scipy.stats.gamma.ppf(q=0.9, a=shape_hat, loc=loc_hat, scale=scale_hat)
            tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true[~source_all],
                                                              [1 if x > decision_threshold else 0 for x in
                                                               y_pred[~source_all]]).ravel()
            prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
            recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
            f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
            # 放入result
            csv_lines.append([str(cat), AUC,
                              auc_s, auc_t, pAUC, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
            csv_lines.append([])
        print('####################')
        aucs = np.array(aucs)
        p_aucs = np.array(p_aucs)
        for cat in categories:
            mean_auc = hmean(aucs[np.array(
                [eval_id.split('_')[0] + '_' + eval_id.split('_')[1] for eval_id in
                 np.unique(eval_ids)]) == cat + '_00'])
            print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
            mean_p_auc = hmean(p_aucs[np.array(
                [eval_id.split('_')[0] + '_' + eval_id.split('_')[1] for eval_id in
                 np.unique(eval_ids)]) == cat + '_00'])
            print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
        print('####################')
        for cat in categories:
            mean_auc = hmean(aucs[np.array(
                [eval_id.split('_')[0] + '_' + eval_id.split('_')[1] for eval_id in
                 np.unique(eval_ids)]) == cat + '_00'])
            mean_p_auc = hmean(p_aucs[np.array(
                [eval_id.split('_')[0] + '_' + eval_id.split('_')[1] for eval_id in
                 np.unique(eval_ids)]) == cat + '_00'])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc_source = hmean(aucs_source)
        print('mean AUC for source domain: ' + str(mean_auc_source * 100))
        mean_p_auc_source = hmean(p_aucs_source)
        print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
        mean_auc_target = hmean(aucs_target)
        print('mean AUC for target domain: ' + str(mean_auc_target * 100))
        mean_p_auc_target = hmean(p_aucs_target)
        print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))
        dev_office_score = hmean([mean_auc_source, mean_auc_target, mean_p_auc])
        final_results_dev[k_ensemble] = np.array(
            [mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])
        csv_lines.append(['mean AUC for source domain', 'mean pAUC for source domain',
                          'mean AUC for target domain', 'mean pAUC for target domain', 'mean AUC', 'mean pAUC'])
        csv_lines.append([mean_auc_source, mean_p_auc_source,
                          mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])
        save_csv(save_file_path='./teams/submission/team_fkie_new_no-bias/result.csv', save_data=csv_lines)
        # print results for eval set
        print(
            '#######################################################################################################')
        print('EVALUATION SET')
        print(
            '#######################################################################################################')
        aucs = []
        p_aucs = []
        aucs_source = []
        p_aucs_source = []
        aucs_target = []
        p_aucs_target = []
        for j, cat in enumerate(np.unique(test_ids)):
            y_pred = pred_test[test_labels == le.transform([cat]), le.transform([cat])]
            y_true = np.array(pd.read_csv(
                './dcase2024_task2_evaluator-main/ground_truth_data/ground_truth_' + cat.split('_')[
                    0] + '_section_' +
                cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 1)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))
            source_all = np.array(pd.read_csv(
                './dcase2024_task2_evaluator-main/ground_truth_domain/ground_truth_' + cat.split('_')[
                    0] + '_section_' +
                cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 0)
            auc = roc_auc_score(y_true[source_all], y_pred[source_all])
            p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
            aucs_source.append(auc)
            p_aucs_source.append(p_auc)
            print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
            p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
            aucs_target.append(auc)
            p_aucs_target.append(p_auc)
            print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
        print('####################')
        aucs = np.array(aucs)
        p_aucs = np.array(p_aucs)
        for cat in categories_test:
            # parts = cat.split('_')
            # parts[3] = 'train_00'
            cat = cat + '_00'
            mean_auc = hmean(aucs[np.array([test_id for test_id in np.unique(test_ids)]) == cat])
            print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
            mean_p_auc = hmean(p_aucs[np.array([test_id for test_id in np.unique(test_ids)]) == cat])
            print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
        print('####################')
        for cat in categories_test:
            # parts = cat.split('_')
            # parts[3] = 'train_00'
            cat = cat + '_00'
            mean_auc = hmean(aucs[np.array([test_id for test_id in np.unique(test_ids)]) == cat])
            mean_p_auc = hmean(p_aucs[np.array([test_id for test_id in np.unique(test_ids)]) == cat])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc_source = hmean(aucs_source)
        print('mean AUC for source domain: ' + str(mean_auc_source * 100))
        mean_p_auc_source = hmean(p_aucs_source)
        print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
        mean_auc_target = hmean(aucs_target)
        print('mean AUC for target domain: ' + str(mean_auc_target * 100))
        mean_p_auc_target = hmean(p_aucs_target)
        print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))
        eval_office_score = hmean([mean_auc_source, mean_auc_target, mean_p_auc])
        final_results_eval[k_ensemble] = np.array(
            [mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])
    # create challenge submission files
    print('creating submission files')
    sub_path = './teams/submission/team_fkie_new_no-bias'
    raw_pred_eval = np.concatenate((pred_eval, pred_unknown), axis=0)
    raw_eval_ids = np.concatenate((eval_ids, unknown_ids), axis=0)
    raw_eval_files = np.concatenate((eval_files, unknown_files), axis=0)
    raw_eval_labels = np.concatenate((eval_labels, unknown_labels), axis=0)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    for j, cat in enumerate(np.unique(raw_eval_ids)):
        # anomaly scores
        file_idx = raw_eval_labels == le.transform([cat])
        results_an = pd.DataFrame()
        results_an['output1'], results_an['output2'] = [[f.split('/')[-1] for f in raw_eval_files[file_idx]],
                                                        [str(s) for s in raw_pred_eval[file_idx, le.transform([cat])]]]
        results_an.to_csv(sub_path + '/anomaly_score_' + cat.split('_')[0] + '_section_' + cat.split('_')[
            -1] + '_' + anomalous_score + str(k_ensemble) + '_train.csv', encoding='utf-8', index=False, header=False)
        # decision results
        train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
        threshold = np.percentile(train_scores, q=90)
        decisions = raw_pred_eval[file_idx, le.transform([cat])] > threshold
        results_dec = pd.DataFrame()
        results_dec['output1'], results_dec['output2'] = [[f.split('/')[-1] for f in raw_eval_files[file_idx]],
                                                          [str(int(s)) for s in decisions]]
        results_dec.to_csv(
            sub_path + '/decision_result_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_train.csv',
            encoding='utf-8', index=False, header=False)
    print('office_score_dev:' + str(np.round(dev_office_score * 100, 2)))
    print('office_score_eval:' + str(np.round(eval_office_score * 100, 2)))
    print('####################')
    print('####################')
    print('####################')
    print('final results for development set')
    print(np.round(np.mean(final_results_dev * 100, axis=0), 2))
    print(np.round(np.std(final_results_dev * 100, axis=0), 2))
    print('final results for evaluation set')
    print(np.round(np.mean(final_results_eval * 100, axis=0), 2))
    print(np.round(np.std(final_results_eval * 100, axis=0), 2))
    # print('office_score_dev:' + str(np.round(dev_office_score * 100, 2)) + '\toffice_score_eval:' + str(
    #     np.round(eval_office_score * 100, 2)))
    print('office_score_dev:' + str(np.round(dev_office_score * 100, 2)))
    print('####################')
    print('>>>> finished! <<<<<')
    print('####################')

    # create challenge submission files
    print('creating submission files')
    sub_path = './teams/submission/team_fkie_new_no-bias'
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    for j, cat in enumerate(np.unique(test_ids)):
        # anomaly scores
        file_idx = test_labels == le.transform([cat])
        results_an = pd.DataFrame()
        results_an['output1'], results_an['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                        [str(s) for s in pred_test[file_idx, le.transform([cat])]]]
        results_an.to_csv(sub_path + '/anomaly_score_' + cat.split('_')[0] + '_section_' + cat.split('_')[
            -1] + '_' + anomalous_score + str(k_ensemble) + '.csv', encoding='utf-8', index=False, header=False)

        # decision results
        train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
        threshold = np.percentile(train_scores, q=90)
        decisions = pred_test[file_idx, le.transform([cat])] > threshold
        results_dec = pd.DataFrame()
        results_dec['output1'], results_dec['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                          [str(int(s)) for s in decisions]]
        results_dec.to_csv(
            sub_path + '/decision_result_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '.csv',
            encoding='utf-8', index=False, header=False)

    print('####################')
    print('####################')
    print('####################')
    print('final results for development set')
    print(np.round(np.mean(final_results_dev * 100, axis=0), 2))
    print(np.round(np.std(final_results_dev * 100, axis=0), 2))
    print('final results for evaluation set')
    print(np.round(np.mean(final_results_eval * 100, axis=0), 2))
    print(np.round(np.std(final_results_eval * 100, axis=0), 2))

    print('####################')
    print('>>>> finished! <<<<<')
    print('####################')

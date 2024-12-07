import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


def generate_data(file):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
        pep_codes = []
        labels = []
        peps = []
        for pep in lines:
            pep, label = pep.split(",")
            len_seq = len(pep)
            pep = pep + 'X' * (30-len_seq)
            peps.append(pep)
            labels.append(int(label))
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, torch.tensor(labels)


data, label = generate_data('example.csv')
train_data, train_label = data[:27], label[:27]
test_data, test_label = data[27:], label[27:]

train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)

batch_size = 32
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def load_ind_data(file):
    seqs, labels = generate_data(file)
    dataset = Data.TensorDataset(seqs, labels)
    batch = 32
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    return data_iter


def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


class Ourmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 32
        self.batch_size = 32
        self.emb_dim = 128

        self.embedding = nn.Embedding(24, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.conv_seq_3 = nn.Sequential(
            nn.Conv1d(30, 30, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(30),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )

        self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, dropout=0.2)

        self.block1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1260, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 128))
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.embedding(x)

        output = self.transformer_encoder(x).permute(1, 0, 2)

        output, ht = self.lstm(output)

        output, _ = self.attention(output, output, output)
        output = output.permute(1, 0, 2)

        output = self.conv_seq_3(output)

        output = self.block1(output)
        output = self.block2(output)

        return output


def Loss_model(net, output, labels):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lb = 0.0
    reg_loss = 0
    for param in net.parameters():
        reg_loss += torch.norm(param, p=2)
    total_loss = criterion(output, labels) + l2_lb * reg_loss
    return total_loss


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "cuda" if torch.cuda.is_available() else


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        outputs_cpu = outputs.to(device)
        y_cpu = y.to(device)
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data


net = Ourmodel().to(device)
lr = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
best_acc = 0
EPOCH = 150  # 50, 100
for epoch in range(EPOCH):
    loss_ls = []
    t0 = time.time()
    net.train()
    for seq, label in train_iter:
        seq, label = seq.to(device), label.to(device)
        output = net(seq)
        loss = Loss_model(net, output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
    net.eval()
    with torch.no_grad():
        train_performance, train_roc_data, train_prc_data = evaluate(train_iter, net)
        test_performance, test_roc_data, test_prc_data = evaluate(test_iter, net)

    results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss_ls):.5f}\n"
    results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
    results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
               + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        test_performance[0], test_performance[1], test_performance[2], test_performance[3],
        test_performance[4]) + '\n' + '=' * 60
    print(results)
    test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
    if test_acc > best_acc:
        best_acc = test_acc
        best_performance = test_performance
        filename = '{}, {}[{:.3f}].pt'.format('AIP-TranLAC' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
        save_path_pt = os.path.join(filename)
        # torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
        best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            best_performance[0], best_performance[1], best_performance[2], best_performance[3],
            best_performance[4]) + '\n' + '=' * 60
        print(best_results)
        best_ROC = test_roc_data
        best_PRC = test_prc_data

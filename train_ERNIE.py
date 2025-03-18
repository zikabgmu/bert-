# coding: UTF-8
# 导入模块部分
import os
import time
import torch
import warnings
import numpy as np # 处理数组和数值计算
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
from sklearn import metrics
from datetime import timedelta
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
PAD, CLS = '[PAD]', '[CLS]'


# 数据集函数
def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
# 数据清理
                lin = line.strip()
                if not lin:
                    continue

                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))

        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


'''
def build_dataset(config): # 主函数，用于构建训练集、验证集和测试集

    def load_dataset(path, pad_size=32): # 嵌套函数 load_dataset：加载一个数据文件，并将其处理为适合模型的格式
        contents = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
# 数据清理
                lin = line.strip()
                if not lin:
                    continue
# 拆分内容和标签
                try:
                    content, label = lin.split('\t')
                except ValueError:
                    print(f"Warning: Skipping invalid line: {lin}")
                    continue
# 分词
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
# 转换为ID
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

# 处理长度
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
# 存储结果
                contents.append((token_ids, int(label), seq_len, mask))
# 加载数据集
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test

'''
# 初始化
class DatasetIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
# 转换为Tensor
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

# 迭代批次
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 日志记录类：Logger
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

# 日志记录函数
    def log(self, *value, end="\n"):
        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
# 定义模型：Model 类
# BERT参数
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

# 前向传播
    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, token_type_ids=None, return_dict=False)
        out = self.fc(pooled)
        return out

logger = Logger(os.path.join("ERNIE/datas/log", "log.txt"))

# 训练函数：train
    # train：主要训练过程，包含优化器设置、损失计算、模型保存等
    # logger.log(...)：记录训练配置的相关信息（如模型名称、设备、超参数）
def train(config, model, train_iter, dev_iter, test_iter):
    train_acc = 0  # 初始化
    dev_loss = 0.0  # 初始化 dev_loss
    dev_acc = 0  # 初始化 dev_acc
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("dropout", config.dropout)
    logger.log("Max Sequence Length:", config.pad_size)
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

# 初始化优化器和学习率调度器
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # AdamW：
    optimizer = AdamW (optimizer_grouped_parameters, lr=config.learning_rate)

# 学习率调度策略
    total_step = len(train_iter) * config.num_epochs
    num_warmup_steps = round(total_step * 0.1)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_step)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()
# 训练循环
    for epoch in range(config.num_epochs):
        logger.log('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
# 训练性能的实时监控
            writer.add_scalar('Train/Loss', loss.item(), total_batch)
            dev_acc = 0.0
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 禁用梯度计算
                for batch in dev_iter:
                    inputs, labels = batch
                    outputs = model(inputs)  # 获取模型输出
                    predictions = outputs.argmax(dim=1)  # 根据模型输出计算预测类别
                    correct_predictions = (predictions == labels).sum().item()  # 计算正确预测的数量
                    dev_acc += correct_predictions  # 更新 dev_acc
                    avg_dev_acc = dev_acc / len(dev_iter)  # 计算平均准确率
                    writer.add_scalar('Validation/Accuracy', avg_dev_acc, total_batch)
#梯度计算与更新
            loss.backward() # 反向传播 计算梯度
            optimizer.step() # 根据梯度更新模型参数
            scheduler.step() # 更新学习率

# 提前停止机制（Early Stopping）
            # 如果验证集的效果长期没有提升 提前结束训练
            if total_batch % 100 == 0:
                writer.add_scalar('Train/Accuracy', train_acc, total_batch)
                writer.add_scalar('Validation/Loss', dev_loss, total_batch)
                writer.add_scalar('Validation/Accuracy', dev_acc, total_batch)
                correct_predictions = (predictions == labels).sum().item()  # 计算正确预测的数量
                train_acc += correct_predictions

                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:       # 验证准确率高的模型保存
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
                      '  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logger.log(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    test(config, model, test_iter)
# 测试函数 test
def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logger.log(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    logger.log(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    plot_confusion_matrix(test_confusion, config.class_list)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

# 绘制图表的代码
def plot_confusion_matrix(confusion, class_list):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_list, yticklabels=class_list)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
# 评估函数：evaluate
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

# 配置类
class Config(object): # config ：存储模型 数据路径和超参数配置

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        # 训练集 验证集 测试集的路径
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding="utf-8").readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型与超参数配置
        self.require_improvement = 10000 # 若超过1000batch效果还没提升，则提前结束训练 10000
        self.num_classes = len(self.class_list) # 分类任务的类别数
        self.num_epochs = 20 # 训练轮数 epoch数 # 5
        self.batch_size = 64 # 本来是 64 每批次样本数 mini-batch大小
        self.dropout = 0.3 # 丢弃率（用于防止过拟合）# 0.9
        self.pad_size = 64  # 每句话处理成的固定长度(短填长切) 64
        self.learning_rate = 0.00005  # 学习率 # 0.0001
        self.bert_path = 'ERNIE/ERNIE_pretrain' # 预训练模型路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # 加载 bert分词器
        print(self.tokenizer)
        self.hidden_size = 768 # bert输出的隐藏层大小

# 主函数 _main_
if __name__ == '__main__':
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    dataset = 'ERNIE/datas'
    config = Config(dataset)

    # 设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的 （控制随机性）
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样 （保证CUDA的计算结果稳定）

    start_time = time.time()
    writer.close()

# 加载数据
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

# 训练模型
    model = Model(config).to(config.device)
    optimizer: AdamW | AdamW = AdamW(model.parameters(), lr=0.001, no_deprecation_warning=True)
    train(config, model, train_iter, dev_iter, test_iter)
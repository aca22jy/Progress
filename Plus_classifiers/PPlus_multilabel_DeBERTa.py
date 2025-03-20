import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, brier_score_loss, recall_score, precision_score, roc_auc_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
# 添加DeBERTa导入
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig
from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config
from transformers import AutoModel, AutoConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import cuda
import os
device = 'cuda' if cuda.is_available() else 'cpu'




parser = argparse.ArgumentParser()
parser.add_argument("--test", default = False, action='store_true')
parser.add_argument("--epoch", "-e", default=10, type=int)
parser.add_argument("--max_len", "-m", default=512, type=int)
parser.add_argument("--learning_rate", "-l", type=float, default=1e-05)
parser.add_argument("--train_batch_size", "-t", default=28, type=int)
parser.add_argument('--journal_name', '-j', action = 'store_true')
parser.add_argument("--bert_model", "-b", default='microsoft/deberta-v3-base')  # 修改默认值为DeBERTa模型
# parser.add_argument("--", "-t", default=16, type=int, action = 'store_true')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate



if args.test == True:

    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[2:11]].values.tolist()
    new_df = df[['text', 'list']].copy()
    new_df = new_df.sample(300)
    results_directory = '../results/'
    VALID_BATCH_SIZE = 4
    TRAIN_BATCH_SIZE = 8
    MAX_LEN = 20
    results_directory = '../results/'

else:
    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'JN','Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    # if args.journal_name == True:
    #     df['text'] = df.PaperTitle + ' ' + df.JN + ' ' + df.Abstract


    # else:
    df['text'] = df.PaperTitle + ' ' + df.Abstract

    df['list'] = df[df.columns[3:12]].values.tolist()
    new_df = df[['text', 'list']].copy()
    results_directory = '../results/'
    VALID_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = args.train_batch_size
    results_directory = '../results/'


print(df.select_dtypes(include=['number']).mean())
LABEL_NUM = 9
list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']  # 添加这一行
tokenizer = AutoTokenizer.from_pretrained(args.bert_model)


# 计算每个类别的正例权重 (类别频率的倒数)
class_frequencies = np.array([0.419483, 0.413022, 0.149105, 0.488569, 0.029324, 
                             0.254970, 0.394632, 0.077038, 0.307654])
                             
# 可以选择限制权重范围，避免极端值
# 更积极的权重计算
weights = np.log1p(1.0 / class_frequencies) * 3  # 对数放大+线性系数
weights = np.clip(weights, 1.0, 30.0)  # 提高上限至30

# 为极低频类别设置特殊权重
rare_threshold = 0.05
for i, freq in enumerate(class_frequencies):
    if freq < rare_threshold:  # Religion, Social等
        weights[i] = 30.0  # 直接给予最大权重

# 为特定问题类别增加权重
problem_categories = [2, 4, 7]  # Occupation, Religion, Social
for i in problem_categories:
    weights[i] = weights[i] * 1.5  # 进一步提高权重


# 特别提高Social类别的权重上限
weights[7] = 45.0  # Social类别的索引是7

# 转换为PyTorch张量并移至正确设备
pos_weights = torch.tensor(weights, dtype=torch.float).to(device)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # 替换过时的pad_to_max_length参数
            truncation=True,  # 明确启用截断
             return_token_type_ids='deberta' not in args.bert_model.lower()  # DeBERTa处理方式
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # 第132-135行左右
        
        if 'roberta' in args.bert_model or 'deberta' in args.bert_model.lower():
            token_type_ids = [0] * len(ids)  # DeBERTa与RoBERTa类似，不使用token_type_ids
        else:
            token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'text': text
        }


# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


class BERT_multilabel(torch.nn.Module):
    def __init__(self):
        super(BERT_multilabel, self).__init__()
        self.l1 = AutoModel.from_pretrained(args.bert_model)
        self.l2 = torch.nn.Dropout(0.3)
        # 动态设置隐藏层维度，适应不同模型大小
        if "large" in args.bert_model:
            hidden_size = 1024
        elif "base" in args.bert_model:
            hidden_size = 768
        else:
            # 自动获取隐藏层大小
            config = AutoConfig.from_pretrained(args.bert_model)
            hidden_size = config.hidden_size
        self.l3 = torch.nn.Linear(hidden_size, LABEL_NUM)

    def forward(self, ids, mask, token_type_ids):
        # DeBERTa与RoBERTa类似，不使用token_type_ids
        if 'deberta' in args.bert_model.lower() or 'roberta' in args.bert_model.lower():
            output_1 = self.l1(ids, attention_mask=mask)
        else:
            output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        # 获取[CLS]对应的向量
        if hasattr(output_1, "pooler_output") and output_1.pooler_output is not None:
            pooled_output = output_1.pooler_output
        else:
            # DeBERTa-v3等模型可能需要手动池化获取[CLS]表示
            pooled_output = output_1.last_hidden_state[:, 0]
            
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        return output




# 为极低频类别增加gamma值
def loss_fn(outputs, targets):
    # 为不同类别设置不同gamma值
    # 为不同类别设置不同gamma值
    gammas = [2.0] * LABEL_NUM
    gammas[4] = 3.0  # Religion使用更高gamma
    gammas[7] = 3.5  # Social使用略高gamma
    
    # 添加标签平滑参数，可以针对不同数据集调整
    smoothing = 0.1  # 平滑系数，通常在0.05-0.2之间
    
    return focal_loss_with_weights_dynamic(outputs, targets, gammas, smoothing)

def focal_loss_with_weights_dynamic(outputs, targets, gammas,smoothing):
    """同时支持类别权重、不同gamma值和标签平滑的Focal Loss"""
    # 初始化各类别的损失
    losses = []
    
    # 应用标签平滑
    # 硬标签0变为smoothing/2，硬标签1变为1-smoothing/2
    smoothed_targets = targets.clone()
    smoothed_targets = smoothed_targets * (1 - smoothing) + 0.5 * smoothing
    
    # 对每个类别单独计算Focal Loss
    for i in range(len(gammas)):
        # 获取当前类的输出和目标
        curr_output = outputs[:, i]
        curr_target = smoothed_targets[:, i]  # 使用平滑后的标签
        
        # 计算当前类的BCE损失
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights[i:i+1], reduction='none')(
            curr_output, curr_target)
        
        # 计算预测概率
        prob = torch.sigmoid(curr_output)
        # 对pt计算也使用原始targets，而非平滑后的标签
        pt = prob * targets[:, i] + (1 - prob) * (1 - targets[:, i])
        
        # 使用当前类的gamma值
        curr_gamma = gammas[i]
        focal_weight = (1 - pt) ** curr_gamma
        
        # 计算加权损失并添加
        curr_loss = (focal_weight * bce).mean()
        losses.append(curr_loss)
    
    # 返回所有类别损失的总和
    return sum(losses) / len(losses)



# 1. 模型定义与初始化部分保持不变
model = BERT_multilabel()
model.to(device)
# 差分学习率设置 - 预训练层较小学习率，分类层较大学习率
optimizer = AdamW([
    {'params': model.l1.parameters(), 'lr': LEARNING_RATE/10},  # 预训练层使用较小学习率
    {'params': list(model.l2.parameters()) + list(model.l3.parameters()), 
     'lr': LEARNING_RATE}  # 分类层使用标准学习率
], weight_decay=0.01)  # AdamW的典型权重衰减值

# 2. 早停机制参数设置
patience = 8  # 允许连续多少个epoch没有改进
best_f1 = 0   # 跟踪最佳F1分数
counter = 0   # 计数器：连续没有改进的epoch数


# 创建包含详细参数信息的模型文件名
def create_model_name():
    # 基本模型类型
    if "roberta" in args.bert_model:
        model_type = "roberta"
    elif "deberta" in args.bert_model.lower():
        model_type = "deberta"
    elif "bert-base" in args.bert_model:
        model_type = "bert"
    else:
        model_type = "custom"
    
    # 模型具体变种
    if "base" in args.bert_model:
        variant = "base"
    elif "large" in args.bert_model:
        variant = "large"
    else:
        variant = "custom"
    
    # 核心参数字符串
    params_str = f"bs{args.train_batch_size}_lr{args.learning_rate:.1e}_ep{args.epoch}"
    
    # 生成最终文件名
    return f"{results_directory}/{model_type}_{variant}_{params_str}.pt"
    

    

# 在训练开始前定义模型保存路径
best_model_path = create_model_name()
print(f"模型将保存为: {best_model_path}")
early_stop = False  # 是否触发早停

# 梯度累积设置
accumulation_steps = 8  # 累积4个批次
effective_batch_size = args.train_batch_size * accumulation_steps
print(f"使用梯度累积: {accumulation_steps}步, 有效批次大小: {effective_batch_size}")

# 添加以下代码到您的训练循环前
num_training_steps = len(training_loader) * EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)  # 10%预热期

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps
)


# 3. 先定义训练与验证函数
def train_multilabel(epoch):
    print(epoch)
    model.train()
    optimizer.zero_grad()  # 在整个循环外部清零梯度
    accumulated_loss = 0
    
    for i, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        
        outputs = model(ids, mask, token_type_ids)
        # 缩放损失值，使总梯度大小与常规训练相当
        loss = loss_fn(outputs, targets) / accumulation_steps
        accumulated_loss += loss.item() * accumulation_steps  # 累积原始损失
        loss.backward()
        
        # 每处理accumulation_steps个批次才更新一次权重
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()
    
    # 处理最后不足accumulation_steps的批次
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()  # 更新学习率
        optimizer.zero_grad()
        
    print(f"Average loss: {accumulated_loss / (i+1):.4f}")

# 4. 定义验证函数 - 这个必须在训练循环前定义
def validation_multilabel(model):
    model = model
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    text_list = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            text = data['text']
            text_list = text_list + text
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets, text_list

# 5. 然后才是训练循环
for epoch in range(EPOCHS):
    train_multilabel(epoch)
    
    # 每个epoch后验证
    outputs, targets, _ = validation_multilabel(model)
    current_f1 = metrics.f1_score(targets, 
                                 [[np.round(float(i)) for i in nested] for nested in outputs], 
                                 average='micro')
    
    print(f"Epoch {epoch}: F1 score = {current_f1:.4f}")
    
    # 检查是否有改进
    if current_f1 > best_f1:
        print(f"F1 improved from {best_f1:.4f} to {current_f1:.4f}! Saving model...")
        best_f1 = current_f1
        # 保存最佳模型
        torch.save(model.state_dict(), best_model_path)
        counter = 0  # 重置计数器
    else:
        counter += 1
        print(f"F1 did not improve. counter: {counter}/{patience}")
        
        # 检查是否应该早停
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            early_stop = True
            break

# 6. 训练结束后加载最佳模型
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))


multilabel_prod, targets, text_list = validation_multilabel(model)
multilabel_prod_array = np.array(multilabel_prod)
# multilabel_prod_array = np.array([np.array(xi) for xi in multilabel_prod])
multilabel_pred = [[np.round(float(i)) for i in nested] for nested in multilabel_prod]
multilabel_pred_array = np.array(multilabel_pred)

testing_results = pd.DataFrame(list(zip(text_list, targets, multilabel_pred, multilabel_prod)),
                               columns =['Text', 'Ground truth', 'Prediction', 'Probability'])


if 'scibert' in args.bert_model:
    results_df_name = 'scibert_' + str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'
elif 'roberta' in args.bert_model:
    results_df_name = 'roberta_' + str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'
elif 'deberta' in args.bert_model.lower():
    results_df_name = 'deberta_' + str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'
else:
    results_df_name = str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'

if args.journal_name == True:
    results_df_name = str('JN_') + results_df_name

testing_results.to_csv(results_directory + results_df_name)


multilabel_f1_score_micro = metrics.f1_score(targets, multilabel_pred, average='micro')
multilabel_f1_score_macro = metrics.f1_score(targets, multilabel_pred, average='macro')

multilabel_pred_array = np.array(multilabel_pred)
targets_array = np.array(targets)

def one_label_f1(label_index):
    label_name = list_of_label[label_index]
    pred_label = multilabel_pred_array[:, label_index]
    prob = multilabel_prod_array[:, label_index]
    true_label = targets_array[:, label_index]
    brier = brier_score_loss(true_label, prob)
    recall = recall_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    return label_name, f1, recall, precision, brier

print('---------------------')

all_brier = []
for i, label in enumerate(list_of_label):
    try:
        label_name, f1, recall, precision, brier = one_label_f1(i)
        print(f"{label_name}")
        print(f"f1={f1:.4f}, recall={recall:.4f}, precision={precision:.4f}, brier={brier:.4f}")
        all_brier.append(brier)
    except Exception as e:
        print(f"处理{label}时出错: {e}")

print(all_brier)
avg_brier = sum(all_brier)/len(all_brier)
print('avg brier :')
roc = roc_auc_score(targets_array, multilabel_prod_array)
print('roc: ', roc)

avg_brier = sum(all_brier)/len(all_brier)
print('avg brier :', avg_brier)
# usecols list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion',
#            'Education', 'Socioeconomic', 'Social', 'Plus']

print(f"multilabel F1 Score (Micro) = {multilabel_f1_score_micro}")
print(f"multilabel F1 Score (Macro) = {multilabel_f1_score_macro}")

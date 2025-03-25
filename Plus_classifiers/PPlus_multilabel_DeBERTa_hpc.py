# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, brier_score_loss, recall_score, precision_score, roc_auc_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig
from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config
from transformers import AutoModel, AutoConfig,AutoTokenizer
from torch import cuda
import os
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

device = 'cuda' if cuda.is_available() else 'cpu'

# 参数设置，使用DeBERTa成功的配置
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action='store_true')
parser.add_argument("--epoch", "-e", default=30, type=int)
parser.add_argument("--max_len", "-m", default=512, type=int)
parser.add_argument("--learning_rate", "-l", type=float, default=5e-6)
parser.add_argument("--train_batch_size", "-t", default=7, type=int)
parser.add_argument('--journal_name', '-j', action='store_true')
parser.add_argument("--bert_model", "-b", default='microsoft/deberta-v3-large')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate

if args.test:
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
else:
    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'JN','Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[3:12]].values.tolist()
    new_df = df[['text', 'list']].copy()
    results_directory = '../results/'
    VALID_BATCH_SIZE = 64
    TRAIN_BATCH_SIZE = args.train_batch_size

print(df.select_dtypes(include=['number']).mean())
LABEL_NUM = 9
list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']
tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

class_frequencies = np.array([0.419483, 0.413022, 0.149105, 0.488569, 0.029324, 
                             0.254970, 0.394632, 0.077038, 0.307654])
weights = np.log1p(1.0 / class_frequencies) * 3
weights = np.clip(weights, 1.0, 30.0)
rare_threshold = 0.05
for i, freq in enumerate(class_frequencies):
    if freq < rare_threshold:
        weights[i] = 30.0
problem_categories = [2, 4, 7]
for i in problem_categories:
    weights[i] = weights[i] * 1.5
weights[7] = 45.0
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
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = [0] * len(ids)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'text': text
        }

train_size = 0.8
train_dataset = new_df.sample(frac=train_size, random_state=200)
test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': True}

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 0,
               'pin_memory': True}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class BERT_multilabel(torch.nn.Module):
    def __init__(self):
        super(BERT_multilabel, self).__init__()
        self.l1 = AutoModel.from_pretrained(args.bert_model)
        hidden_size = 1024 if "large" in args.bert_model else 768
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()
        self.layernorm = torch.nn.LayerNorm(hidden_size)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden_size, LABEL_NUM)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask)
        pooled_output = output_1.last_hidden_state[:, 0]
        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.layernorm(x)
        x = self.dropout2(x)
        output = self.l3(x)
        return output

def loss_fn(outputs, targets):
    gammas = [2.0] * LABEL_NUM
    gammas[4] = 3.0
    gammas[7] = 3.5
    return focal_loss_with_weights_dynamic(outputs, targets, gammas)

def focal_loss_with_weights_dynamic(outputs, targets, gammas):
    losses = []
    for i in range(len(gammas)):
        curr_output = outputs[:, i]
        curr_target = targets[:, i]
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights[i:i+1], reduction='none')(curr_output, curr_target)
        prob = torch.sigmoid(curr_output)
        pt = prob * curr_target + (1 - prob) * (1 - curr_target)
        curr_gamma = gammas[i]
        focal_weight = (1 - pt) ** curr_gamma
        curr_loss = (focal_weight * bce).mean()
        losses.append(curr_loss)
    return sum(losses) / len(losses)

model = BERT_multilabel()
model.to(device)
optimizer = torch.optim.Adam([
    {'params': model.l1.parameters(), 'lr': LEARNING_RATE / 2},
    {'params': list(model.dense1.parameters()) + list(model.layernorm.parameters()) + list(model.l3.parameters()), 'lr': LEARNING_RATE}
])

total_steps = len(training_loader) * EPOCHS
num_warmup_steps = 0.15 * total_steps  # 10%预热步数
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
)

patience = 15
best_f1 = 0
counter = 0

def create_model_name():
    model_type = "deberta"
    variant = "large" if "large" in args.bert_model else "base"
    params_str = f"bs{args.train_batch_size}_lr{args.learning_rate:.1e}_ep{args.epoch}"
    return f"{results_directory}/{model_type}_{variant}_{params_str}.pt"

best_model_path = create_model_name()
print(f"模型将保存为: {best_model_path}")
early_stop = False

accumulation_steps = 4
effective_batch_size = args.train_batch_size * accumulation_steps
print(f"使用梯度累积: {accumulation_steps}步, 有效批次大小: {effective_batch_size}")

scaler = GradScaler()

def train_multilabel(epoch):
    print(epoch)
    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0
    
    for i, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        
        with autocast(device_type='cuda'):
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets) / accumulation_steps

        accumulated_loss += loss.item() * accumulation_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            # 先优化器，再调度器
            scheduler.step()  # 移动到这里
            optimizer.zero_grad()
    
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        # 先优化器，再调度器
        scheduler.step()  # 移动到这里
        optimizer.zero_grad()
        
    print(f"Average loss: {accumulated_loss / (i+1):.4f}")

def validation_multilabel(model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    text_list = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            text = data['text']
            text_list += text
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, text_list

for epoch in range(EPOCHS):
    train_multilabel(epoch)
    outputs, targets, _ = validation_multilabel(model)
    current_f1 = metrics.f1_score(targets, [[np.round(float(i)) for i in nested] for nested in outputs], average='micro')
    
    print(f"Epoch {epoch}: F1 score = {current_f1:.4f}")
    
    if current_f1 > best_f1:
        print(f"F1 improved from {best_f1:.4f} to {current_f1:.4f}! Saving model...")
        best_f1 = current_f1
        torch.save(model.state_dict(), best_model_path)
        counter = 0
    else:
        counter += 1
        print(f"F1 did not improve. counter: {counter}/{patience}")
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            early_stop = True
            break

if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

multilabel_prod, targets, text_list = validation_multilabel(model)
multilabel_prod_array = np.array(multilabel_prod)
multilabel_pred = [[np.round(float(i)) for i in nested] for nested in multilabel_prod]
multilabel_pred_array = np.array(multilabel_pred)

testing_results = pd.DataFrame(list(zip(text_list, targets, multilabel_pred, multilabel_prod)),
                               columns=['Text', 'Ground truth', 'Prediction', 'Probability'])

results_df_name = 'deberta_' + str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_' + 'multilabel_results.csv'
if args.journal_name:
    results_df_name = 'JN_' + results_df_name

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
avg_brier = sum(all_brier) / len(all_brier)
print('avg brier :')
roc = roc_auc_score(targets_array, multilabel_prod_array)
print('roc: ', roc)

avg_brier = sum(all_brier) / len(all_brier)
print('avg brier :', avg_brier)

print(f"multilabel F1 Score (Micro) = {multilabel_f1_score_micro}")
print(f"multilabel F1 Score (Macro) = {multilabel_f1_score_macro}")

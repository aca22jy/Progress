import argparse
import numpy as np  # 正确的语法
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, brier_score_loss, recall_score, precision_score, roc_auc_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'




parser = argparse.ArgumentParser()
parser.add_argument("--test", default = False, action='store_true')
parser.add_argument("--epoch", "-e", default=200, type=int)
parser.add_argument("--max_len", "-m", default=512, type=int)
parser.add_argument("--learning_rate", "-l", default=1e-05, action = 'store_true')
parser.add_argument("--train_batch_size", "-t", default=16, type=int)
parser.add_argument('--journal_name', '-j', action = 'store_true')
parser.add_argument("--bert_model", "-b", default='bert-base-uncased')
# parser.add_argument("--", "-t", default=16, type=int, action = 'store_true')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate



if args.test == True:

    df = pd.read_csv('./sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    df['text'] = df.PaperTitle + ' ' + df.Abstract
    df['list'] = df[df.columns[2:11]].values.tolist()
    new_df = df[['text', 'list']].copy()
    new_df = new_df.sample(150)
    results_directory = '../results/'
    VALID_BATCH_SIZE = 4
    TRAIN_BATCH_SIZE = 8
    MAX_LEN = 20
    results_directory = '../results/'

else:
    df = pd.read_csv('./sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'JN','Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    if args.journal_name == True:
        df['text'] = df.PaperTitle + ' ' + df.JN + ' ' + df.Abstract


    else:
        df['text'] = df.PaperTitle + ' ' + df.Abstract

    df['list'] = df[df.columns[3:12]].values.tolist()
    new_df = df[['text', 'list']].copy()
    results_directory = './results/'
    VALID_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = args.train_batch_size
    results_directory = './results/'


print(df.select_dtypes(include=['number']).mean())
LABEL_NUM = 9
if args.bert_model == 'allenai/scibert_scivocab_uncased':
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
else:
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

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
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
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
        self.l1 = transformers.BertModel.from_pretrained(args.bert_model)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, LABEL_NUM)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = output_1[1]
        # print(output_1) # dropout(): argument 'input' (position 1) must be Tensor, not str
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        return output





def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


model = BERT_multilabel()
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train_multilabel(epoch):
    print(epoch)
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        
        optimizer.zero_grad()  # 只需调用一次
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print(loss)

for epoch in range(EPOCHS):
    train_multilabel(epoch)


# define validating
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


multilabel_prod, targets, text_list = validation_multilabel(model)
multilabel_prod_array = np.array(multilabel_prod)
# multilabel_prod_array = np.array([np.array(xi) for xi in multilabel_prod])
multilabel_pred = [[np.round(float(i)) for i in nested] for nested in multilabel_prod]
multilabel_pred_array = np.array(multilabel_pred)

testing_results = pd.DataFrame(list(zip(text_list, targets, multilabel_pred, multilabel_prod)),
                               columns =['Text', 'Ground truth', 'Prediction', 'Probability'])


if args.bert_model == 'allenai/scibert_scivocab_uncased':
    results_df_name = 'scibert_' + str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'
elif args.bert_model == 'bert_base_uncased':
    results_df_name = str(args.max_len) + 'len_' + str(args.train_batch_size) + 'b_' + str(args.epoch) + 'e_'+ 'multilabel_results.csv'
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
    label_name, f1, recall, precision, brier = one_label_f1(i)
    print(label_name)
    print('f1, recall, precision, brier', label_name, f1, recall, precision, brier)
    all_brier.append(brier)

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

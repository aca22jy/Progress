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
from transformers import AutoModel, AutoConfig
from torch import cuda
import os
device = 'cuda' if cuda.is_available() else 'cpu'




parser = argparse.ArgumentParser()
parser.add_argument("--test", default = False, action='store_true')
parser.add_argument("--epoch", "-e", default=30, type=int)
parser.add_argument("--max_len", "-m", default=512, type=int)
parser.add_argument("--learning_rate", "-l", type=float, default=5e-06)
parser.add_argument("--train_batch_size", "-t", default=6, type=int)
parser.add_argument('--journal_name', '-j', action = 'store_true')
parser.add_argument("--bert_model", "-b", default='roberta-large')  # Change default value
# parser.add_argument("--", "-t", default=16, type=int, action = 'store_true')
args = parser.parse_args()

EPOCHS = args.epoch
MAX_LEN = args.max_len
LEARNING_RATE = args.learning_rate



if args.test == True:

    df = pd.read_csv('../sources/ProgressTrainingCombined.tsv', sep='\t',
                     usecols=['PaperTitle', 'Abstract', 'Place', 'Race', 'Occupation', 'Gender', 'Religion',
                              'Education', 'Socioeconomic', 'Social', 'Plus'])
    
    if args.journal_name == True:
        df['text'] = df.PaperTitle + ' ' + df.JN + ' ' + df.Abstract


    else:
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
    if args.journal_name == True:
        df['text'] = df.PaperTitle + ' ' + df.JN + ' ' + df.Abstract


    else:
     df['text'] = df.PaperTitle + ' ' + df.Abstract

    df['list'] = df[df.columns[3:12]].values.tolist()
    new_df = df[['text', 'list']].copy()
    results_directory = '../results/'
    VALID_BATCH_SIZE = 16
    TRAIN_BATCH_SIZE = args.train_batch_size
    results_directory = '../results/'


print(df.select_dtypes(include=['number']).mean())
LABEL_NUM = 9
list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']  
tokenizer = AutoTokenizer.from_pretrained(args.bert_model)


# Calculate positive class weights (inverse of class frequency)
class_frequencies = np.array([0.419483, 0.413022, 0.149105, 0.488569, 0.029324, 
                             0.254970, 0.394632, 0.077038, 0.307654])
                             
# Optionally limit weight range to avoid extreme values
# More aggressive weight calculation
weights = np.log1p(1.0 / class_frequencies) * 3  # Log amplification + linear coefficient
weights = np.clip(weights, 1.0, 30.0)  # Increase upper limit to 30

# Set special weights for very rare classes
rare_threshold = 0.05
for i, freq in enumerate(class_frequencies):
    if freq < rare_threshold:  # Religion, Social, etc.
        weights[i] = 30.0  # Directly assign max weight

# Further increase weights for specific problematic categories
problem_categories = [2, 4, 7]  # Occupation, Religion, Social
for i in problem_categories:
    weights[i] = weights[i] * 1.5  # Further increase weight



# Convert to PyTorch tensor and move to correct device
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
            padding='max_length',  # Replace deprecated pad_to_max_length parameter
            truncation=True,  # Explicitly enable truncation
            return_token_type_ids='roberta' not in args.bert_model  # RoBERTa does not need token_type_ids
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        if 'roberta' in args.bert_model:
          token_type_ids = [0] * len(ids)  # RoBERTa does not use, but keep interface consistent
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
        # Dynamically set hidden layer size to adapt to different model sizes
        hidden_size = 1024 if "large" in args.bert_model else 768
        self.l3 = torch.nn.Linear(hidden_size, LABEL_NUM)

    def forward(self, ids, mask, token_type_ids):
        # RoBERTa does not use token_type_ids
        if 'roberta' in args.bert_model:
            output_1 = self.l1(ids, attention_mask=mask)
        else:
            output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        # Get the vector corresponding to [CLS]
        if hasattr(output_1, "pooler_output"):
            pooled_output = output_1.pooler_output
        elif type(output_1) is tuple:
            pooled_output = output_1[1]
        else:
            pooled_output = output_1.last_hidden_state[:, 0]
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        return output




# Increase gamma value for very rare classes
def loss_fn(outputs, targets):
    # Set different gamma values for different classes
    gammas = [2.0] * LABEL_NUM
    gammas[4] = 3.0  # Higher gamma for Religion
    gammas[7] = 3.5  # Slightly higher gamma for Social
    return focal_loss_with_weights_dynamic(outputs, targets, gammas)

def focal_loss_with_weights_dynamic(outputs, targets, gammas):
    """Support different gamma values for each class in Focal Loss"""
    # Initialize loss for each class
    losses = []
    
    # Compute Focal Loss for each class separately
    for i in range(len(gammas)):
        # Get current class output and target
        curr_output = outputs[:, i]
        curr_target = targets[:, i]
        
        # Compute BCE loss for current class
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights[i:i+1], reduction='none')(
            curr_output, curr_target)
        
        # Compute predicted probability
        prob = torch.sigmoid(curr_output)
        pt = prob * curr_target + (1 - prob) * (1 - curr_target)
        
        # Use current class's gamma value
        curr_gamma = gammas[i]
        focal_weight = (1 - pt) ** curr_gamma
        
        # Compute weighted loss and add
        curr_loss = (focal_weight * bce).mean()
        losses.append(curr_loss)
    
    # Return the average loss of all classes
    return sum(losses) / len(losses)




model = BERT_multilabel()
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# 2. Early stop mechanism parameter setting
patience = 15 # How many consecutive epochs are allowed without improvement
best_f1 = 0 # Keep track of the best F1 score
counter = 0 # counter: number of consecutive epochs without improvement



def create_model_name():
    
    model_type = "roberta" if "roberta" in args.bert_model else "scibert"
    
   
    if "base" in args.bert_model:
        variant = "base"
    elif "large" in args.bert_model:
        variant = "large"
    else:
        variant = "custom"
    
    
    params_str = f"bs{args.train_batch_size}_lr{args.learning_rate:.1e}_ep{args.epoch}"
    
    
    return f"{results_directory}/{model_type}_{variant}_{params_str}.pt"
    

    

# Create a unique model name based on parameters
best_model_path = create_model_name()
print(f"The model will be saved as: {best_model_path}")
early_stop = False  

# Gradient accumulation setting
accumulation_steps = 2  # 4 batches
effective_batch_size = args.train_batch_size * accumulation_steps
print(f"Using gradient accumulation: {accumulation_steps}step, Effective batch size: {effective_batch_size}")


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
        
        outputs = model(ids, mask, token_type_ids)
     
        loss = loss_fn(outputs, targets) / accumulation_steps
        accumulated_loss += loss.item() * accumulation_steps  
        loss.backward()
        
        # Weights are updated only once every batch of calculation_steps is processed.
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Processing of batches with final insufficient accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    print(f"Average loss: {accumulated_loss / (i+1):.4f}")


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


for epoch in range(EPOCHS):
    train_multilabel(epoch)
    

    outputs, targets, _ = validation_multilabel(model)
    current_f1 = metrics.f1_score(targets, 
                                 [[np.round(float(i)) for i in nested] for nested in outputs], 
                                 average='micro')
    
    print(f"Epoch {epoch}: F1 score = {current_f1:.4f}")
    
    # Check for improvements
    if current_f1 > best_f1:
        print(f"F1 improved from {best_f1:.4f} to {current_f1:.4f}! Saving model...")
        best_f1 = current_f1
        # Keep the best model
        torch.save(model.state_dict(), best_model_path)
        counter = 0  
    else:
        counter += 1
        print(f"F1 did not improve. counter: {counter}/{patience}")
        
        # Check if you should stop early
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            early_stop = True
            break

# 6. Loading the best model at the end of training
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

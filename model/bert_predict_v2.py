# https://www.kdnuggets.com/2020/11/tabular-data-huggingface-transformers.html
# https://multimodal-toolkit.readthedocs.io/en/latest/modules/data.html?highlight=load_data#multimodal_transformers.data.load_data

# %%
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from multimodal_transformers.data import load_data


DATA_DIR = Path("../data")
MODEL_UTILS_DIR = Path(f"{DATA_DIR}/model_utils")
MODEL_INPUTS_DIR = Path(f"{DATA_DIR}/model_inputs")
MODEL_RESULTS_DIR = Path(f"{DATA_DIR}/model_results")
MODEL_DIR = Path("../model")

# %%
# w2v_data = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data.pkl")
train_df = pd.read_csv(f"{MODEL_INPUTS_DIR}/w2v_data_train_bertuse.csv", encoding='utf-8-sig').fillna(0)
train_df['index'] = train_df.index
train_df = train_df.astype({col : "float32" for col in train_df.columns.tolist() if train_df[col].dtype=="float64"})
train_df = train_df.astype({col : "int16" for col in train_df.columns.tolist() if train_df[col].dtype=="int64"})
val_df = pd.read_csv(f"{MODEL_INPUTS_DIR}/w2v_data_val_bertuse.csv", encoding='utf-8-sig').fillna(0)
test_df = pd.read_csv(f"{MODEL_INPUTS_DIR}/w2v_data_test_bertuse.csv", encoding='utf-8-sig').fillna(0)

# %%
# 載入預訓練的BERT模型和tokenizer
model_name = "bert-base-chinese"

# not yet：要有index (The label col is expected to contain integers from 0 to N_classes - 1)
label_col = 'index' 
text_col = [
        "title", "metaDescription", 'description',
        'requiredTools', 'recommendedBackground', 'willLearn', 'targetGroup',
        'owner.metaDescription', 'owner_description', 'owner.skills', 'owner.interests',
        ]
numerical_cols = [col for col in train_df.columns.tolist() if col not in text_col]
label_list = ['sales'] 


tokenizer = AutoTokenizer.from_pretrained(model_name)

torch_dataset = load_data(
    data_df=train_df,
    label_col=label_col,
    text_cols=text_col,
    tokenizer=tokenizer,
    numerical_cols=numerical_cols,
    sep_text_token_str=tokenizer.sep_token
)

# %%
config = AutoConfig.from_pretrained(model_name)
tabular_config = TabularConfig(
    num_labels=1,
    numerical_feat_dim=torch_dataset.numerical_feats.shape[1],
    combine_feat_method='weighted_feature_sum_on_transformer_cat_and_numerical_feats',
)
config.tabular_config = tabular_config

# %%
model = AutoModelWithTabular.from_pretrained(model_name, config=config)
training_args = TrainingArguments(
    output_dir="./logs/model_name",
    logging_dir="./logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=32,
    num_train_epochs=1,
    # evaluate_during_training='epoch',
    logging_steps=25,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=torch_dataset
)

trainer.train()

# %%
# 載入和處理數據
data = pd.read_csv("sales_data.csv") # 請替換成您的數據文件路徑

# 假設數據文件中有兩列，一列是文本描述，一列是銷售額
descriptions = data["description"].values
sales = data["sales"].values

# %%
def feature_to_tensor(df, col):
    if "mean" in col:
        return torch.Tensor(df[col].tolist())
    return torch.Tensor(df[col].tolist()).unsqueeze(1)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1) # 回歸模型

feature_cols = [col for col in train_df.columns.tolist() if col != "sales"]
# 使用tokenizer將文本描述轉換成BERT的輸入格式
inputs = tokenizer(
    torch.cat([feature_to_tensor(train_df, col) for col in feature_cols], dim=1).view(-1, 1, 568), 
    padding=True, 
    truncation=True, 
    return_tensors="pt", 
    max_length=128
    )

# %%
# 將銷售額轉換成PyTorch tensor
targets = torch.tensor(sales, dtype=torch.float32).unsqueeze(1)

# 切分數據集為訓練、驗證和測試集
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    TensorDataset(inputs["input_ids"], inputs["attention_mask"], targets),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # 設定隨機種子
)

# 創建DataLoader
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 定義優化器和學習率計劃
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

# 定義損失函數
loss_fn = torch.nn.MSELoss()

# 訓練模型
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

# 驗證模型
model.eval()
total_val_loss = 0.0
with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Validation"):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_val_loss += loss.item()

avg_val_loss = total_val_loss / len(val_dataloader)
print(f"Average validation loss: {avg_val_loss:.4f}")

# 測試模型
model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids, attention_mask, _ = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.logits.squeeze(1).tolist())

# 將預測結果轉換為DataFrame
results = pd.DataFrame({"description": descriptions[-len(predictions):], "sales": predictions})

# 儲存預測結果
results.to_csv("sales_predictions.csv", index=False)

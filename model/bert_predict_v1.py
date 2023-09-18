# %%
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from transformers import BertTokenizer, BertForSequenceClassification

DATA_DIR = Path("../data")
MODEL_UTILS_DIR = Path(f"{DATA_DIR}/model_utils")
MODEL_INPUTS_DIR = Path(f"{DATA_DIR}/model_inputs")

# %%
# w2v_data = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data.pkl")
train_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_train.pkl")
test_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_test.pkl")

# %%
"""
load data
"""
class HahowDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            batch_size: int = 16, 
            num_workers: int = 8
            ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def feature_to_tensor(self, df, col):
        return torch.Tensor(df[col].tolist())
    
    def setup(self, stage: str) -> None:
        feature_cols = [col for col in self.train_df.columns.tolist() if col != "numSoldTickets"]

        self.train_dataset = TensorDataset(
            torch.stack([self.feature_to_tensor(self.train_df, col) for col in feature_cols], dim=1).view(-1, 50, 2),
            torch.tensor(self.train_df['numSoldTickets'].tolist())
            )
        
        self.val_dataset = TensorDataset(
            torch.stack([self.feature_to_tensor(self.val_df, col) for col in feature_cols], dim=1).view(-1, 50, 2),
            torch.tensor(self.val_df['numSoldTickets'].tolist())
            )
        
        self.test_dataset = TensorDataset(
            torch.stack([self.feature_to_tensor(self.test_df, col) for col in feature_cols], dim=1).view(-1, 50, 2),
            torch.tensor(self.test_df['numSoldTickets'].tolist())
            )
    
    def prepare_data(self):
        """讀取 train, val, test dataset
        """
        self.train_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_train.pkl")
        self.val_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_val.pkl")
        self.test_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_test.pkl")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# %%
# 定義PyTorch Lightning模型
class HahowSalesBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-chinese"
            )
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, 1)  # 假設 BERT 的輸出維度為 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 取得[CLS]的特徵向量
        output = self.fc(self.dropout(pooled_output))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs, attention_mask=(inputs != 0).float())
        loss = torch.sqrt(torch.nn.functional.mse_loss(outputs.squeeze(), labels))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs, attention_mask=(inputs != 0).float())
        val_loss = torch.sqrt(torch.nn.functional.mse_loss(outputs.squeeze(), labels))
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs, attention_mask=(inputs != 0).float())
        test_loss = torch.sqrt(torch.nn.functional.mse_loss(outputs.squeeze(), labels))
        self.log("test_loss", test_loss)

# %%
# 初始化模型和訓練器
# 初始化BertTokenizer和BertForSequenceClassification模型
datamodule = HahowDataModule(batch_size=1, num_workers=0)
# tokenizer = BertTokenizer.from_pretrained(
#     pretrained_model_name_or_path='bert-base-chinese',
#     )
# bert_model = BertForSequenceClassification.from_pretrained(
#     pretrained_model_name_or_path='bert-base-uncased', 
#     num_labels=1,
#     )

hahowbert = HahowSalesBERT()
trainer = pl.Trainer(max_epochs=10)  # 假設你有一個GPU可用

# 開始訓練
trainer.fit(hahowbert, datamodule)

# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/raw_text_train.pkl")


# 假設每個文本向量的維度是 n
n = len(train_df['title'][0])
# encoded_texts_tensor = torch.tensor([text for text in train_df['title']], dtype=torch.float64)

# 使用BERT tokenizer轉換encoded_texts_tensor
input_ids = []
attention_masks = []

for encoded_text in train_df['title']:
    encoding = tokenizer(
        # encoded_text.tolist(), 
        train_df['title'].tolist(), 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
        )
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

# 將轉換後的結果轉換為PyTorch Tensor
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# %%
# 超參數設定
batch_size = 32
learning_rate = 1e-4
max_epochs = 10

class HahowSalesBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, input_ids, attention_mask):
        return self.bert_model(input_ids, attention_mask)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs.logits.squeeze(), labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs.logits.squeeze(), labels)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs.logits.squeeze(), labels)
        self.log('test_loss', loss)

# 創建資料集和資料加載器
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(x_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型和Trainer
model = HahowSalesBERT()
trainer = pl.Trainer(max_epochs=max_epochs)

# 開始訓練
trainer.fit(model, train_dataloader, val_dataloader)

# 驗證和測試
trainer.test(test_dataloaders=test_dataloader)
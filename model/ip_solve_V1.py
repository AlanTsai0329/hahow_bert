# %%
import pulp
import pandas as pd
from pathlib import Path
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import dump, load
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = Path("../data")
MODEL_UTILS_DIR = Path(f"{DATA_DIR}/model_utils")
MODEL_INPUTS_DIR = Path(f"{DATA_DIR}/model_inputs")
MODEL_RESULTS_DIR = Path(f"{DATA_DIR}/model_results")

# %%
# 載入模型
class HahowSalesRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HahowSalesRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化RNN的隱藏狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # 前向傳播
        out, _ = self.rnn(x, h0)
        
        # 使用最後一個時間步的輸出來預測
        out = self.fc(out[:, -1, :])
        return out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx): 
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
        return loss
    

checkpoint_path = f"{MODEL_RESULTS_DIR}/sales_rnn_model_V1.ckpt"
input_size = 568
hidden_size = 128
num_layers = 3
output_size = 1
model = HahowSalesRNN.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    )

# %%
# 讀資料、把編碼後、縮放後的數據 reverse 回去
test_df = pd.read_pickle(f"{MODEL_INPUTS_DIR}/w2v_data_test.pkl")
price_sc = load(f'{MODEL_UTILS_DIR}/prices_standscaler_model.bin')
video_len_sc = load(f'{MODEL_UTILS_DIR}/totalVideoLengthInSeconds_standscaler_model.bin')
cate_le = load(f'{MODEL_UTILS_DIR}/課程類別_LabelEncoder_model.bin')

# 類別、價格 reverse standard 回去
reverse_test_df = (
    test_df
    .assign(prices=price_sc.inverse_transform(test_df['prices'].values.reshape(-1, 1)))
    .assign(課程類別=cate_le.inverse_transform(test_df['課程類別'].values.reshape(-1, 1)))
)


# %%
# 創建價格上下限、範圍
all_minmax_price_df = pd.read_csv(f"{DATA_DIR}/min_max_price.csv", encoding='utf-8-sig')
ready_to_be_solve_df = reverse_test_df.iloc[0:2, :]

# 價格上下限
minmax_price = all_minmax_price_df.query("課程類別 == @ready_to_be_solve_df.課程類別.values[0]")

# 範圍
middle_prices = list(range(minmax_price['最低價'].values[0] + 1, minmax_price['最高價'].values[0])) 

# 將範圍插進
price_range_df = (
    ready_to_be_solve_df.append(pd.DataFrame({'prices': middle_prices}), ignore_index=True)
    .assign(sales=lambda df: df['sales'].fillna(0))
    .fillna(method='ffill')
    .sort_values(by='prices')
)
price_range_df


# %%
# 價格、類別 standard 
price_range_df = (
    price_range_df
    .assign(prices=price_sc.transform(price_range_df['prices'].values.reshape(-1, 1)))
    .assign(課程類別=cate_le.transform(price_range_df['課程類別'].values.reshape(-1, 1)))
)

# %%
# 創建 dataset
def feature_to_tensor(df, col):
    if "mean" in col:
        return torch.Tensor(df[col].tolist())
    return torch.Tensor(df[col].tolist()).unsqueeze(1)

feature_cols = [col for col in price_range_df.columns.tolist() if col != "sales"]
test_dataset = TensorDataset(
            torch.cat([feature_to_tensor(price_range_df, col) for col in feature_cols], dim=1).view(-1, 1, 568),
            torch.tensor(price_range_df['sales'].tolist())
            )

test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

# %%
# 輸入模型得到銷售額
model.eval()
test_prediction_list = []
with torch.no_grad():
    for test_batch in test_dataloader:
        test_x, test_y = test_batch
        test_x, test_y = test_x.float(), test_y.float()
        test_prediction = model(test_x)
        if test_y.item() == 0 :
            test_prediction_list.append(round(np.expm1(test_prediction).item(), 0))
        else:
            test_prediction_list.append(round(np.expm1(test_y).item(), 0))

price_range_df['sales'] = test_prediction_list

# %%
# 創建影片長度的成本
# 組成表：價格、銷售額、價格上下限、成本
processed_price_range_df = (
    price_range_df
    .assign(
        prices=price_sc.inverse_transform(price_range_df['prices'].values.reshape(-1, 1)),
        sales=test_prediction_list,
        min_prices=[minmax_price['最低價'].values[0]] * len(price_range_df),
        max_prices=[minmax_price['最高價'].values[0]] * len(price_range_df),
        cost=video_len_sc.inverse_transform(price_range_df['totalVideoLengthInSeconds'].values.reshape(-1, 1)) * 0.11,
        )
    [['prices', 'sales', 'min_prices', 'max_prices', 'cost']]
)


# %%
import pandas as pd
from scipy.optimize import minimize

# 載入CSV資料
data = processed_price_range_df  # 請將'your_data.csv'替換為實際的CSV檔案路徑

# 定義目標函數，這裡我們要最大化利潤，所以是負成本
def objective(x):
    price, sales = x
    cost = data['cost'].values[0]
    return -(price * sales - cost)

# 定義約束條件，包括價格上下限
constraints = (
    {'type': 'ineq', 'fun': lambda x: x[0] - data['min_prices']},  # 價格必須大於等於價格下限
    {'type': 'ineq', 'fun': lambda x: data['max_prices'] - x[0]},  # 價格必須小於等於價格上限
    {'type': 'ineq', 'fun': lambda x: x[1]},  # 銷售額必須大於等於0
    {'type': 'ineq', 'fun': lambda x: data['sales'] - x[1]},  # 銷售額必須大於等於0
    # {'type': 'ineq', 'fun': lambda x: data['sales'] - x[1]} # sales必須小於等於實際sales
    )  

# 初始猜測值
x0 = [data['prices'].mean(), data['sales'].mean()]

# 使用 minimize 函數求解
result = minimize(objective, x0, constraints=constraints)

# 最佳價格和銷售額
best_price, best_sales = result.x

# 最大利潤
max_profit = -result.fun

# 顯示結果
print(f"最佳價格: {best_price}")
print(f"最佳銷售額: {best_sales}")
print(f"最大利潤: {max_profit}")




# %%
# 比較利潤

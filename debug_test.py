#%%
import qlib
from qlib.config import REG_CN
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
# 初始化配置
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_rolling', region=REG_CN)

from qlib.data import D
from qlib.contrib.strategy import TopkDropoutStrategy
# from qlib.contrib.evaluate import (
#     backtest as normal_backtest,
#     risk_analysis,
# )
#%%
# 加载数据
df = D.features(D.instruments('csi300'), ['$open', '$low', '$close'], freq='day', start_time='2020-01-01', end_time='2021-01-01')

# 转换数据为Tensor，适配你的表达式类
stock_data = StockData(data=torch.tensor(df.values), max_backtrack_days=50, max_future_days=0, n_days=len(df), n_stocks=1)

# 定义复杂因子
expr = Mul(
    EMA(Sub(Delta(Mul(Log(Feature('open')), Constant(-30.0)), 50), Constant(-0.01)), 40),
    Mul(Div(Abs(EMA(Feature('low'), 50)), Feature('close')), Constant(0.01))
)

# 生成因子
def generate_tensor_factor(stock_data):
    return expr.evaluate(stock_data).numpy()  # 转换回Numpy数组以适配Qlib

# 使用模型
class TensorFactorModel(AlphaModel):
    def generate_factor(self, df):
        tensor_factor = generate_tensor_factor(stock_data)
        df['custom_factor'] = tensor_factor.squeeze()  # 确保维度正确
        return df[['custom_factor']]

model = TensorFactorModel()
signals = model.generate_factor(df)

# 策略配置
strategy = TopkDropoutStrategy(topk=50, n_drop=5)

# 回测
portfolio_metric, portfolio_daily = normal_backtest(signals, strategy=strategy)

# 输出结果
print(portfolio_metric)
print(portfolio_daily.head())


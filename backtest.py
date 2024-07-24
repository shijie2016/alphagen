from typing import Optional, TypeVar, Callable, Optional
import os
import pickle
import warnings
import pandas as pd
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = None,
        return_report: bool = False
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=5,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result)

        print(report)
        print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


def check_prediction():
    global fig, mean, std
    import matplotlib.pyplot as plt
    # # 筛选特定资产的信号数据
    # specific_instrument_data = data_df.xs('SH600006', level='instrument')
    # specific_instrument_data.plot()
    # 如果信号数据有多列，可以选择一个代表列或多列进行绘制
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    # 创建一个新的图和一个3D子图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    # 将多级索引转换为普通列
    data_flat = data_df.reset_index()
    # 为每个资产和日期生成数值索引
    data_flat['date_idx'] = data_flat['datetime'].factorize()[0]
    data_flat['instrument_idx'] = data_flat['instrument'].factorize()[0]
    # 绘制三维散点图
    ax.scatter(data_flat['date_idx'], data_flat['instrument_idx'], data_flat.iloc[:, 2], c=data_flat.iloc[:, 2],
               cmap='viridis')
    # 设置标签
    ax.set_xlabel('Date Index')
    ax.set_ylabel('Instrument Index')
    ax.set_zlabel('Signal Value')
    # 自定义x轴和y轴的标签显示真实日期和资产名称
    ax.set_xticks(data_flat['date_idx'].unique())
    ax.set_xticklabels(data_flat['datetime'].unique(), rotation=45, ha='right')
    ax.set_yticks(data_flat['instrument_idx'].unique())
    ax.set_yticklabels(data_flat['instrument'].unique())
    plt.title('3D View of Asset Signals Over Time')
    plt.show()
    # %% dingwei 尖峰
    data_flat.rename(columns={data_flat.columns[2]: 'signal'}, inplace=True)
    # 假设你已经有了一个名为 'signal' 的列
    mean = data_flat['signal'].mean()
    std = data_flat['signal'].std()
    # 定位尖峰，这里使用平均值加三倍标准差作为阈值
    spikes = data_flat[data_flat['signal'] > mean + 5 * std]
    print(spikes)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    plt.plot(data_flat['datetime'], data_flat['signal'], label='Signal')
    plt.scatter(spikes['datetime'], spikes['signal'], color='red', label='Spikes')  # 假设尖峰数据已经被找到并存入spikes
    plt.legend()
    plt.title('Signal with Spikes Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Signal Value')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from alphagen_qlib.utils import load_alpha_pool_by_path, load_recent_data
    from alphagen_qlib.calculator import QLibStockDataCalculator

    qlib_backtest = QlibBacktest(benchmark='sh000300', top_k=10, n_drop=2)

    data = StockData(instrument='csi300_filtered',
                     start_time='2022-01-01',
                     end_time='2023-12-31')
    POOL_PATH = 'out/checkpoints/new_csi300_filtered_200_666_20240722135956/20480_steps_pool.json'
    # POOL_PATH = 'out/checkpoints/new_csi300_filtered_20_66_20240719145701/43008_steps_pool.json'
    # POOL_PATH = 'out/checkpoints/new_csi300_filtered_50_666_20240719134812/65536_steps_pool.json'
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)
    calculator = QLibStockDataCalculator(data=data, target=None)
    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    # expr = Mul(EMA(Sub(Delta(Mul(Log(open_),Constant(-30.0)),50),Constant(-0.01)),40),Mul(Div(Abs(EMA(low,50)),close),Constant(0.01)))
    # expr = EMA(low,50)
    # data_df = data.make_dataframe(expr.evaluate(data))
    import numpy as np
    data_df = data.make_dataframe(ensemble_alpha)
    # check_prediction()

    out = qlib_backtest.run(data_df, return_report=True)
    from qlib.contrib.report import analysis_model, analysis_position
    fig = analysis_position.report_graph(out, show_notebook=False)
    fig[0].show(renderer='browser')
# -*- coding: utf-8 -*-
"""
完整版：基于技术指标波动率的市场反转点预测系统
包含Transformer和LSTM双模型支持
改进版：提高模型准确率 - 增加回测可视化功能
"""

# 导入必要的库
import os  # 操作系统接口
import pickle  # 对象序列化
from datetime import datetime  # 日期时间处理
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import yfinance as yf  # 雅虎财经数据下载
import matplotlib.pyplot as plt  # 绘图
import matplotlib.dates as mdates  # 日期格式化
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.metrics import classification_report  # 分类报告
from sklearn.model_selection import TimeSeriesSplit  # 时间序列分割
from sklearn.utils import resample  # 数据重采样
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader  # 数据加载
import warnings  # 警告处理
from typing import Dict, List, Tuple  # 类型提示

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib
import sys

# 检查是否在IDE中运行，设置合适的后端
if 'pydevd' in sys.modules or 'debugpy' in sys.modules:
    matplotlib.use('Agg')  # 在调试模式下使用非交互后端
else:
    try:
        matplotlib.use('TkAgg')  # 尝试使用TkAgg后端
    except:
        matplotlib.use('Agg')  # 失败时使用非交互后端


class RobustMarketDataset(Dataset):
    """
    稳健的市场数据集
    处理时间序列数据的批量加载
    """

    def __init__(self, features, labels, sequence_length=30, prediction_horizon=1):
        # 初始化特征、标签、序列长度和预测范围
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        # 返回数据集长度，考虑序列长度和预测范围
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        # 获取单个数据样本
        end_idx = idx + self.sequence_length
        features_seq = self.features[idx:end_idx]  # 获取特征序列

        # 预测未来prediction_horizon步的标签
        label_idx = end_idx + self.prediction_horizon - 1
        if label_idx < len(self.labels):
            label = self.labels[label_idx]  # 获取对应标签
        else:
            label = 1  # 默认平稳

        # 返回特征序列和标签的张量
        return torch.FloatTensor(features_seq), torch.LongTensor([label])


class EfficientMarketDataProcessor:
    """
    高效市场数据处理器
    使用向量化计算提升性能
    """

    def __init__(self, symbol='^NDX', initial_period='2y'):
        # 初始化股票符号、数据周期、数据和缓存
        self.symbol = symbol
        self.initial_period = initial_period
        self.data = None
        self.technical_cache = {}  # 技术指标缓存
        self.volatility_windows = [20, 60]  # 波动率计算窗口

    def fetch_data(self, period="2y", cache_dir="./data_cache"):
        """获取数据（带缓存）"""
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"ndx_data_{period}.pkl")

        # 检查缓存文件是否存在且较新
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 7:
                print("从缓存加载数据...")
                try:
                    # 从缓存加载数据
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    self.data = cached_data
                    print(f"缓存数据加载完成，共{len(self.data)}条记录")
                    return self.data
                except Exception as e:
                    print(f"缓存加载失败: {e}，重新下载数据...")

        print("正在下载纳斯达克指数数据...")
        try:
            # 从yfinance下载数据
            data = yf.download("^NDX", period=period, interval='1h', proxy="http://127.0.0.1:7890")
            data = data.dropna()  # 删除空值

            if data.empty:
                raise ValueError("下载的数据为空")

            print(f"数据下载完成，共{len(data)}条记录")

            # 修复：处理MultiIndex列名问题
            if isinstance(data.columns, pd.MultiIndex):
                print("检测到MultiIndex列，进行扁平化处理...")
                # 方法1：直接使用第一层列名
                data.columns = data.columns.get_level_values(0)

            print(f"处理后的列名: {data.columns.tolist()}")

            try:
                # 保存数据到缓存
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"数据已保存到缓存: {cache_file}")
            except Exception as e:
                print(f"缓存保存失败: {e}")

            self.data = data
            return data

        except Exception as e:
            print(f"数据下载失败: {e}")
            # 如果下载失败，尝试使用缓存
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    self.data = cached_data
                    return self.data
                except Exception as cache_error:
                    print(f"缓存数据也失败: {cache_error}")
            raise e

    def enhanced_calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        增强版技术指标计算（替换原函数）
        """
        print("增强版向量化计算技术指标...")

        # 提取收盘价数据
        if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
            close = df['Close']
        else:
            # 处理多级列名
            close = df[('Close', '^NDX')] if ('Close', '^NDX') in df.columns else df.iloc[:, 0]

        # 确保close是Series
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        # 创建增强特征DataFrame
        df_enhanced = pd.DataFrame(index=df.index)
        df_enhanced['returns'] = close.pct_change()  # 收益率
        df_enhanced['log_returns'] = np.log(close / close.shift(1))  # 对数收益率

        # 多周期移动平均
        for window in [5, 10, 20, 50]:
            df_enhanced[f'SMA_{window}'] = close.rolling(window=window).mean()  # 简单移动平均
            df_enhanced[f'EMA_{window}'] = close.ewm(span=window).mean()  # 指数移动平均

            # 修复：确保price_ratio计算正确
            sma_col = df_enhanced[f'SMA_{window}']
            if isinstance(sma_col, pd.DataFrame):
                sma_col = sma_col.squeeze()
            df_enhanced[f'price_ratio_ma_{window}'] = close / sma_col  # 价格与移动平均比率

        # 动量指标
        df_enhanced['momentum_5'] = close.pct_change(5)  # 5期动量
        df_enhanced['momentum_10'] = close.pct_change(10)  # 10期动量
        df_enhanced['ROC_10'] = (close / close.shift(10) - 1) * 100  # 价格变化率

        # 波动率指标
        for window in [5, 10, 20]:
            df_enhanced[f'volatility_{window}'] = close.pct_change().rolling(window).std()  # 滚动标准差

        # RSI 多周期
        def compute_rsi(series, window=14):
            """计算相对强弱指数"""
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # 平均涨幅
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # 平均跌幅
            rs = gain / loss  # 相对强度
            rsi = 100 - (100 / (1 + rs))  # RSI计算
            return rsi

        for period in [6, 14, 21]:
            df_enhanced[f'RSI_{period}'] = compute_rsi(close, period)  # 计算不同周期的RSI

        # 布林带
        for window in [10, 20]:
            sma = close.rolling(window=window).mean()  # 中轨
            std = close.rolling(window=window).std()  # 标准差
            df_enhanced[f'BB_upper_{window}'] = sma + (std * 2)  # 上轨
            df_enhanced[f'BB_lower_{window}'] = sma - (std * 2)  # 下轨

            # 修复：确保布林带宽度计算正确
            bb_width = (df_enhanced[f'BB_upper_{window}'] - df_enhanced[f'BB_lower_{window}']) / sma
            if isinstance(bb_width, pd.DataFrame):
                bb_width = bb_width.squeeze()
            df_enhanced[f'BB_width_{window}'] = bb_width  # 布林带宽度

            # 修复：确保布林带位置计算正确
            bb_position = (close - df_enhanced[f'BB_lower_{window}']) / (
                        df_enhanced[f'BB_upper_{window}'] - df_enhanced[f'BB_lower_{window}'])
            if isinstance(bb_position, pd.DataFrame):
                bb_position = bb_position.squeeze()
            df_enhanced[f'BB_position_{window}'] = bb_position  # 价格在布林带中的位置

        # 成交量相关指标（如果有成交量数据）
        if 'Volume' in df.columns or ('Volume', '^NDX') in df.columns:
            volume = df['Volume'] if 'Volume' in df.columns else df[('Volume', '^NDX')]
            if isinstance(volume, pd.DataFrame):
                volume = volume.squeeze()

            df_enhanced['volume_sma_10'] = volume.rolling(10).mean()  # 成交量移动平均
            df_enhanced['volume_ratio'] = volume / df_enhanced['volume_sma_10']  # 成交量比率
            df_enhanced['OBV'] = (np.sign(close.diff()) * volume).cumsum()  # 能量潮指标

        # 价格位置特征
        df_enhanced['high_20'] = close.rolling(20).max()  # 20期最高价
        df_enhanced['low_20'] = close.rolling(20).min()  # 20期最低价

        # 修复：确保价格位置计算正确
        price_position = (close - df_enhanced['low_20']) / (df_enhanced['high_20'] - df_enhanced['low_20'])
        if isinstance(price_position, pd.DataFrame):
            price_position = price_position.squeeze()
        df_enhanced['price_position'] = price_position  # 价格在区间内的位置

        # 删除NaN值
        df_enhanced = df_enhanced.dropna()

        return df_enhanced

    def calculate_volatility_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算波动率特征
        """
        print("向量化计算波动率特征...")
        volatility_features = pd.DataFrame(index=df.index)

        # 基础指标列表
        base_indicators = ['RSI_14', 'RSI_21', 'BB_width_20', 'Volatility_20',
                           'Volatility_60', 'Momentum_5', 'Momentum_10', 'BB_position']

        # 为每个指标和窗口计算波动率
        for window in self.volatility_windows:
            for indicator in base_indicators:
                if indicator in df.columns:
                    # 确保指标是Series
                    indicator_series = df[indicator]
                    if isinstance(indicator_series, pd.DataFrame):
                        indicator_series = indicator_series.squeeze()

                    vol_col = f'{indicator}_vol_{window}'
                    # 计算滚动标准差作为波动率
                    volatility_features[vol_col] = indicator_series.rolling(window=window, min_periods=window).std()

        return volatility_features

    def create_enhanced_labels(self, features_df: pd.DataFrame, price_data, lookforward_days=3):
        """
        增强版标签生成（替换原函数）
        """
        print("创建增强版智能标签...")

        # 提取收盘价数据
        if isinstance(price_data, pd.DataFrame):
            if 'Close' in price_data.columns:
                close_prices = price_data['Close'].values
            else:
                close_prices = price_data[('Close', '^NDX')].values
        else:
            close_prices = price_data

        # 确保 close_prices 是一维数组
        if hasattr(close_prices, 'shape') and len(close_prices.shape) > 1:
            close_prices = close_prices.flatten()

        # 修复：确保 close_prices 是 numpy 数组
        close_prices = np.array(close_prices)

        # 计算未来收益率 - 修复版本
        future_returns = []
        for i in range(len(close_prices) - lookforward_days):
            try:
                # 确保我们获取的是单个值而不是序列
                current_price = close_prices[i]
                future_price = close_prices[i + lookforward_days]

                # 确保价格是数值类型
                if hasattr(current_price, '__len__') and not isinstance(current_price, (int, float)):
                    current_price = current_price[0] if len(current_price) > 0 else np.nan
                if hasattr(future_price, '__len__') and not isinstance(future_price, (int, float)):
                    future_price = future_price[0] if len(future_price) > 0 else np.nan

                # 计算收益率
                if not np.isnan(current_price) and current_price != 0:
                    future_return = (future_price - current_price) / current_price
                else:
                    future_return = np.nan

                future_returns.append(future_return)
            except (IndexError, TypeError, ValueError) as e:
                future_returns.append(np.nan)
                continue

        # 创建标签：0=下跌, 1=平稳, 2=上涨
        labels = []
        for ret in future_returns:
            if np.isnan(ret):
                labels.append(1)  # 平稳作为默认值
            elif ret > 0.01:
                labels.append(2)  # 上涨
            elif ret < -0.005:
                labels.append(0)  # 下跌
            else:
                labels.append(1)  # 平稳

        labels = np.array(labels)

        # 确保长度匹配
        min_length = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_length]
        labels = labels[:min_length]

        print(f"增强标签分布 - 下跌: {np.sum(labels == 0)}, 平稳: {np.sum(labels == 1)}, 上涨: {np.sum(labels == 2)}")

        return features_df, labels


class RealisticBacktester:
    """
    现实回测系统
    考虑交易成本、滑点等现实因素
    """

    def __init__(self, initial_capital: float = 100000):
        # 初始化回测参数
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 0.1%交易成本
        self.slippage = 0.0005  # 0.05%滑点
        self.positions = []  # 持仓记录
        self.trade_log = []  # 交易日志
        self.equity_curve = []  # 权益曲线
        self.trade_points = []  # 交易点记录

    def calculate_position_size(self, current_capital: float, confidence: float,
                                volatility: float) -> float:
        """
        基于凯利公式和波动率计算仓位大小
        """
        # 简化版凯利公式
        if confidence > 0.6 and volatility < 0.05:  # 高信心低波动
            kelly_fraction = min(0.2, confidence - 0.5)
        elif confidence > 0.55:  # 中等信心
            kelly_fraction = 0.1
        else:  # 低信心
            kelly_fraction = 0.05

        # 波动率调整
        vol_adjustment = max(0.1, 1 - volatility * 10)
        position_size = current_capital * kelly_fraction * vol_adjustment

        return position_size

    def execute_trade(self, signal: int, price, current_capital: float,
                      confidence: float, volatility: float, timestamp: datetime) -> Dict:
        """
        执行交易（考虑现实因素）- 修复价格数据类型问题
        """
        # 修复：确保price是单个浮点数
        if hasattr(price, '__len__') and not isinstance(price, str):
            # 如果是序列，取第一个元素
            if len(price) > 0:
                price_value = float(price[0])
            else:
                price_value = 100.0  # 默认价格
        else:
            price_value = float(price)

        # 应用滑点
        execution_price = price_value * (1 + self.slippage) if signal == 2 else price_value * (1 - self.slippage)

        # 计算仓位大小
        position_size = self.calculate_position_size(current_capital, confidence, volatility)

        # 计算交易成本
        trade_cost = position_size * self.transaction_cost

        # 创建交易记录
        trade_record = {
            'timestamp': timestamp,
            'signal': signal,
            'price': execution_price,
            'position_size': position_size,
            'cost': trade_cost,
            'shares': position_size / execution_price if signal == 2 else 0  # 买入时计算股数
        }

        self.trade_log.append(trade_record)
        return trade_record

    def run_backtest(self, predictions: List[int], confidence_scores: List[float],
                     prices: List[float], volatilities: List[float],
                     timestamps: List[datetime]) -> Dict[str, float]:
        """
        运行现实回测 - 修复peak_capital更新问题
        """
        print("运行基于回撤止损 + 6%止盈的回测...")
        print(f"预测列表长度: {len(predictions)}")
        print(f"价格列表长度: {len(prices)}")

        # 初始化回测参数
        capital = self.initial_capital  # 初始资金
        position = 0  # 当前持仓股数
        max_drawdown = 0  # 最大回撤
        peak_capital = self.initial_capital  # 资金峰值
        trades = 0  # 交易次数统计

        # 风险控制参数
        drawdown_threshold = 0.006  # 回撤止损阈值
        take_profit_threshold = 0.044  # 止盈阈值
        entry_price = 0  # 入场价格
        position_direction = 0  # 持仓方向: 0=无持仓, 1=多头

        # 权益曲线记录
        self.equity_curve = [capital]
        self.trade_points = []  # 重置交易点记录
        self.drawdown_stop_trades = []  # 回撤止损交易记录
        self.take_profit_trades = []  # 止盈交易记录

        # 主回测循环：遍历每个预测时点
        for i in range(1, len(predictions)):
            # ==================== 数据预处理和验证 ====================
            current_price = prices[i]
            if hasattr(current_price, '__len__') and not isinstance(current_price, str):
                if len(current_price) > 0:
                    current_price = float(current_price[0])
                else:
                    current_price = 100.0
            else:
                current_price = float(current_price)

            current_timestamp = timestamps[i]

            # 获取置信度
            if i < len(confidence_scores):
                if hasattr(confidence_scores[i], '__len__') and len(confidence_scores[i]) > predictions[i]:
                    confidence = confidence_scores[i][predictions[i]]
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            # ==================== 关键修复：计算当前总资产价值 ====================
            # 无论是否有持仓，都计算当前总资产
            current_value = capital + (position * current_price if position > 0 else 0)

            # 关键修复：peak_capital应该在每次资产创新高时更新，无论是否持仓
            if current_value > peak_capital:
                peak_capital = current_value

            # ==================== 风险指标计算 ====================
            # 计算当前回撤（基于当前peak_capital）
            drawdown = (peak_capital - current_value) / peak_capital if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # ==================== 止盈检查 ====================
            take_profit_triggered = False
            if position > 0 and entry_price > 0:
                current_return = (current_price - entry_price) / entry_price
                if current_return >= take_profit_threshold:
                    take_profit_triggered = True
                    print(f"止盈触发: 当前收益率 {current_return:.2%} >= 止盈阈值 {take_profit_threshold:.2%}")

            # ==================== 回撤止损检查 ====================
            drawdown_stop_triggered = False
            if position > 0 and drawdown >= drawdown_threshold:
                drawdown_stop_triggered = True
                print(f"回撤止损触发: 当前回撤 {drawdown:.2%} > 阈值 {drawdown_threshold:.2%}")

            # ==================== 交易信号处理 ====================
            signal = predictions[i]
            prev_signal = predictions[i - 1] if i > 0 else 1

            # 执行交易的标志
            should_trade = (signal != prev_signal) or take_profit_triggered or drawdown_stop_triggered

            if should_trade and position > 0:
                # 平仓逻辑
                close_trade = self.execute_trade(0, current_price, capital,
                                                 confidence, 0.02, current_timestamp)

                # 计算盈亏
                profit_loss = (current_price - entry_price) * position - close_trade['cost']
                profit_loss_pct = (profit_loss / (entry_price * position)) * 100 if entry_price > 0 else 0

                # 更新资金
                capital += position * current_price - close_trade['cost']

                # 关键修复：平仓后立即更新当前价值和peak_capital
                current_value = capital  # 平仓后总资产就是现金
                peak_capital = current_value

                # 记录交易类型
                trade_type = 'normal'
                if take_profit_triggered:
                    trade_type = 'take_profit'
                    self.take_profit_trades.append({
                        'timestamp': current_timestamp,
                        'price': current_price,
                        'entry_price': entry_price,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct,
                        'return_at_exit': (current_price - entry_price) / entry_price
                    })
                elif drawdown_stop_triggered:
                    trade_type = 'drawdown_stop'
                    self.drawdown_stop_trades.append({
                        'timestamp': current_timestamp,
                        'price': current_price,
                        'entry_price': entry_price,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct,
                        'drawdown_at_stop': drawdown
                    })

                # 记录卖出点
                self.trade_points.append({
                    'timestamp': current_timestamp,
                    'price': current_price,
                    'type': 'sell',
                    'equity': current_value,
                    'trade_type': trade_type,
                    'profit_loss': profit_loss,
                    'drawdown': drawdown,
                    'return_pct': profit_loss_pct
                })

                position = 0
                trades += 1
                position_direction = 0

            # 开仓逻辑
            if signal == 2 and position == 0 and capital > 0:
                buy_trade = self.execute_trade(2, current_price, capital,
                                               confidence, 0.02, current_timestamp)
                if buy_trade['shares'] > 0:
                    position = buy_trade['shares']
                    entry_price = current_price
                    position_direction = 1

                    # 更新资金
                    capital -= buy_trade['position_size'] + buy_trade['cost']
                    trades += 1

                    # 关键修复：开仓后重新计算当前价值和peak_capital
                    current_total_value = capital + (position * current_price)
                    peak_capital = current_total_value

                    print(
                        f"开仓: 价格 {current_price:.2f}, 止盈阈值: {take_profit_threshold:.2%}, 回撤止损阈值: {drawdown_threshold:.2%}")

                    # 记录买入点
                    self.trade_points.append({
                        'timestamp': current_timestamp,
                        'price': current_price,
                        'type': 'buy',
                        'equity': current_total_value,
                        'take_profit_threshold': take_profit_threshold,
                        'drawdown_threshold': drawdown_threshold
                    })

            # 记录当前时刻的权益曲线点
            self.equity_curve.append(current_value)

        # ==================== 回测结束处理 ====================
        # 最终平仓
        if position > 0:
            final_price = float(prices[-1])
            close_trade = self.execute_trade(0, final_price, capital, 0.5, 0.02, timestamps[-1])
            capital += position * final_price - close_trade['cost']

            profit_loss = (final_price - entry_price) * position - close_trade['cost']
            profit_loss_pct = (profit_loss / (entry_price * position)) * 100 if entry_price > 0 else 0

            # 关键修复：最终平仓后更新peak_capital
            current_value = capital

            self.trade_points.append({
                'timestamp': timestamps[-1],
                'price': final_price,
                'type': 'sell',
                'equity': capital,
                'trade_type': 'close_out',
                'profit_loss': profit_loss,
                'return_pct': profit_loss_pct
            })

        # ==================== 绩效指标计算 ====================
        # 重新计算最终的最大回撤，确保准确性
        final_max_drawdown = 0
        final_peak = self.equity_curve[0]

        for value in self.equity_curve:
            if value > final_peak:
                final_peak = value
            current_drawdown = (final_peak - value) / final_peak if final_peak > 0 else 0
            final_max_drawdown = max(final_max_drawdown, current_drawdown)

        # 其他绩效指标计算保持不变...
        total_return = (capital - self.initial_capital) / self.initial_capital
        annual_return = total_return / (len(predictions) / 252) if len(predictions) > 0 else 0

        returns = []
        if len(self.equity_curve) > 1:
            returns = [(self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
                       for i in range(1, len(self.equity_curve))]

        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        # 胜率计算
        winning_trades = 0
        total_trade_pairs = 0

        for i in range(0, len(self.trade_points) - 1, 2):
            if i + 1 < len(self.trade_points):
                buy_point = self.trade_points[i]
                sell_point = self.trade_points[i + 1]
                if buy_point['type'] == 'buy' and sell_point['type'] == 'sell':
                    total_trade_pairs += 1
                    if sell_point.get('profit_loss', 0) > 0:
                        winning_trades += 1

        win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0

        # 止盈和止损统计
        take_profit_count = len(self.take_profit_trades)
        take_profit_rate = take_profit_count / total_trade_pairs if total_trade_pairs > 0 else 0

        drawdown_stop_count = len(self.drawdown_stop_trades)
        drawdown_stop_rate = drawdown_stop_count / total_trade_pairs if total_trade_pairs > 0 else 0

        # ==================== 结果汇总 ====================
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': final_max_drawdown,  # 使用重新计算的最大回撤
            'sharpe_ratio': sharpe_ratio,
            'total_trades': trades,
            'win_rate': win_rate,
            'take_profit_count': take_profit_count,
            'take_profit_rate': take_profit_rate,
            'take_profit_threshold': take_profit_threshold,
            'drawdown_stop_count': drawdown_stop_count,
            'drawdown_stop_rate': drawdown_stop_rate,
            'drawdown_threshold': drawdown_threshold,
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }

        # 打印回测结果摘要
        self.print_backtest_results(results)
        return results

    def print_backtest_results(self, results: Dict[str, float]):
        """打印基于回撤止损+止盈的回测结果"""
        print("\n" + "=" * 70)
        print("基于回撤止损 + 6%止盈的回测结果")
        print("=" * 70)
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annual_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"止盈次数: {results['take_profit_count']}")
        print(f"止盈率: {results['take_profit_rate']:.2%}")
        print(f"止盈阈值: {results['take_profit_threshold']:.2%}")
        print(f"回撤止损次数: {results['drawdown_stop_count']}")
        print(f"回撤止损率: {results['drawdown_stop_rate']:.2%}")
        print(f"回撤止损阈值: {results['drawdown_threshold']:.2%}")
        print("=" * 70)

        # 打印止盈交易的详细信息
        if self.take_profit_trades:
            avg_take_profit = np.mean([t['profit_loss_pct'] for t in self.take_profit_trades])
            avg_return_at_take_profit = np.mean([t['return_at_exit'] for t in self.take_profit_trades])
            print(f"止盈交易平均收益率: {avg_take_profit:.2f}%")
            print(f"触发止盈时的平均收益率: {avg_return_at_take_profit:.2%}")

        # 打印回撤止损交易的详细信息
        if self.drawdown_stop_trades:
            avg_drawdown_stop = np.mean([t['profit_loss_pct'] for t in self.drawdown_stop_trades])
            avg_drawdown_at_stop = np.mean([t['drawdown_at_stop'] for t in self.drawdown_stop_trades])
            print(f"回撤止损平均亏损: {avg_drawdown_stop:.2f}%")
            print(f"触发止损时的平均回撤: {avg_drawdown_at_stop:.2%}")

    def plot_backtest_results(self, prices: List[float], timestamps: List[datetime],
                              predictions: List[int] = None, save_path: str = None):
        """
        绘制回测结果图表 - 增强版，显示止盈点

        参数:
            prices: 价格序列
            timestamps: 时间戳序列
            predictions: 预测信号序列 (可选)
            save_path: 图片保存路径 (可选)
        """
        if not self.equity_curve:
            print("警告: 没有回测数据可绘制")
            return

        print("生成回测结果图表...")

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12),
                                       gridspec_kw={'height_ratios': [2, 1]})

        # 确保时间戳和价格长度匹配
        min_len = min(len(timestamps), len(prices), len(self.equity_curve))
        timestamps = timestamps[:min_len]
        prices = prices[:min_len]
        equity_curve = self.equity_curve[:min_len]

        # 第一个子图：价格走势和买卖点
        ax1.plot(timestamps, prices, label='价格', color='blue', linewidth=1, alpha=0.7)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('价格走势与交易信号 (绿色=买入, 红色=卖出, 金色=止盈)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 绘制买卖点 - 区分止盈点
        buy_points = [point for point in self.trade_points if point['type'] == 'buy']
        sell_points_normal = [point for point in self.trade_points if
                              point['type'] == 'sell' and point.get('trade_type') != 'take_profit']
        sell_points_take_profit = [point for point in self.trade_points if
                                   point['type'] == 'sell' and point.get('trade_type') == 'take_profit']

        if buy_points:
            buy_times = [point['timestamp'] for point in buy_points]
            buy_prices = [point['price'] for point in buy_points]
            ax1.scatter(buy_times, buy_prices, color='green', marker='^',
                        s=100, label='买入', zorder=5)

        if sell_points_normal:
            sell_times = [point['timestamp'] for point in sell_points_normal]
            sell_prices = [point['price'] for point in sell_points_normal]
            ax1.scatter(sell_times, sell_prices, color='red', marker='v',
                        s=100, label='普通卖出', zorder=5)

        if sell_points_take_profit:
            take_profit_times = [point['timestamp'] for point in sell_points_take_profit]
            take_profit_prices = [point['price'] for point in sell_points_take_profit]
            ax1.scatter(take_profit_times, take_profit_prices, color='gold', marker='v',
                        s=120, label='止盈卖出', zorder=6, edgecolors='orange', linewidth=2)

        # 如果有预测信号，在背景中显示
        if predictions is not None and len(predictions) == len(prices):
            # 创建信号背景色
            for i in range(1, len(predictions)):
                if predictions[i] == 2:  # 买入信号
                    ax1.axvspan(timestamps[i - 1], timestamps[i],
                                alpha=0.1, color='green', label='买入信号' if i == 1 else "")
                elif predictions[i] == 0:  # 卖出信号
                    ax1.axvspan(timestamps[i - 1], timestamps[i],
                                alpha=0.1, color='red', label='卖出信号' if i == 1 else "")

        ax1.legend(loc='upper left')

        # 格式化x轴
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 第二个子图：权益曲线
        ax2.plot(timestamps, equity_curve, label='投资组合价值', color='purple', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--',
                    label=f'初始资金 (${self.initial_capital:,.0f})', alpha=0.7)
        ax2.set_ylabel('投资组合价值 ($)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_title('投资组合价值曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # 格式化x轴
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        sell_points = [p for p in self.trade_points if p['type'] == 'sell']
        take_profit_rate = len(self.take_profit_trades) / len(sell_points) if sell_points else 0
        # 添加统计信息文本框 - 包含止盈信息
        stats_text = f"""
    回测统计信息:
    初始资金: ${self.initial_capital:,.0f}
    最终资金: ${equity_curve[-1]:,.0f}
    总收益率: {(equity_curve[-1] / self.initial_capital - 1):.2%}
    总交易次数: {len(self.trade_points)}
    止盈次数: {len(self.take_profit_trades)}
    止盈率: {take_profit_rate:.2%}
    最大回撤: {max(0, (max(equity_curve) - min(equity_curve)) / max(equity_curve)):.2%}
        """

        # 在权益曲线图上添加统计信息
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

        return fig

    def plot_detailed_analysis(self, prices: List[float], timestamps: List[datetime],
                               predictions: List[int], save_path: str = None):
        """
        绘制详细分析图表（包含更多技术指标和止盈分析）
        """
        print("生成详细分析图表...")

        # 创建更详细的图表
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 2, 2, 2, 2])

        ax1 = fig.add_subplot(gs[0])  # 价格和交易信号
        ax2 = fig.add_subplot(gs[1])  # 权益曲线
        ax3 = fig.add_subplot(gs[2])  # 收益率分布
        ax4 = fig.add_subplot(gs[3])  # 交易统计
        ax5 = fig.add_subplot(gs[4])  # 止盈分析

        # 子图1：价格和交易信号
        ax1.plot(timestamps, prices, label='价格', color='navy', linewidth=1.5)

        # 标记买卖点 - 区分止盈点
        buy_dates = [point['timestamp'] for point in self.trade_points if point['type'] == 'buy']
        buy_prices = [point['price'] for point in self.trade_points if point['type'] == 'buy']

        sell_dates_normal = [point['timestamp'] for point in self.trade_points if
                             point['type'] == 'sell' and point.get('trade_type') != 'take_profit']
        sell_prices_normal = [point['price'] for point in self.trade_points if
                              point['type'] == 'sell' and point.get('trade_type') != 'take_profit']

        sell_dates_take_profit = [point['timestamp'] for point in self.trade_points if
                                  point['type'] == 'sell' and point.get('trade_type') == 'take_profit']
        sell_prices_take_profit = [point['price'] for point in self.trade_points if
                                   point['type'] == 'sell' and point.get('trade_type') == 'take_profit']

        ax1.scatter(buy_dates, buy_prices, color='lime', marker='^', s=120,
                    label=f'买入 ({len(buy_dates)})', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax1.scatter(sell_dates_normal, sell_prices_normal, color='red', marker='v', s=120,
                    label=f'普通卖出 ({len(sell_dates_normal)})', zorder=5, edgecolors='darkred', linewidth=2)
        ax1.scatter(sell_dates_take_profit, sell_prices_take_profit, color='gold', marker='v', s=150,
                    label=f'止盈卖出 ({len(sell_dates_take_profit)})', zorder=6, edgecolors='orange', linewidth=2)

        ax1.set_title('价格走势与交易点位 (包含6%止盈策略)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：权益曲线
        ax2.plot(timestamps[:len(self.equity_curve)], self.equity_curve,
                 label='投资组合价值', color='purple', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--',
                    label='初始资金', alpha=0.7)

        # 计算并绘制最大回撤
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        max_dd_idx = np.argmax(drawdown)

        ax2.fill_between(timestamps[:len(self.equity_curve)], self.equity_curve, peak,
                         alpha=0.3, color='red', label='回撤区域')

        ax2.set_title('投资组合价值与回撤分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('价值 ($)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3：收益率分布
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        ax3.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--',
                    label=f'平均收益率: {returns.mean():.4%}')
        ax3.set_title('日收益率分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('收益率')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4：交易统计
        if self.trade_log:
            trade_profits = []
            trade_types = []  # 记录交易类型
            for i in range(0, len(self.trade_points) - 1, 2):
                if i + 1 < len(self.trade_points):
                    buy_point = self.trade_points[i]
                    sell_point = self.trade_points[i + 1]
                    if buy_point['type'] == 'buy' and sell_point['type'] == 'sell':
                        profit = (sell_point['price'] - buy_point['price']) / buy_point['price']
                        trade_profits.append(profit)
                        trade_types.append(sell_point.get('trade_type', 'normal'))

            if trade_profits:
                # 为不同类型的交易设置不同颜色
                colors = []
                for trade_type in trade_types:
                    if trade_type == 'take_profit':
                        colors.append('gold')
                    elif trade_type == 'drawdown_stop':
                        colors.append('red')
                    else:
                        colors.append('blue' if trade_profits[len(colors)] > 0 else 'lightcoral')

                bars = ax4.bar(range(len(trade_profits)), trade_profits, color=colors, alpha=0.7)
                ax4.axhline(y=0, color='black', linewidth=0.8)
                ax4.set_title('单笔交易收益率 (金色=止盈, 红色=止损)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('交易序号')
                ax4.set_ylabel('收益率')
                ax4.grid(True, alpha=0.3)

                # 添加平均线
                avg_profit = np.mean(trade_profits)
                ax4.axhline(y=avg_profit, color='blue', linestyle='--',
                            label=f'平均收益率: {avg_profit:.2%}')
                ax4.legend()

        # 子图5：止盈分析
        if self.take_profit_trades:
            take_profit_returns = [t['return_at_exit'] for t in self.take_profit_trades]
            take_profit_dates = [t['timestamp'] for t in self.take_profit_trades]

            ax5.bar(take_profit_dates, take_profit_returns, color='gold', alpha=0.7, label='止盈收益率')
            ax5.axhline(y=0.06, color='red', linestyle='--', label='6%止盈目标')
            ax5.set_title('止盈交易收益率分析', fontsize=14, fontweight='bold')
            ax5.set_xlabel('时间')
            ax5.set_ylabel('收益率')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 添加止盈统计
            avg_take_profit = np.mean(take_profit_returns)
            ax5.text(0.02, 0.98, f'平均止盈收益率: {avg_take_profit:.2%}',
                     transform=ax5.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            detailed_path = save_path.replace('.png', '_detailed.png')
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            print(f"详细图表已保存至: {detailed_path}")
        else:
            plt.show()

        return fig


class EnhancedLSTMModel(nn.Module):
    """
    增强版LSTM模型（替换原SimpleLSTMPredictor）
    包含双向LSTM、注意力机制、批归一化等
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=3, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 批归一化层 - 修复：移除或修改批归一化
        # self.batch_norm = nn.BatchNorm1d(input_size)

        # 双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(64),  # 在分类器中保留批归一化
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # 修复：移除输入层的批归一化，因为输入是3D (batch, seq, features)
        # 直接使用LSTM处理

        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(x)

        # 注意力机制
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 分类
        out = self.classifier(context_vector)
        return out


class ImprovedTransformerPredictor(nn.Module):
    """
    改进的Transformer预测器，支持多种池化策略
    """

    def __init__(self, input_dim: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, dim_feedforward: int = 64,
                 pooling: str = 'last'):
        super(ImprovedTransformerPredictor, self).__init__()

        self.pooling = pooling  # 'last', 'mean', 'max'

        self.input_projection = nn.Linear(input_dim, 32)  # 输入投影层

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 3)  # 3个类别
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier初始化
                nn.init.constant_(layer.bias, 0.0)  # 偏置初始化为0

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)

        Returns:
            分类logits
        """
        # 处理输入维度
        if x.dim() == 2:
            # 二维输入: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D input")

        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, 32)

        # Transformer编码
        x = self.transformer(x)  # (batch_size, seq_len, 32)

        # 池化策略
        if self.pooling == 'last':
            x = x[:, -1, :]  # 最后一个时间步
        elif self.pooling == 'mean':
            x = x.mean(dim=1)  # 平均池化
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]  # 最大池化
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # 分类
        output = self.classifier(x)  # (batch_size, 3)

        return output


class MultiModelPredictor:
    """
    多模型预测器
    支持Transformer和LSTM模型的选择和比较
    """

    def __init__(self, input_dim: int, model_type: str = 'lstm', **model_kwargs):
        """
        初始化多模型预测器

        Args:
            input_dim: 输入特征维度
            model_type: 模型类型 ('lstm', 'transformer', 'ensemble')
            **model_kwargs: 模型参数
        """
        self.input_dim = input_dim
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # 创建模型
        if model_type == 'lstm':
            self.model = EnhancedLSTMModel(input_dim, **model_kwargs)  # 使用增强版LSTM
        elif model_type == 'transformer':
            self.model = ImprovedTransformerPredictor(input_dim, **model_kwargs)
        elif model_type == 'ensemble':
            self.models = {
                'lstm': EnhancedLSTMModel(input_dim, **model_kwargs.get('lstm', {})),
                'transformer': ImprovedTransformerPredictor(input_dim, **model_kwargs.get('transformer', {}))
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        self.scaler = StandardScaler()  # 数据标准化器
        self.is_fitted = False  # 模型是否已训练

    def create_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4):
        """创建优化器"""
        if self.model_type == 'ensemble':
            params = []
            for model in self.models.values():
                params.extend(list(model.parameters()))
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_enhanced_lstm(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
        """增强版LSTM训练（替换原训练函数）"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择

        if self.model_type == 'ensemble':
            model = self.models['lstm']
        else:
            model = self.model

        model = model.to(device)

        # 带类别权重的损失函数
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([2, 1, 1.5]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience = 20
        counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 验证
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y.squeeze()).sum().item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Acc = {accuracy:.2f}%')

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_lstm_model.pth')  # 保存最佳模型
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # 加载最佳模型
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            batch_size: int = 32, validation_split: float = 0.2):
        """训练模型"""
        if self.optimizer is None:
            self.create_optimizer()

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 创建数据集
        dataset = RobustMarketDataset(X_scaled, y)

        # 分割训练验证集
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 使用增强训练方法
        if self.model_type == 'lstm' or (self.model_type == 'ensemble' and 'lstm' in self.models):
            self.train_enhanced_lstm(train_loader, val_loader, epochs)
        else:
            # 原有训练逻辑（用于transformer）
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                # 训练阶段
                if self.model_type == 'ensemble':
                    for model in self.models.values():
                        model.train()
                else:
                    self.model.train()

                train_loss = 0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()

                    if self.model_type == 'ensemble':
                        # 集成学习：平均所有模型的输出
                        outputs = []
                        for model in self.models.values():
                            output = model(batch_X)
                            outputs.append(output)
                        outputs = torch.stack(outputs).mean(dim=0)
                    else:
                        outputs = self.model(batch_X)

                    loss = self.criterion(outputs, batch_y.squeeze())
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                # 验证阶段
                if self.model_type == 'ensemble':
                    for model in self.models.values():
                        model.eval()
                else:
                    self.model.eval()

                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        if self.model_type == 'ensemble':
                            outputs = []
                            for model in self.models.values():
                                output = model(batch_X)
                                outputs.append(output)
                            outputs = torch.stack(outputs).mean(dim=0)
                        else:
                            outputs = self.model(batch_X)

                        loss = self.criterion(outputs, batch_y.squeeze())
                        val_loss += loss.item()

                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, '
                          f'Val Loss = {val_losses[-1]:.4f}')

        self.is_fitted = True  # 标记模型已训练

    def predict(self, X: np.ndarray, return_probabilities: bool = False):
        """预测 - 修复版本"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)  # 标准化输入

        # 修复：正确处理输入维度
        if len(X_scaled.shape) == 1:
            # 单个样本
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        elif len(X_scaled.shape) == 2:
            # 多个样本，每个样本是特征向量
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)  # (n_samples, 1, input_dim)
        else:
            # 已经是3D
            X_tensor = torch.FloatTensor(X_scaled)

        if self.model_type == 'ensemble':
            for model in self.models.values():
                model.eval()
        else:
            self.model.eval()

        with torch.no_grad():
            if self.model_type == 'ensemble':
                outputs = []
                for model in self.models.values():
                    output = model(X_tensor)
                    outputs.append(output)
                outputs = torch.stack(outputs).mean(dim=0)  # 集成预测
            else:
                outputs = self.model(X_tensor)

            # 修复：确保输出维度正确
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            probabilities = torch.softmax(outputs, dim=1).numpy()  # 计算概率
            predictions = np.argmax(probabilities, axis=1)  # 获取预测类别

        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions

    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       feature_names: List[str] = None):
        """比较不同模型的性能"""
        if self.model_type != 'ensemble':
            print("只有集成模型才能进行比较")
            return

        results = {}

        for name, model in self.models.items():
            model.eval()
            X_scaled = self.scaler.transform(X_test)
            X_tensor = torch.FloatTensor(X_scaled)

            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()
                accuracy = np.mean(predictions == y_test)

            results[name] = {
                'accuracy': accuracy,
                'predictions': predictions
            }

            print(f"{name.upper()} 模型准确率: {accuracy:.4f}")

        # 集成模型性能
        ensemble_pred, _ = self.predict(X_test, return_probabilities=True)
        ensemble_accuracy = np.mean(ensemble_pred == y_test)
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred
        }
        print(f"集成模型准确率: {ensemble_accuracy:.4f}")

        return results


# 新增辅助函数
def create_sequences(features, targets, sequence_length=30):
    """创建序列数据"""
    X_sequences = []
    y_sequences = []

    for i in range(len(features) - sequence_length):
        X_sequences.append(features[i:(i + sequence_length)])  # 特征序列
        y_sequences.append(targets[i + sequence_length])  # 对应标签

    return np.array(X_sequences), np.array(y_sequences)


def enhance_training_data(X_train, y_train):
    """数据增强"""
    X_augmented = [X_train]
    y_augmented = [y_train]

    # 添加噪声增强
    noise_factor = 0.01
    X_noise = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_augmented.append(X_noise)
    y_augmented.append(y_train)

    # 时间扭曲
    if len(X_train) > 1:
        X_shift = np.roll(X_train, shift=1, axis=0)
        X_shift[0] = X_train[0]  # 保持第一个元素不变
        X_augmented.append(X_shift)
        y_augmented.append(y_train)

    return np.vstack(X_augmented), np.hstack(y_augmented)


def balance_dataset(X, y):
    """平衡数据集"""
    # 分离各类别
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    class_2 = X[y == 2]

    # 找到最大类别大小
    max_size = max(len(class_0), len(class_1), len(class_2))

    # 上采样少数类别
    class_0_upsampled = resample(class_0, replace=True, n_samples=max_size, random_state=42)
    class_1_upsampled = resample(class_1, replace=True, n_samples=max_size, random_state=42)
    class_2_upsampled = resample(class_2, replace=True, n_samples=max_size, random_state=42)

    # 合并
    X_balanced = np.vstack([class_0_upsampled, class_1_upsampled, class_2_upsampled])
    y_balanced = np.hstack([
        np.zeros(max_size),
        np.ones(max_size),
        np.full(max_size, 2)
    ])

    # 打乱数据
    shuffle_idx = np.random.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


class EnhancedIncrementalLearningModel:
    """
    增强的增量学习模型
    支持LSTM和Transformer模型
    """

    def __init__(self, input_dim: int, model_type: str = 'lstm', **model_kwargs):
        self.input_dim = input_dim
        self.model_type = model_type
        self.predictor = MultiModelPredictor(input_dim, model_type, **model_kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False

        # 增量学习相关
        self.update_frequency = 100  # 更新频率
        self.batch_size = 32  # 批次大小
        self.online_data_buffer = []  # 在线数据缓冲区
        self.update_count = 0  # 更新计数器

    def initial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """初始训练"""
        print(f"使用{self.model_type.upper()}模型进行初始训练...")
        self.predictor.fit(X, y, epochs=epochs)
        self.is_fitted = True

    def partial_fit(self, X_new: np.ndarray, y_new: np.ndarray):
        """增量学习"""
        if not self.is_fitted:
            self.initial_fit(X_new, y_new)
            return

        self.online_data_buffer.append((X_new, y_new))  # 添加到缓冲区
        self.update_count += 1

        if self.update_count >= self.update_frequency:
            self._update_model()  # 达到更新频率时更新模型

    def _update_model(self):
        """更新模型"""
        if not self.online_data_buffer:
            return

        print("执行增量模型更新...")
        all_X = np.vstack([data[0] for data in self.online_data_buffer])
        all_y = np.concatenate([data[1] for data in self.online_data_buffer])

        # 使用新数据微调模型
        self.predictor.fit(all_X, all_y, epochs=10)

        self.online_data_buffer = []  # 清空缓冲区
        self.update_count = 0  # 重置计数器

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测 - 修复版本"""
        predictions, probabilities = self.predictor.predict(X, return_probabilities=True)

        # 确保返回正确的形状
        if len(predictions.shape) == 0:
            predictions = np.array([predictions])
        if len(probabilities.shape) == 1:
            probabilities = probabilities.reshape(1, -1)

        return predictions, probabilities


class AdvancedTrendPredictor:
    """
    高级趋势预测器
    支持模型选择和比较
    """

    def __init__(self, model_type: str = 'lstm', **model_kwargs):
        self.data_processor = EfficientMarketDataProcessor()  # 数据处理器
        self.model_type = model_type  # 模型类型
        self.model_kwargs = model_kwargs  # 模型参数
        self.model = None  # 预测模型
        self.backtester = RealisticBacktester()  # 回测器
        self.is_trained = False  # 训练状态

    def enhanced_model_comparison(self, features: np.ndarray, labels: np.ndarray, sequence_length=30):
        """
        增强版模型比较（替换原函数）
        """
        print("运行增强模型比较...")

        # 修复：确保features是2D数组
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        # 数据预处理
        X_sequences, y_sequences = create_sequences(features, labels, sequence_length)

        # 数据平衡
        X_balanced, y_balanced = balance_dataset(X_sequences, y_sequences)

        # 数据增强
        X_enhanced, y_enhanced = enhance_training_data(X_balanced, y_balanced)

        # 分割数据集
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_enhanced, y_enhanced, test_size=0.3, random_state=42, stratify=y_enhanced
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # 修复：创建正确的数据集，确保输入维度正确
        # 直接使用TensorDataset，因为数据已经是序列格式
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 训练增强LSTM模型
        print("训练增强LSTM模型...")
        lstm_model = EnhancedLSTMModel(
            input_size=features.shape[1],
            hidden_size=128,
            num_layers=3,
            output_size=3,
            dropout_rate=0.3
        )

        # 创建优化器和训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lstm_model = lstm_model.to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([2, 1, 1.5]).to(device))
        optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience = 20
        counter = 0

        for epoch in range(100):
            lstm_model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 验证
            lstm_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Acc = {accuracy:.2f}%')

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(lstm_model.state_dict(), 'best_enhanced_lstm_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # 加载最佳模型
        lstm_model.load_state_dict(torch.load('best_enhanced_lstm_model.pth'))

        # 评估模型
        lstm_model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = lstm_model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        lstm_accuracy = test_correct / test_total
        print(f"增强LSTM模型准确率: {lstm_accuracy:.4f}")

        return lstm_model, lstm_accuracy

    def run_advanced_pipeline(self, compare_models: bool = False):
        """运行高级预测流程 - 修复版本"""
        print("开始高级趋势预测流程...")

        # 1. 获取和处理数据
        data = self.data_processor.fetch_data(period="2y")
        tech_data = self.data_processor.enhanced_calculate_technical_indicators(data)
        vol_features = self.data_processor.calculate_volatility_features_vectorized(tech_data)

        # 2. 合并特征和创建标签
        all_features = pd.concat([tech_data, vol_features], axis=1)

        # 修复：更新特征列选择以匹配新的技术指标名称
        feature_columns = [col for col in all_features.columns if any(x in col for x in
                                                                      ['SMA', 'EMA', 'RSI', 'BB', 'volatility',
                                                                       'momentum', '_vol_', 'returns', 'log_returns',
                                                                       'price_ratio', 'ROC', 'volume', 'OBV',
                                                                       'price_position'])]
        features_df = all_features[feature_columns].dropna()

        # 使用增强版标签生成
        features_df, labels = self.data_processor.create_enhanced_labels(features_df, data)

        print(f"特征形状: {features_df.shape}")
        print(f"标签分布: {np.unique(labels, return_counts=True)}")

        # 3. 模型比较（可选）
        if compare_models:
            self.enhanced_model_comparison(features_df.values, labels)

        # 4. 使用选定模型训练
        features_array = features_df.values

        if self.model_type == 'ensemble':
            self.model = EnhancedIncrementalLearningModel(
                features_array.shape[1],
                model_type='ensemble',
                lstm={'hidden_dim': 128, 'num_layers': 3, 'dropout_rate': 0.3},
                transformer={'num_heads': 4, 'num_layers': 2}
            )
        else:
            self.model = EnhancedIncrementalLearningModel(
                features_array.shape[1],
                model_type=self.model_type,
                **self.model_kwargs
            )

        # 5. 时间序列交叉验证训练
        self.train_with_incremental_learning(features_array, labels)

        # 6. 现实回测 - 传递正确的数据
        results = self.run_realistic_backtest(features_df, labels, data)

        if results:
            print("回测完成!")
        else:
            print("回测失败!")

        return features_df, labels, all_features

    def train_with_incremental_learning(self, features: np.ndarray, labels: np.ndarray):
        """使用增量学习训练模型 - 修复版本"""
        print(f"使用{self.model_type.upper()}模型进行增量学习训练...")

        tscv = TimeSeriesSplit(n_splits=5)  # 时间序列交叉验证
        fold_accuracies = []
        all_predictions = []
        all_true_labels = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            print(f"\n--- 训练折叠 {fold + 1}/5 ---")
            print(f"训练集大小: {len(train_idx)}, 测试集大小: {len(test_idx)}")

            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            try:
                # 初始训练
                if fold == 0:
                    self.model.initial_fit(X_train, y_train, epochs=50)
                else:
                    # 后续折叠使用增量学习
                    self.model.partial_fit(X_train, y_train)

                # 在测试集上预测
                predictions, confidence_scores = self.model.predict(X_test)

                # 处理不同的预测结果格式
                if isinstance(predictions, (list, np.ndarray)):
                    predictions_flat = np.array(predictions).flatten()
                    y_test_flat = np.array(y_test).flatten()
                else:
                    predictions_flat = np.array([predictions])
                    y_test_flat = np.array([y_test])

                # 添加到累积列表
                all_predictions.extend(predictions_flat.tolist())
                all_true_labels.extend(y_test_flat.tolist())

                # 计算当前折叠的准确率
                fold_accuracy = np.mean(predictions_flat == y_test_flat)
                fold_accuracies.append(fold_accuracy)
                print(f"折叠 {fold + 1} 准确率: {fold_accuracy:.4f}")

            except Exception as e:
                print(f"折叠 {fold + 1} 训练出错: {e}")
                continue

        # 计算整体性能
        if all_predictions and all_true_labels:
            all_predictions = np.array(all_predictions)
            all_true_labels = np.array(all_true_labels)

            print(f"\n=== 最终评估 ===")
            print(f"总预测样本数: {len(all_predictions)}")
            print(f"总真实样本数: {len(all_true_labels)}")

            min_length = min(len(all_predictions), len(all_true_labels))
            if min_length > 0:
                all_predictions = all_predictions[:min_length]
                all_true_labels = all_true_labels[:min_length]

                accuracy = np.mean(all_predictions == all_true_labels)
                print(f"{self.model_type.upper()}模型最终准确率: {accuracy:.4f}")

                if len(fold_accuracies) > 0:
                    print(f"各折叠准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
                    print(f"平均折叠准确率: {np.mean(fold_accuracies):.4f}")

                unique_labels = np.unique(np.concatenate([all_true_labels, all_predictions]))
                print(f"实际出现的类别: {unique_labels}")

                label_names = []
                for label in sorted(unique_labels):
                    if label == 0:
                        label_names.append('下跌')
                    elif label == 1:
                        label_names.append('平稳')
                    elif label == 2:
                        label_names.append('上涨')
                    else:
                        label_names.append(f'类别{label}')

                print("\n分类报告:")
                print(classification_report(all_true_labels, all_predictions,
                                            labels=sorted(unique_labels),
                                            target_names=label_names))
            else:
                print("错误: 没有有效的预测结果")
        else:
            print("错误: 没有收集到任何预测结果")

        self.is_trained = True  # 标记模型已训练


    def run_realistic_backtest(self, features_df: pd.DataFrame, labels: np.ndarray,
                               full_data: pd.DataFrame):
        """运行现实回测 - 修复价格数据提取问题"""
        if not self.is_trained or self.model is None:
            print("模型未训练，无法进行回测")
            return

        # 获取预测结果
        features_array = features_df.values
        predictions, confidence_scores = self.model.predict(features_array)

        # 修复：查找正确的价格列名
        price_columns = ['Close', 'close', 'CLOSE', 'Price', 'price', 'Last', 'last']
        price_col = None

        for col in price_columns:
            if col in full_data.columns:
                price_col = col
                print(f"找到价格列: {price_col}")
                break

        if price_col is None:
            print("警告: 在full_data中未找到标准价格列")
            possible_price_cols = [col for col in full_data.columns if
                                   any(x in str(col).lower() for x in ['close', 'price', 'last'])]
            if possible_price_cols:
                price_col = possible_price_cols[0]
                print(f"使用可能的价格列: {price_col}")
            else:
                numeric_cols = full_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_col = numeric_cols[0]
                    print(f"使用数值列作为价格: {price_col}")
                else:
                    print("错误: 无法找到价格数据")
                    return None

        # 修复：查找波动率列
        vol_columns = ['Volatility_20', 'volatility_20', 'Volatility', 'volatility']
        vol_col = None

        for col in vol_columns:
            if col in full_data.columns:
                vol_col = col
                print(f"找到波动率列: {vol_col}")
                break

        if vol_col is None:
            print("警告: 未找到波动率列，使用默认波动率")
            volatilities = np.full(len(predictions), 0.02)
        else:
            common_idx = features_df.index.intersection(full_data.index)
            if len(common_idx) < len(features_df.index):
                print(f"警告: 索引不完全匹配，使用共同索引 {len(common_idx)}/{len(features_df.index)}")

            volatilities = full_data.loc[common_idx, vol_col].fillna(0.02).values
            if len(volatilities) != len(predictions):
                print(f"波动率长度不匹配: {len(volatilities)} vs {len(predictions)}")
                min_len = min(len(volatilities), len(predictions))
                volatilities = volatilities[:min_len]
                predictions = predictions[:min_len]
                confidence_scores = confidence_scores[:min_len]

        # 获取价格数据 - 修复：确保提取的是数值而不是序列
        try:
            common_idx = features_df.index.intersection(full_data.index)
            price_data = full_data.loc[common_idx, price_col]

            if hasattr(price_data, 'values'):
                prices = price_data.values
            else:
                prices = price_data

            if hasattr(prices, 'flatten'):
                prices = prices.flatten()

            print(f"价格数据形状: {prices.shape if hasattr(prices, 'shape') else 'No shape'}")

            # 如果长度不匹配，进行调整
            if len(prices) != len(predictions):
                print(f"价格长度不匹配: {len(prices)} vs {len(predictions)}")
                min_len = min(len(prices), len(predictions))
                prices = prices[:min_len]
                predictions = predictions[:min_len]
                confidence_scores = confidence_scores[:min_len]
                volatilities = volatilities[:min_len] if len(volatilities) >= min_len else np.full(min_len, 0.02)

        except Exception as e:
            print(f"获取价格数据出错: {e}")
            print("创建模拟价格数据用于回测")
            prices = np.linspace(100, 200, len(predictions))
            print(f"使用模拟价格数据，范围: {prices[0]:.2f} - {prices[-1]:.2f}")

        # 准备时间戳
        timestamps = features_df.index.to_pydatetime()
        if len(timestamps) != len(predictions):
            timestamps = timestamps[:len(predictions)]

        print(f"回测数据准备完成:")
        print(f"- 预测数量: {len(predictions)}")
        print(f"- 价格数量: {len(prices)}")
        print(f"- 波动率数量: {len(volatilities)}")
        print(f"- 时间戳数量: {len(timestamps)}")

        # 运行回测
        print("开始执行回测逻辑...")
        results = self.backtester.run_backtest(
            predictions.tolist(),
            confidence_scores.tolist(),
            prices.tolist(),
            volatilities.tolist(),
            timestamps.tolist()
        )

        # 绘制回测结果图表
        if results and len(prices) > 0:
            print("\n生成回测可视化图表...")

            # 基本图表
            self.backtester.plot_backtest_results(
                prices=prices,
                timestamps=timestamps,
                predictions=predictions,
                save_path='backtest_results.png'
            )

            # 详细分析图表
            self.backtester.plot_detailed_analysis(
                prices=prices,
                timestamps=timestamps,
                predictions=predictions,
                save_path='backtest_results.png'
            )

            print("回测可视化完成！图表已保存为 'backtest_results.png' 和 'backtest_results_detailed.png'")

        return results


# 主程序
if __name__ == "__main__":
    print("启动多模型趋势预测系统...")

    try:
        # 可以选择不同的模型类型
        model_types = ['lstm', 'transformer', 'ensemble']
        selected_model = 'lstm'  # 可以更改为 'transformer' 或 'ensemble'

        # LSTM特定参数 - 使用增强参数
        lstm_kwargs = {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout_rate': 0.3
        }

        predictor = AdvancedTrendPredictor(
            model_type=selected_model,
            **lstm_kwargs
        )

        features_df, labels, full_data = predictor.run_advanced_pipeline(compare_models=True)

        print(f"\n使用{selected_model.upper()}模型的系统运行完成！")
        print("回测图表已生成，请查看当前目录下的PNG文件")

        # 让程序保持运行以便查看图表
        if matplotlib.get_backend() != 'Agg':
            print("按Enter键退出...")
            input()

    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息

# if __name__ == "__main__":
#     # 可以选择不同的模型类型
#     model_types = ['lstm', 'transformer', 'ensemble']
#     selected_model = 'lstm'  # 可以更改为 'transformer' 或 'ensemble'
#
#     # LSTM特定参数 - 使用增强参数
#     lstm_kwargs = {
#         'hidden_size': 128,
#         'num_layers': 3,
#         'dropout_rate': 0.3
#     }
#
#     predictor = AdvancedTrendPredictor(
#         model_type=selected_model,
#         **lstm_kwargs
#     )
#
#     # 1. 获取和处理数据
#     data = AdvancedTrendPredictor().data_processor.fetch_data(period="2y")
#     tech_data = AdvancedTrendPredictor().data_processor.enhanced_calculate_technical_indicators(data)
#     vol_features = AdvancedTrendPredictor().data_processor.calculate_volatility_features_vectorized(tech_data)
#
#     # 2. 合并特征和创建标签
#     all_features = pd.concat([tech_data, vol_features], axis=1)
#
#     # 修复：更新特征列选择以匹配新的技术指标名称
#     feature_columns = [col for col in all_features.columns if any(x in col for x in
#                                                                   ['SMA', 'EMA', 'RSI', 'BB', 'volatility',
#                                                                    'momentum', '_vol_', 'returns', 'log_returns',
#                                                                    'price_ratio', 'ROC', 'volume', 'OBV',
#                                                                    'price_position'])]
#     features_df = all_features[feature_columns].dropna()
#
#     # 使用增强版标签生成
#     features_df, labels = AdvancedTrendPredictor().data_processor.create_enhanced_labels(features_df, data)
#     results = predictor.run_realistic_backtest(features_df, labels, data)

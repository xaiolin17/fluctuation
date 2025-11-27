# -*- coding: utf-8 -*-
"""
完整版：基于技术指标波动率的市场反转点预测系统
包含Transformer和LSTM双模型支持
"""
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RobustMarketDataset(Dataset):
    """
    稳健的市场数据集
    处理时间序列数据的批量加载
    """

    def __init__(self, features, labels, sequence_length=30, prediction_horizon=1):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        features_seq = self.features[idx:end_idx]

        # 预测未来prediction_horizon步的标签
        label_idx = end_idx + self.prediction_horizon - 1
        if label_idx < len(self.labels):
            label = self.labels[label_idx]
        else:
            label = 1  # 默认平稳

        return torch.FloatTensor(features_seq), torch.LongTensor([label])


class EfficientMarketDataProcessor:
    """
    高效市场数据处理器
    使用向量化计算提升性能
    """

    def __init__(self, symbol='^NDX', initial_period='2y'):
        self.symbol = symbol
        self.initial_period = initial_period
        self.data = None
        self.technical_cache = {}  # 技术指标缓存
        self.volatility_windows = [20, 60]

    def fetch_data(self, period="2y", cache_dir="./data_cache"):
        """获取数据（带缓存）"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"ndx_data_{period}.pkl")

        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 7:
                print("从缓存加载数据...")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    self.data = cached_data
                    print(f"缓存数据加载完成，共{len(self.data)}条记录")
                    return self.data
                except Exception as e:
                    print(f"缓存加载失败: {e}，重新下载数据...")

        print("正在下载纳斯达克指数数据...")
        try:
            data = yf.download("^NDX", period=period, interval='1h', proxy="http://127.0.0.1:7890")
            data = data.dropna()

            if data.empty:
                raise ValueError("下载的数据为空")

            print(f"数据下载完成，共{len(data)}条记录")

            # 修复：处理MultiIndex列名问题
            if isinstance(data.columns, pd.MultiIndex):
                print("检测到MultiIndex列，进行扁平化处理...")
                # 方法1：直接使用第一层列名
                data.columns = data.columns.get_level_values(0)
                # 或者方法2：重命名列
                # data = data.rename(columns={col: col[0] for col in data.columns})

            print(f"处理后的列名: {data.columns.tolist()}")

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"数据已保存到缓存: {cache_file}")
            except Exception as e:
                print(f"缓存保存失败: {e}")

            self.data = data
            return data

        except Exception as e:
            print(f"数据下载失败: {e}")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    self.data = cached_data
                    return self.data
                except Exception as cache_error:
                    print(f"缓存数据也失败: {cache_error}")
            raise e

    def calculate_technical_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算技术指标（性能优化）
        """
        print("向量化计算技术指标...")
        result_df = df.copy()

        # 确保价格数据是Series而不是DataFrame
        if isinstance(result_df['Close'], pd.DataFrame):
            print("检测到Close是DataFrame，转换为Series...")
            # 如果有多个列，取第一列
            if len(result_df['Close'].columns) > 1:
                result_df['Close'] = result_df['Close'].iloc[:, 0]
            else:
                result_df['Close'] = result_df['Close'].squeeze()

        # 基础收益率
        result_df['returns'] = result_df['Close'].pct_change()
        result_df['log_returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))

        # 向量化移动平均线
        for window in [5, 10, 20, 50]:
            result_df[f'SMA_{window}'] = result_df['Close'].rolling(window=window, min_periods=window).mean()
            result_df[f'EMA_{window}'] = result_df['Close'].ewm(span=window, min_periods=window).mean()

        # 向量化RSI
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        result_df['RSI_14'] = compute_rsi(result_df['Close'], 14)
        result_df['RSI_21'] = compute_rsi(result_df['Close'], 21)

        # 向量化布林带
        for window in [20]:
            rolling_mean = result_df['Close'].rolling(window=window).mean()
            rolling_std = result_df['Close'].rolling(window=window).std()
            result_df[f'BB_upper_{window}'] = rolling_mean + 2 * rolling_std
            result_df[f'BB_lower_{window}'] = rolling_mean - 2 * rolling_std
            result_df[f'BB_middle_{window}'] = rolling_mean
            result_df[f'BB_width_{window}'] = (2 * 2 * rolling_std) / rolling_mean

        # 向量化波动率
        result_df['Volatility_20'] = result_df['returns'].rolling(window=20).std()
        result_df['Volatility_60'] = result_df['returns'].rolling(window=60).std()

        # 价格动量指标
        result_df['Momentum_5'] = result_df['Close'] / result_df['Close'].shift(5) - 1
        result_df['Momentum_10'] = result_df['Close'] / result_df['Close'].shift(10) - 1

        # 修复：确保布林带位置计算正确
        # 价格位置相对布林带 - 修正版本
        if 'BB_upper_20' in result_df.columns and 'BB_lower_20' in result_df.columns:
            # 确保所有相关列都是Series
            close_series = result_df['Close'].squeeze()
            bb_upper_series = result_df['BB_upper_20']
            bb_lower_series = result_df['BB_lower_20']

            bb_range = bb_upper_series - bb_lower_series
            # 避免除零错误
            bb_range = bb_range.replace(0, np.nan)
            result_df['BB_position'] = (close_series - bb_lower_series) / bb_range
        else:
            # 如果布林带计算失败，使用默认值
            result_df['BB_position'] = 0.5

        # 缓存结果
        self.technical_cache['last_calculation'] = datetime.now()

        return result_df

    def calculate_volatility_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算波动率特征
        """
        print("向量化计算波动率特征...")
        volatility_features = pd.DataFrame(index=df.index)

        # 基础指标列表
        base_indicators = ['RSI_14', 'RSI_21', 'BB_width_20', 'Volatility_20',
                           'Volatility_60', 'Momentum_5', 'Momentum_10', 'BB_position']

        for window in self.volatility_windows:
            for indicator in base_indicators:
                if indicator in df.columns:
                    # 确保指标是Series
                    indicator_series = df[indicator]
                    if isinstance(indicator_series, pd.DataFrame):
                        indicator_series = indicator_series.squeeze()

                    vol_col = f'{indicator}_vol_{window}'
                    volatility_features[vol_col] = indicator_series.rolling(window=window, min_periods=window).std()

        return volatility_features

    def create_smart_labels(self, df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20],
                            volatility_adjust: bool = True) -> np.ndarray:
        """
        智能标签定义
        结合多时间框架和波动率调整
        """
        print("创建智能标签...")
        labels = np.ones(len(df))  # 默认平稳

        # 确保Close是Series
        close_series = df[('Close', '^NDX')]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()

        # 计算自适应阈值
        if volatility_adjust:
            # 使用近期波动率调整阈值
            returns_series = df[('returns', '')]
            if isinstance(returns_series, pd.DataFrame):
                returns_series = returns_series.squeeze()

            recent_volatility = returns_series.rolling(window=20).std().fillna(0.02)
            threshold_multiplier = 1.5  # 波动率倍数
            dynamic_threshold = recent_volatility * threshold_multiplier
        else:
            dynamic_threshold = pd.Series(0.015, index=df.index)

        # 多时间框架趋势判断
        trend_signals = pd.DataFrame(index=df.index)

        for period in lookback_periods:
            # 价格变化
            price_change = (close_series - close_series.shift(period)) / close_series.shift(period)
            trend_signals[f'trend_{period}'] = price_change

            # 移动平均趋势
            sma_short = close_series.rolling(window=period // 2).mean()
            sma_long = close_series.rolling(window=period).mean()
            trend_signals[f'ma_trend_{period}'] = (sma_short - sma_long) / sma_long

        # 综合趋势得分
        trend_score = trend_signals.mean(axis=1, skipna=True)

        # 局部极值点检测
        def detect_local_extremes(series, window=10):
            """检测局部极值点"""
            local_max = series.rolling(window=window, center=True).max() == series
            local_min = series.rolling(window=window, center=True).min() == series
            return local_max, local_min

        local_max, local_min = detect_local_extremes(close_series, window=15)

        # 智能标签分配
        for i in range(max(lookback_periods) + 1, len(df)):
            current_threshold = dynamic_threshold.iloc[i]

            # 基于趋势得分和阈值
            if trend_score.iloc[i] > current_threshold and not local_min.iloc[i]:
                labels[i] = 2  # 上涨趋势
            elif trend_score.iloc[i] < -current_threshold and not local_max.iloc[i]:
                labels[i] = 0  # 下跌趋势
            # 在极值点附近保持平稳标签，避免噪声

            # 额外条件：布林带位置
            if 'BB_position' in df.columns:
                bb_pos = df['BB_position'].iloc[i]
                if isinstance(bb_pos, pd.Series):
                    bb_pos = bb_pos.iloc[0] if len(bb_pos) > 0 else 0.5

                if bb_pos > 0.8 and labels[i] == 2:  # 上轨附近，可能超买
                    labels[i] = 1  # 改为平稳
                elif bb_pos < 0.2 and labels[i] == 0:  # 下轨附近，可能超卖
                    labels[i] = 1  # 改为平稳

        print(f"标签分布 - 下跌: {sum(labels == 0)}, 平稳: {sum(labels == 1)}, 上涨: {sum(labels == 2)}")
        return labels


class RealisticBacktester:
    """
    现实回测系统
    考虑交易成本、滑点等现实因素
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 0.1%交易成本
        self.slippage = 0.0005  # 0.05%滑点
        self.positions = []
        self.trade_log = []

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
        vol_adjustment = max(0.5, 1 - volatility * 10)
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

        print(f"调试: 交易价格转换 - 原始: {price}, 转换后: {price_value}")  # 调试信息

        # 应用滑点
        execution_price = price_value * (1 + self.slippage) if signal == 2 else price_value * (1 - self.slippage)

        # 计算仓位大小
        position_size = self.calculate_position_size(current_capital, confidence, volatility)

        # 计算交易成本
        trade_cost = position_size * self.transaction_cost

        trade_record = {
            'timestamp': timestamp,
            'signal': signal,
            'price': execution_price,
            'position_size': position_size,
            'cost': trade_cost,
            'shares': position_size / execution_price if signal == 2 else 0
        }

        self.trade_log.append(trade_record)
        return trade_record

    def run_backtest(self, predictions: List[int], confidence_scores: List[float],
                     prices: List[float], volatilities: List[float],
                     timestamps: List[datetime]) -> Dict[str, float]:
        """
        运行现实回测 - 修复价格数据类型问题
        """
        print("运行现实回测...")
        print(f"预测列表长度: {len(predictions)}")
        print(f"置信度列表长度: {len(confidence_scores)}")
        print(f"价格列表长度: {len(prices)}")
        print(f"波动率列表长度: {len(volatilities)}")
        print(f"时间戳列表长度: {len(timestamps)}")

        # 检查数据类型
        print(f"价格数据类型: {type(prices[0])}")
        print(f"前5个价格值: {prices[:5]}")

        # 检查置信度分数的结构
        if confidence_scores and hasattr(confidence_scores[0], '__len__'):
            print(f"置信度分数结构: 每个样本有 {len(confidence_scores[0])} 个类别的概率")
        else:
            print(f"置信度分数结构: {type(confidence_scores[0])}")

        capital = self.initial_capital
        position = 0
        max_drawdown = 0
        peak_capital = self.initial_capital
        trades = 0

        equity_curve = [capital]

        for i in range(1, len(predictions)):
            # 修复：确保current_price是单个浮点数
            current_price = prices[i]
            if hasattr(current_price, '__len__') and not isinstance(current_price, str):
                if len(current_price) > 0:
                    current_price = float(current_price[0])
                else:
                    current_price = 100.0  # 默认价格
            else:
                current_price = float(current_price)

            current_vol = volatilities[i] if i < len(volatilities) else 0.02
            current_timestamp = timestamps[i]

            # 调试：检查置信度分数的访问方式
            if i < len(confidence_scores):
                if hasattr(confidence_scores[i], '__len__') and len(confidence_scores[i]) > predictions[i]:
                    confidence = confidence_scores[i][predictions[i]]
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            # 当前持仓价值
            current_value = capital + (position * current_price if position > 0 else 0)

            # 更新最大回撤
            if current_value > peak_capital:
                peak_capital = current_value
            drawdown = (peak_capital - current_value) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            # 交易信号
            signal = predictions[i]
            prev_signal = predictions[i - 1] if i > 0 else 1

            # 执行交易逻辑
            if signal != prev_signal:
                # 平仓现有持仓
                if position > 0:
                    close_trade = self.execute_trade(0, current_price, capital,
                                                     confidence, current_vol, current_timestamp)
                    capital = position * current_price - close_trade['cost']
                    position = 0
                    trades += 1

                # 开新仓
                if signal == 2:  # 买入信号
                    buy_trade = self.execute_trade(2, current_price, capital,
                                                   confidence, current_vol, current_timestamp)
                    position = buy_trade['shares']
                    capital -= buy_trade['position_size'] + buy_trade['cost']
                    trades += 1

            equity_curve.append(current_value)

        # 最终平仓
        if position > 0:
            final_value = position * prices[-1]
            capital += final_value

        # 计算绩效指标
        total_return = (capital - self.initial_capital) / self.initial_capital
        annual_return = total_return / (len(predictions) / 252)  # 近似年化

        # 计算夏普比率（简化）
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # 胜率计算
        winning_trades = len([t for t in self.trade_log if t.get('profit', 0) > 0])
        win_rate = winning_trades / len(self.trade_log) if self.trade_log else 0

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': trades,
            'win_rate': win_rate,
            'final_equity': equity_curve[-1] if equity_curve else self.initial_capital
        }

        self.print_backtest_results(results)
        return results

    def print_backtest_results(self, results: Dict[str, float]):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("现实回测结果")
        print("=" * 60)
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annual_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']:.2%}")
        print("=" * 60)


class SimpleLSTMPredictor(nn.Module):
    """
    简化的LSTM预测器
    适用于时间序列分类任务
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super(SimpleLSTMPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 注意力机制（可选）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 分类器
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3个类别
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def attention_forward(self, lstm_output):
        """注意力机制前向传播"""
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 加权求和
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        return context_vector, attention_weights

    def forward(self, x, return_attention: bool = False):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            return_attention: 是否返回注意力权重

        Returns:
            分类logits或 (logits, attention_weights)
        """
        # 处理二维输入：添加序列长度维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        batch_size, seq_len, _ = x.shape

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            # 双向LSTM：合并最后两个时间步的隐藏状态
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # 单向LSTM：使用最后一个时间步的隐藏状态
            final_hidden = hidden[-1]

        # 使用注意力机制
        context_vector, attention_weights = self.attention_forward(lstm_out)

        # 分类
        output = self.classifier(context_vector)

        if return_attention:
            return output, attention_weights
        else:
            return output

    def get_feature_importance(self, dataloader, device='cpu'):
        """计算特征重要性（基于梯度）"""
        self.eval()
        feature_gradients = []

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_features.requires_grad = True

            outputs = self(batch_features)

            # 计算对平稳类别的梯度
            target_class = 1  # 平稳类别
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)

            # 创建目标
            batch_size = outputs.shape[0]
            targets = torch.full((batch_size,), target_class, dtype=torch.long).to(device)

            # 计算损失并反向传播
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            loss.backward()

            # 获取梯度
            gradients = batch_features.grad.data.cpu().numpy()
            feature_gradients.append(np.abs(gradients).mean(axis=(0, 1)))

        if feature_gradients:
            importance = np.mean(feature_gradients, axis=0)
            return importance
        else:
            return None


class ImprovedTransformerPredictor(nn.Module):
    """
    改进的Transformer预测器，支持多种池化策略
    """

    def __init__(self, input_dim: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, dim_feedforward: int = 64,
                 pooling: str = 'last'):
        super(ImprovedTransformerPredictor, self).__init__()

        self.pooling = pooling  # 'last', 'mean', 'max'

        self.input_projection = nn.Linear(input_dim, 32)

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
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

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
            self.model = SimpleLSTMPredictor(input_dim, **model_kwargs)
        elif model_type == 'transformer':
            self.model = ImprovedTransformerPredictor(input_dim, **model_kwargs)
        elif model_type == 'ensemble':
            self.models = {
                'lstm': SimpleLSTMPredictor(input_dim, **model_kwargs.get('lstm', {})),
                'transformer': ImprovedTransformerPredictor(input_dim, **model_kwargs.get('transformer', {}))
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4):
        """创建优化器"""
        if self.model_type == 'ensemble':
            params = []
            for model in self.models.values():
                params.extend(list(model.parameters()))
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

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

        # 训练循环
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

        self.is_fitted = True
        return train_losses, val_losses

    def predict(self, X: np.ndarray, return_probabilities: bool = False):
        """预测 - 修复版本"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)

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
                outputs = torch.stack(outputs).mean(dim=0)
            else:
                outputs = self.model(X_tensor)

            # 修复：确保输出维度正确
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            probabilities = torch.softmax(outputs, dim=1).numpy()
            predictions = np.argmax(probabilities, axis=1)

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
        self.update_frequency = 100
        self.batch_size = 32
        self.online_data_buffer = []
        self.update_count = 0

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

        self.online_data_buffer.append((X_new, y_new))
        self.update_count += 1

        if self.update_count >= self.update_frequency:
            self._update_model()

    def _update_model(self):
        """更新模型"""
        if not self.online_data_buffer:
            return

        print("执行增量模型更新...")
        all_X = np.vstack([data[0] for data in self.online_data_buffer])
        all_y = np.concatenate([data[1] for data in self.online_data_buffer])

        # 使用新数据微调模型
        self.predictor.fit(all_X, all_y, epochs=10)

        self.online_data_buffer = []
        self.update_count = 0

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
        self.data_processor = EfficientMarketDataProcessor()
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model = None
        self.backtester = RealisticBacktester()
        self.is_trained = False

    def run_model_comparison(self, features: np.ndarray, labels: np.ndarray):
        """运行模型比较"""
        print("运行模型比较...")

        # 分割数据
        split_idx = int(0.8 * len(features))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        # 创建集成模型进行比较
        ensemble_predictor = MultiModelPredictor(
            X_train.shape[1],
            model_type='ensemble',
            lstm={'hidden_dim': 64, 'num_layers': 2},
            transformer={'num_heads': 4, 'num_layers': 2}
        )

        # 训练集成模型
        ensemble_predictor.fit(X_train, y_train, epochs=30)

        # 比较模型性能
        results = ensemble_predictor.compare_models(X_test, y_test)

        return results

    def run_advanced_pipeline(self, compare_models: bool = False):
        """运行高级预测流程 - 修复版本"""
        print("开始高级趋势预测流程...")

        # 1. 获取和处理数据
        data = self.data_processor.fetch_data(period="2y")
        tech_data = self.data_processor.calculate_technical_indicators_vectorized(data)
        vol_features = self.data_processor.calculate_volatility_features_vectorized(tech_data)

        # 2. 合并特征和创建标签
        all_features = pd.concat([tech_data, vol_features], axis=1)

        # 调试：检查数据列名
        print("原始数据列名:", data.columns.tolist())
        print("技术指标数据列名:", tech_data.columns.tolist()[:10])
        print("波动率特征列名:", vol_features.columns.tolist()[:10])

        feature_columns = [col for col in all_features.columns if any(x in col for x in
                                                                      ['SMA', 'EMA', 'RSI', 'BB', 'Volatility',
                                                                       'Momentum', '_vol_'])]
        features_df = all_features[feature_columns].dropna()

        labels = self.data_processor.create_smart_labels(all_features)

        # 对齐数据
        common_idx = features_df.index.intersection(all_features.index)
        features_df = features_df.loc[common_idx]
        labels = labels[all_features.index.isin(common_idx)]

        print(f"特征形状: {features_df.shape}")
        print(f"标签分布: {np.unique(labels, return_counts=True)}")

        # 3. 模型比较（可选）
        if compare_models:
            self.run_model_comparison(features_df.values, labels)

        # 4. 使用选定模型训练
        features_array = features_df.values

        if self.model_type == 'ensemble':
            self.model = EnhancedIncrementalLearningModel(
                features_array.shape[1],
                model_type='ensemble',
                lstm={'hidden_dim': 64, 'num_layers': 2},
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
        # 确保传递包含价格数据的原始数据
        results = self.run_realistic_backtest(features_df, labels, data)  # 传递 data 而不是 all_features

        if results:
            print("回测完成!")
        else:
            print("回测失败!")

        return features_df, labels, all_features

    def train_with_incremental_learning(self, features: np.ndarray, labels: np.ndarray):
        """使用增量学习训练模型 - 修复版本"""
        print(f"使用{self.model_type.upper()}模型进行增量学习训练...")

        tscv = TimeSeriesSplit(n_splits=5)
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
                    self.model.initial_fit(X_train, y_train, epochs=30)
                else:
                    # 后续折叠使用增量学习
                    self.model.partial_fit(X_train, y_train)

                # 在测试集上预测
                predictions, confidence_scores = self.model.predict(X_test)

                # 调试信息
                print(f"预测结果类型: {type(predictions)}")
                print(f"预测结果形状: {getattr(predictions, 'shape', 'No shape')}")
                print(f"预测的唯一值: {np.unique(predictions)}")
                print(f"真实标签的唯一值: {np.unique(y_test)}")

                # 处理不同的预测结果格式
                if isinstance(predictions, (list, np.ndarray)):
                    predictions_flat = np.array(predictions).flatten()
                    y_test_flat = np.array(y_test).flatten()
                else:
                    # 如果是单个预测值
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

            # 确保形状匹配
            min_length = min(len(all_predictions), len(all_true_labels))
            if min_length > 0:
                all_predictions = all_predictions[:min_length]
                all_true_labels = all_true_labels[:min_length]

                accuracy = np.mean(all_predictions == all_true_labels)
                print(f"{self.model_type.upper()}模型最终准确率: {accuracy:.4f}")

                if len(fold_accuracies) > 0:
                    print(f"各折叠准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
                    print(f"平均折叠准确率: {np.mean(fold_accuracies):.4f}")

                # 修复：动态确定类别标签
                unique_labels = np.unique(np.concatenate([all_true_labels, all_predictions]))
                print(f"实际出现的类别: {unique_labels}")

                # 创建对应的标签名称
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

        self.is_trained = True

    def run_realistic_backtest(self, features_df: pd.DataFrame, labels: np.ndarray,
                               full_data: pd.DataFrame):
        """运行现实回测 - 修复价格数据提取问题"""
        if not self.is_trained or self.model is None:
            print("模型未训练，无法进行回测")
            return

        # 获取预测结果
        features_array = features_df.values
        predictions, confidence_scores = self.model.predict(features_array)

        # 调试：检查 full_data 的列名
        print("调试信息 - full_data 列名:", full_data.columns.tolist()[:10])
        print("调试信息 - features_df 索引:", len(features_df.index))
        print("调试信息 - full_data 索引:", len(full_data.index))

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
            print("可用列:", [col for col in full_data.columns if not col.startswith(('_', 'vol_'))][:20])
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

            # 修复：确保提取单个价格值而不是序列
            price_data = full_data.loc[common_idx, price_col]

            # 如果price_data是DataFrame或Series，提取数值
            if hasattr(price_data, 'values'):
                prices = price_data.values
            else:
                prices = price_data

            # 确保prices是1D数组
            if hasattr(prices, 'flatten'):
                prices = prices.flatten()

            print(f"价格数据形状: {prices.shape if hasattr(prices, 'shape') else 'No shape'}")
            print(f"价格数据类型: {type(prices)}")
            print(f"前5个价格值: {prices[:5] if hasattr(prices, '__getitem__') else prices}")

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
            # 创建模拟价格数据用于测试
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
        return results

# 主程序
if __name__ == "__main__":
    print("启动多模型趋势预测系统...")

    try:
        # 可以选择不同的模型类型
        model_types = ['lstm', 'transformer', 'ensemble']
        selected_model = 'lstm'  # 可以更改为 'transformer' 或 'ensemble'

        # LSTM特定参数
        lstm_kwargs = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True
        }

        predictor = AdvancedTrendPredictor(
            model_type=selected_model,
            **lstm_kwargs
        )

        features_df, labels, full_data = predictor.run_advanced_pipeline(compare_models=True)

        print(f"\n使用{selected_model.upper()}模型的系统运行完成！")

    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback

        traceback.print_exc()
# -*- coding: utf-8 -*-
"""
基于9转序列和马丁格尔策略的市场交易回测系统
改进版：从本地CSV文件读取数据
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LocalCSVDataProcessor:
    """
    本地CSV文件数据处理器
    """

    def __init__(self, csv_file_path: str = None):
        """
        初始化数据处理器

        Args:
            csv_file_path: CSV文件路径，如果为None则尝试自动查找
        """
        self.csv_file_path = csv_file_path
        self.data = None

    def find_csv_file(self, directory: str = "."):
        """在当前目录查找CSV文件"""
        print(f"在目录 {directory} 中查找CSV文件...")
        csv_files = []

        for file in os.listdir(directory):
            if file.endswith('.csv'):
                csv_files.append(file)
                print(f"找到CSV文件: {file}")

        if csv_files:
            # 优先选择包含nasdaq或ndx的文件
            for file in csv_files:
                if 'nasdaq' in file.lower() or 'ndx' in file.lower():
                    return os.path.join(directory, file)
            # 否则返回第一个
            return os.path.join(directory, csv_files[0])

        print("未找到CSV文件")
        return None

    def read_csv_data(self, csv_file_path: str = None) -> pd.DataFrame:
        """
        从CSV文件读取数据

        Args:
            csv_file_path: CSV文件路径

        Returns:
            处理后的市场数据DataFrame
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path

        if csv_file_path is None:
            csv_file_path = self.find_csv_file()

        if csv_file_path is None or not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_file_path}")

        print(f"正在读取CSV文件: {csv_file_path}")

        try:
            # 尝试不同编码读取
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    # 先读取前几行查看数据结构
                    with open(csv_file_path, 'r', encoding=encoding) as f:
                        first_lines = [f.readline() for _ in range(3)]

                    print(f"使用编码 {encoding} 读取的样本行:")
                    for line in first_lines:
                        print(f"  {line.strip()}")

                    # 现在读取整个文件
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    print(f"成功使用编码 {encoding} 读取文件")
                    print(f"数据形状: {df.shape}")
                    print(f"列名: {list(df.columns)}")

                    # 显示前几行数据
                    print("\n前5行数据:")
                    print(df.head())

                    break
                except UnicodeDecodeError:
                    print(f"编码 {encoding} 失败，尝试下一个...")
                    continue
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            raise

        # 处理数据
        processed_df = self.process_csv_data(df)
        self.data = processed_df
        return processed_df

    def process_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理CSV数据，转换为标准格式

        Args:
            df: 原始CSV数据

        Returns:
            处理后的DataFrame
        """
        print("正在处理CSV数据...")

        # 创建副本
        result_df = df.copy()

        # 显示原始列名
        print(f"原始列名: {list(result_df.columns)}")

        # 查找日期列
        date_columns = []
        for col in result_df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['date', 'datum', '时间', '日期']):
                date_columns.append(col)

        if date_columns:
            date_col = date_columns[0]
            print(f"使用日期列: {date_col}")

            # 转换日期格式
            try:
                # 尝试多种日期格式
                for date_format in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']:
                    try:
                        result_df['Date'] = pd.to_datetime(result_df[date_col], format=date_format)
                        print(f"使用日期格式: {date_format}")
                        break
                    except:
                        continue

                # 如果以上格式都不行，使用自动检测
                if 'Date' not in result_df.columns:
                    result_df['Date'] = pd.to_datetime(result_df[date_col])

                # 设置为索引并排序
                result_df.set_index('Date', inplace=True)
                result_df.sort_index(inplace=True)

            except Exception as e:
                print(f"日期转换失败: {e}")
                # 使用默认日期范围
                result_df['Date'] = pd.date_range(end=datetime.now(), periods=len(result_df), freq='D')
                result_df.set_index('Date', inplace=True)

        # 查找价格列
        price_mappings = {
            'open': ['开盘', 'open', 'eröffnung', '开盘价', '开盘价'],
            'high': ['高', 'high', '最高', 'hoch', '最高价'],
            'low': ['低', 'low', '最低', 'tief', '最低价'],
            'close': ['收盘', 'close', '收盘价', 'schluss', 'letzter', 'last'],
            'volume': ['成交量', 'volume', 'vol', '交易量']
        }

        standard_columns = {}

        for standard_name, possible_names in price_mappings.items():
            for col in result_df.columns:
                col_lower = str(col).lower()
                for possible in possible_names:
                    if possible.lower() in col_lower:
                        standard_columns[standard_name] = col
                        print(f"将列 '{col}' 映射为 '{standard_name}'")
                        break
                if standard_name in standard_columns:
                    break

        # 创建标准DataFrame
        final_data = pd.DataFrame(index=result_df.index)

        # 复制数据
        for std_name, orig_name in standard_columns.items():
            final_data[std_name.capitalize()] = result_df[orig_name]

        # 如果没有Volume列，添加一个默认值
        if 'Volume' not in final_data.columns:
            final_data['Volume'] = 1000000  # 默认成交量

        # 确保所有必需列都存在
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in final_data.columns:
                print(f"警告: 缺少列 '{col}'，使用Close值填充")
                if 'Close' in final_data.columns:
                    final_data[col] = final_data['Close']
                else:
                    # 如果连Close都没有，使用随机数据
                    final_data[col] = np.random.randn(len(final_data)) * 100 + 1000

        # 确保数据按日期排序
        final_data.sort_index(inplace=True)

        print(f"处理后的数据形状: {final_data.shape}")
        print(f"数据时间范围: {final_data.index[0]} 到 {final_data.index[-1]}")

        return final_data

    def fetch_data(self, csv_file_path: str = None, cache_dir="./data_cache"):
        """
        获取数据（带缓存）

        Args:
            csv_file_path: CSV文件路径
            cache_dir: 缓存目录

        Returns:
            市场数据DataFrame
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 生成缓存文件名
        if csv_file_path:
            cache_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        else:
            cache_name = "local_data"

        cache_file = os.path.join(cache_dir, f"{cache_name}_processed.pkl")

        # 检查缓存
        if os.path.exists(cache_file):
            try:
                print("从缓存加载处理后的数据...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.data = cached_data
                print(f"缓存数据加载完成，共{len(self.data)}条记录")
                return self.data
            except Exception as e:
                print(f"缓存加载失败: {e}，重新处理数据...")

        # 从CSV读取并处理数据
        data = self.read_csv_data(csv_file_path)

        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"处理后的数据已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"缓存保存失败: {e}")

        self.data = data
        return data


class EfficientMarketDataProcessor:
    """
    高效市场数据处理器（用于技术指标计算）
    """

    def __init__(self, csv_file_path: str = None):
        """
        初始化数据处理器

        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_processor = LocalCSVDataProcessor(csv_file_path)
        self.data = None

    def fetch_data(self, csv_file_path: str = None, cache_dir="./data_cache"):
        """获取数据（从CSV文件）"""
        self.data = self.csv_processor.fetch_data(csv_file_path, cache_dir)
        return self.data

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础技术指标
        """
        result_df = df.copy()

        # 确保有Close列
        if 'Close' not in result_df.columns:
            if '收盘' in result_df.columns:
                result_df['Close'] = result_df['收盘']
            else:
                result_df['Close'] = result_df.iloc[:, 0]  # 使用第一列作为价格

        # 收益率
        result_df['returns'] = result_df['Close'].pct_change()

        # 移动平均线
        result_df['SMA_20'] = result_df['Close'].rolling(window=20).mean()
        result_df['SMA_50'] = result_df['Close'].rolling(window=50).mean()

        # 波动率
        result_df['Volatility_20'] = result_df['returns'].rolling(window=20).std()

        # 布林带
        result_df['BB_middle'] = result_df['Close'].rolling(window=20).mean()
        bb_std = result_df['Close'].rolling(window=20).std()
        result_df['BB_upper'] = result_df['BB_middle'] + 2 * bb_std
        result_df['BB_lower'] = result_df['BB_middle'] - 2 * bb_std

        # RSI
        delta = result_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result_df['RSI'] = 100 - (100 / (1 + rs))

        return result_df

    def calculate_nine_sequence_signals(self, df: pd.DataFrame,
                                        sequence_length: int = 9) -> pd.DataFrame:
        """
        计算9转序列信号

        Args:
            df: 包含价格数据的DataFrame
            sequence_length: 序列长度，默认为9

        Returns:
            添加了9转序列信号的DataFrame
        """
        print(f"计算{sequence_length}转序列信号...")
        result_df = df.copy()

        # 确保有Close列
        if 'Close' not in result_df.columns:
            print("警告: 没有Close列，使用第一列作为价格")
            result_df['Close'] = result_df.iloc[:, 0]

        # 初始化信号列
        result_df['nine_seq_buy'] = 0  # 买入信号
        result_df['nine_seq_sell'] = 0  # 卖出信号

        # 计算收盘价的变化
        closes = result_df['Close'].values

        for i in range(sequence_length, len(result_df)):
            # 检查买入序列（连续下跌后的反转）
            buy_signal = True
            for j in range(sequence_length):
                if i - j < sequence_length:
                    buy_signal = False
                    break
                # 检查连续下跌：当前收盘价低于4天前的收盘价
                if i - j >= 4 and closes[i - j] > closes[i - j - 4]:
                    buy_signal = False
                    break

            if buy_signal:
                result_df.loc[result_df.index[i], 'nine_seq_buy'] = 1

            # 检查卖出序列（连续上涨后的反转）
            sell_signal = True
            for j in range(sequence_length):
                if i - j < sequence_length:
                    sell_signal = False
                    break
                # 检查连续上涨：当前收盘价高于4天前的收盘价
                if i - j >= 4 and closes[i - j] < closes[i - j - 4]:
                    sell_signal = False
                    break

            if sell_signal:
                result_df.loc[result_df.index[i], 'nine_seq_sell'] = -1

        # 统计信号数量
        buy_signals = result_df['nine_seq_buy'].sum()
        sell_signals = abs(result_df['nine_seq_sell'].sum())
        print(f"买入9转信号数量: {buy_signals}")
        print(f"卖出9转信号数量: {sell_signals}")

        return result_df


class NineSequenceMartingaleStrategy:
    """
    基于9转序列的马丁格尔策略
    在出现同方向9转信号时加仓
    """

    def __init__(self,
                 initial_position_size: float = 1000,
                 max_multiplier: int = 5,
                 profit_target: float = 0.02,
                 stop_loss: float = 0.05,
                 sequence_length: int = 9):
        """
        初始化9转序列马丁格尔策略

        Args:
            initial_position_size: 初始仓位大小
            max_multiplier: 最大加仓倍数
            profit_target: 盈利目标百分比
            stop_loss: 止损百分比
            sequence_length: 9转序列长度
        """
        self.initial_position_size = initial_position_size
        self.max_multiplier = max_multiplier
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.sequence_length = sequence_length

        # 交易状态
        self.consecutive_losses = 0
        self.total_loss = 0
        self.current_position = 0
        self.entry_price = 0
        self.last_signal_type = None  # 记录上一个信号类型
        self.signal_count = 0  # 连续同向信号计数

    def get_position_size(self, signal_type: str) -> float:
        """
        获取当前仓位大小

        Args:
            signal_type: 信号类型 ('buy' 或 'sell')
        """
        # 如果信号类型改变，重置连续同向信号计数
        if self.last_signal_type is not None and self.last_signal_type != signal_type:
            self.signal_count = 0
            self.consecutive_losses = 0

        self.last_signal_type = signal_type
        self.signal_count += 1

        # 基于连续亏损次数和同向信号次数计算仓位
        if self.consecutive_losses == 0:
            base_size = self.initial_position_size
        else:
            # 马丁格尔：每次亏损后加倍仓位，但不超过最大倍数
            martingale_multiplier = min(2 ** self.consecutive_losses, self.max_multiplier)
            base_size = self.initial_position_size * martingale_multiplier

        # 同向信号次数越多，仓位越大（但不超过基础仓位的2倍）
        signal_multiplier = min(1.0 + (self.signal_count - 1) * 0.2, 2.0)

        return base_size * signal_multiplier

    def record_trade_result(self, profit: float, signal_type: str):
        """记录交易结果"""
        if profit > 0:
            # 盈利：重置马丁格尔计数，但保持同向信号计数
            self.consecutive_losses = 0
            self.total_loss = 0
        else:
            # 亏损：增加连续亏损计数
            self.consecutive_losses += 1
            self.total_loss += abs(profit)

            # 如果达到最大加仓次数，强制重置
            if self.consecutive_losses >= self.max_multiplier:
                print(f"达到最大加仓次数{self.max_multiplier}，强制重置马丁格尔")
                self.consecutive_losses = 0
                self.total_loss = 0
                self.signal_count = 0

    def should_take_profit(self, current_price: float) -> bool:
        """检查是否应该止盈"""
        if self.entry_price == 0 or self.current_position == 0:
            return False
        profit_pct = abs(current_price - self.entry_price) / self.entry_price
        return profit_pct >= self.profit_target

    def should_stop_loss(self, current_price: float) -> bool:
        """检查是否应该止损"""
        if self.entry_price == 0 or self.current_position == 0:
            return False
        loss_pct = abs(current_price - self.entry_price) / self.entry_price
        return loss_pct >= self.stop_loss

    def update_position(self, shares: float, price: float, position_type: str):
        """更新持仓状态"""
        self.current_position = shares
        self.entry_price = price

        # 如果是开新仓，重置连续同向信号计数
        if shares > 0:
            self.signal_count = 1

    def clear_position(self):
        """清空持仓"""
        self.current_position = 0
        self.entry_price = 0


class RealisticBacktester:
    """
    现实回测系统
    基于9转序列和马丁格尔策略
    """

    def __init__(self,
                 initial_capital: float = 10000,
                 strategy_type: str = 'nine_seq_martingale',
                 **strategy_params):
        """
        初始化回测系统

        Args:
            initial_capital: 初始资金
            strategy_type: 策略类型 ('nine_seq_martingale')
            **strategy_params: 策略参数
        """
        self.initial_capital = initial_capital
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params
        self.transaction_cost = 0.001  # 0.1%交易成本
        self.slippage = 0.0005  # 0.05%滑点

        # 初始化策略
        if strategy_type == 'nine_seq_martingale':
            self.strategy = NineSequenceMartingaleStrategy(**strategy_params)

        # 交易记录
        self.trade_log = []
        self.equity_curve = []
        self.position_history = []

    def execute_trade(self,
                      signal_type: str,
                      price: float,
                      current_capital: float,
                      position_size: float,
                      timestamp: datetime,
                      reason: str = "") -> Dict:
        """
        执行交易

        Args:
            reason: 交易原因，如"9转买入信号"、"止盈"等
        """
        # 应用滑点
        if signal_type == 'buy':
            execution_price = price * (1 + self.slippage)
        else:  # sell
            execution_price = price * (1 - self.slippage)

        # 计算交易成本
        trade_value = position_size
        trade_cost = trade_value * self.transaction_cost

        # 计算股票数量
        shares = position_size / execution_price if signal_type == 'buy' else 0

        trade_record = {
            'timestamp': timestamp,
            'type': signal_type,
            'price': execution_price,
            'position_size': position_size,
            'cost': trade_cost,
            'shares': shares,
            'equity_before': current_capital,
            'reason': reason
        }

        self.trade_log.append(trade_record)
        return trade_record

    def run_nine_sequence_backtest(self,
                                   data: pd.DataFrame,
                                   use_rsi_filter: bool = True) -> Dict[str, float]:
        """
        运行9转序列马丁格尔策略回测

        Args:
            data: 包含9转信号的数据
            use_rsi_filter: 是否使用RSI过滤信号
        """
        print(f"运行9转序列马丁格尔策略回测 (最大加仓次数: {self.strategy.max_multiplier})...")

        capital = self.initial_capital
        max_drawdown = 0
        peak_capital = self.initial_capital
        trades = 0

        self.equity_curve = [capital]

        # 获取价格和时间戳
        prices = data['Close'].values
        timestamps = data.index.to_pydatetime()

        for i in range(1, len(data)):
            current_price = prices[i]
            current_timestamp = timestamps[i]
            current_row = data.iloc[i]

            # 当前权益
            current_shares = self.strategy.current_position
            current_equity = capital + (current_shares * current_price if current_shares > 0 else 0)
            self.equity_curve.append(current_equity)

            # 更新最大回撤
            if current_equity > peak_capital:
                peak_capital = current_equity
            drawdown = (peak_capital - current_equity) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            # 如果有持仓，检查止盈止损
            if current_shares > 0 and self.strategy.entry_price > 0:
                # 检查止盈
                if self.strategy.should_take_profit(current_price):
                    # 平仓止盈
                    trade_value = current_shares * current_price
                    trade = self.execute_trade('sell', current_price, capital,
                                               trade_value, current_timestamp, "止盈")
                    profit = trade_value - trade['cost']
                    capital += profit
                    self.strategy.record_trade_result(profit, 'buy')
                    self.strategy.clear_position()
                    trades += 1
                    continue

                # 检查止损
                if self.strategy.should_stop_loss(current_price):
                    # 平仓止损
                    trade_value = current_shares * current_price
                    trade = self.execute_trade('sell', current_price, capital,
                                               trade_value, current_timestamp, "止损")
                    loss = trade_value - trade['cost']
                    capital += loss
                    self.strategy.record_trade_result(-abs(loss), 'buy')
                    self.strategy.clear_position()
                    trades += 1
                    continue

            # 检查9转买入信号
            if current_row['nine_seq_buy'] == 1 and capital > 1000:
                # 可选：使用RSI过滤信号
                should_buy = True
                if use_rsi_filter and 'RSI' in current_row:
                    # RSI低于30时更可靠
                    if current_row['RSI'] > 50:
                        should_buy = False
                        print(f"时间{current_timestamp}: RSI={current_row['RSI']:.1f}，过滤买入信号")

                if should_buy:
                    # 计算仓位大小
                    position_size = self.strategy.get_position_size('buy')

                    # 确保有足够资金
                    if position_size <= capital:
                        # 如果有持仓，先平仓再开新仓
                        if current_shares > 0:
                            trade_value = current_shares * current_price
                            trade = self.execute_trade('sell', current_price, capital,
                                                       trade_value, current_timestamp, "换仓")
                            capital += trade_value - trade['cost']
                            trades += 1

                        # 开新仓
                        trade = self.execute_trade('buy', current_price, capital,
                                                   position_size, current_timestamp, "9转买入信号")
                        shares = trade['shares']
                        self.strategy.update_position(shares, current_price, 'buy')
                        capital -= position_size + trade['cost']
                        trades += 1

                        print(f"时间{current_timestamp}: 执行9转买入，仓位={position_size:.0f}，价格={current_price:.2f}")

            # 检查9转卖出信号（如果要做空的话）
            # 这里我们只做多，所以忽略卖出信号或用于平仓
            elif current_row['nine_seq_sell'] == -1 and current_shares > 0:
                # 平仓
                trade_value = current_shares * current_price
                trade = self.execute_trade('sell', current_price, capital,
                                           trade_value, current_timestamp, "9转卖出信号平仓")
                profit = trade_value - trade['cost']
                capital += profit

                # 记录交易结果
                if profit > 0:
                    self.strategy.record_trade_result(profit, 'buy')
                else:
                    self.strategy.record_trade_result(-abs(profit), 'buy')

                self.strategy.clear_position()
                trades += 1

        # 最终平仓
        if self.strategy.current_position > 0:
            final_value = self.strategy.current_position * prices[-1]
            capital += final_value

        return self._calculate_performance_metrics(capital, max_drawdown, trades)

    def _calculate_performance_metrics(self,
                                       final_capital: float,
                                       max_drawdown: float,
                                       total_trades: int) -> Dict[str, float]:
        """
        计算绩效指标
        """
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # 计算夏普比率（基于日收益率）
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 计算胜率
        winning_trades = 0
        total_trade_pairs = 0

        # 找到所有买入-卖出对
        buy_trades = [t for t in self.trade_log if t['type'] == 'buy']
        sell_trades = [t for t in self.trade_log if t['type'] == 'sell']

        for buy_trade in buy_trades:
            # 找到对应的卖出交易
            corresponding_sells = [s for s in sell_trades if s['timestamp'] > buy_trade['timestamp']]
            if corresponding_sells:
                sell_trade = corresponding_sells[0]
                profit_pct = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                if profit_pct > 0:
                    winning_trades += 1
                total_trade_pairs += 1

        win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0

        # 计算平均盈利和平均亏损
        total_profit = 0
        total_loss = 0
        profit_trades = 0
        loss_trades = 0

        for buy_trade in buy_trades:
            corresponding_sells = [s for s in sell_trades if s['timestamp'] > buy_trade['timestamp']]
            if corresponding_sells:
                sell_trade = corresponding_sells[0]
                profit = sell_trade['price'] * buy_trade.get('shares', 1) - buy_trade['price'] * buy_trade.get('shares',
                                                                                                               1)
                if profit > 0:
                    total_profit += profit
                    profit_trades += 1
                else:
                    total_loss += abs(profit)
                    loss_trades += 1

        avg_profit = total_profit / profit_trades if profit_trades > 0 else 0
        avg_loss = total_loss / loss_trades if loss_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': total_return / (len(self.equity_curve) / 252),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_trades': profit_trades,
            'loss_trades': loss_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': self.strategy.consecutive_losses
        }

        self.print_backtest_results(results)
        return results

    def print_backtest_results(self, results: Dict[str, float]):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("9转序列马丁格尔策略回测结果")
        print("=" * 60)
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annual_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"盈利交易: {results['profit_trades']}次，平均盈利: ${results['avg_profit']:.2f}")
        print(f"亏损交易: {results['loss_trades']}次，平均亏损: ${results['avg_loss']:.2f}")
        print(f"盈利因子: {results['profit_factor']:.2f}")
        print(f"最大连续亏损次数: {results['max_consecutive_losses']}")
        print("=" * 60)

    def plot_backtest_results(self, data: pd.DataFrame):
        """绘制回测结果图表"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

        # 子图1：价格和交易信号
        ax1 = plt.subplot(gs[0])
        ax1.plot(data.index, data['Close'], label='价格', linewidth=1, alpha=0.7)

        # 标记9转买入信号
        buy_signals = data[data['nine_seq_buy'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'],
                        color='green', s=100, label='9转买入信号', marker='^', alpha=0.7, zorder=5)

        # 标记9转卖出信号
        sell_signals = data[data['nine_seq_sell'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                        color='red', s=100, label='9转卖出信号', marker='v', alpha=0.7, zorder=5)

        # 标记交易
        buy_times = [t['timestamp'] for t in self.trade_log if t['type'] == 'buy']
        buy_prices = [t['price'] for t in self.trade_log if t['type'] == 'buy']
        sell_times = [t['timestamp'] for t in self.trade_log if t['type'] == 'sell']
        sell_prices = [t['price'] for t in self.trade_log if t['type'] == 'sell']

        ax1.scatter(buy_times, buy_prices, color='blue', s=50, label='买入', marker='>', alpha=0.7)
        ax1.scatter(sell_times, sell_prices, color='orange', s=50, label='卖出', marker='<', alpha=0.7)

        ax1.set_title('9转序列交易信号')
        ax1.set_ylabel('价格')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 子图2：权益曲线
        ax2 = plt.subplot(gs[1])
        equity_timestamps = data.index[:len(self.equity_curve)]
        ax2.plot(equity_timestamps, self.equity_curve, label='权益曲线', color='blue', linewidth=2)
        ax2.fill_between(equity_timestamps, self.equity_curve, self.initial_capital,
                         where=np.array(self.equity_curve) >= self.initial_capital,
                         color='green', alpha=0.3, label='盈利区域')
        ax2.fill_between(equity_timestamps, self.equity_curve, self.initial_capital,
                         where=np.array(self.equity_curve) < self.initial_capital,
                         color='red', alpha=0.3, label='亏损区域')

        ax2.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='初始资金')
        ax2.set_title('权益曲线')
        ax2.set_ylabel('资金 ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 子图3：回撤曲线
        ax3 = plt.subplot(gs[2])
        drawdown_curve = self._calculate_drawdown_curve()
        ax3.fill_between(equity_timestamps[:len(drawdown_curve)], drawdown_curve * 100, 0,
                         color='red', alpha=0.3)
        ax3.plot(equity_timestamps[:len(drawdown_curve)], drawdown_curve * 100,
                 color='red', linewidth=1)
        ax3.set_title('回撤曲线')
        ax3.set_ylabel('回撤 (%)')
        ax3.set_xlabel('时间')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _calculate_drawdown_curve(self) -> np.ndarray:
        """计算回撤曲线"""
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return drawdown

    def plot_trade_analysis(self):
        """绘制交易分析图表"""
        if not self.trade_log:
            print("没有交易记录可分析")
            return

        # 提取交易数据
        trades = []
        buy_trades = [t for t in self.trade_log if t['type'] == 'buy']
        sell_trades = [t for t in self.trade_log if t['type'] == 'sell']

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]

            if sell['timestamp'] > buy['timestamp']:
                holding_period = (sell['timestamp'] - buy['timestamp']).total_seconds() / (24 * 3600)  # 天数
                profit_pct = (sell['price'] - buy['price']) / buy['price']
                profit_abs = sell['price'] * buy.get('shares', 1) - buy['price'] * buy.get('shares', 1)

                trades.append({
                    'entry_time': buy['timestamp'],
                    'exit_time': sell['timestamp'],
                    'entry_price': buy['price'],
                    'exit_price': sell['price'],
                    'holding_period': holding_period,
                    'profit_pct': profit_pct,
                    'profit_abs': profit_abs,
                    'reason': sell.get('reason', '')
                })

        if not trades:
            return

        trades_df = pd.DataFrame(trades)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 盈利分布
        axes[0, 0].hist(trades_df['profit_pct'] * 100, bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('单笔交易收益率分布 (%)')
        axes[0, 0].set_xlabel('收益率 (%)')
        axes[0, 0].set_ylabel('交易次数')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 持仓时间分布
        axes[0, 1].hist(trades_df['holding_period'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('持仓时间分布 (天)')
        axes[0, 1].set_xlabel('持仓时间 (天)')
        axes[0, 1].set_ylabel('交易次数')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 累计盈利曲线
        cumulative_profit = trades_df['profit_abs'].cumsum()
        axes[1, 0].plot(range(len(cumulative_profit)), cumulative_profit, linewidth=2, color='blue')
        axes[1, 0].fill_between(range(len(cumulative_profit)), 0, cumulative_profit,
                                where=cumulative_profit >= 0, color='green', alpha=0.3)
        axes[1, 0].fill_between(range(len(cumulative_profit)), 0, cumulative_profit,
                                where=cumulative_profit < 0, color='red', alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('累计盈利曲线')
        axes[1, 0].set_xlabel('交易序号')
        axes[1, 0].set_ylabel('累计盈利 ($)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 按交易原因分类的盈利
        if 'reason' in trades_df.columns and len(trades_df['reason'].unique()) > 0:
            reason_groups = trades_df.groupby('reason')['profit_pct'].mean() * 100
            axes[1, 1].bar(range(len(reason_groups)), reason_groups.values, alpha=0.7)
            axes[1, 1].set_xticks(range(len(reason_groups)))
            axes[1, 1].set_xticklabels(reason_groups.index, rotation=45, ha='right')
            axes[1, 1].set_title('按交易原因分类的平均收益率')
            axes[1, 1].set_ylabel('平均收益率 (%)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class StrategyOptimizer:
    """
    策略参数优化器
    """

    def __init__(self, data_processor: EfficientMarketDataProcessor):
        self.data_processor = data_processor

    def optimize_nine_sequence_strategy(self,
                                        data: pd.DataFrame,
                                        param_grid: Dict = None) -> Dict:
        """
        优化9转序列策略参数
        """
        if param_grid is None:
            param_grid = {
                'initial_position_size': 10000,
                'max_multiplier': 3,
                'profit_target': 0.2,
                'stop_loss': 0.05,
                'sequence_length': 9
            }

        best_params = None
        best_sharpe = -float('inf')
        results = []

        print("开始优化9转序列策略参数...")

        # 简单网格搜索
        from itertools import product

        keys = param_grid.keys()
        values = param_grid.values()

        for combination in product(*values):
            params = dict(zip(keys, combination))

            try:
                # 重新计算9转信号（因为sequence_length可能变化）
                temp_data = data.copy()
                temp_processor = EfficientMarketDataProcessor()
                temp_data = temp_processor.calculate_nine_sequence_signals(
                    temp_data, sequence_length=params['sequence_length']
                )

                backtester = RealisticBacktester(
                    initial_capital=100000,
                    strategy_type='nine_seq_martingale',
                    **{k: v for k, v in params.items() if k != 'sequence_length'}
                )

                result = backtester.run_nine_sequence_backtest(temp_data)

                if result['sharpe_ratio'] > best_sharpe:
                    best_sharpe = result['sharpe_ratio']
                    best_params = params

                results.append({
                    'params': params,
                    'sharpe': result['sharpe_ratio'],
                    'return': result['total_return'],
                    'drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate']
                })

            except Exception as e:
                print(f"参数组合{params}测试失败: {e}")
                continue

        # 打印最佳参数
        if best_params:
            print(f"\n最佳参数组合: {best_params}")
            print(f"最佳夏普比率: {best_sharpe:.4f}")

        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'all_results': results
        }


# 主程序
if __name__ == "__main__":
    print("启动9转序列马丁格尔策略回测系统...")

    try:
        # 1. 获取数据 - 直接从指定CSV文件
        csv_file_path = "./data_cache/HistoricalData_1765973879677.csv"  # 您的CSV文件名

        # 如果文件不存在，尝试在当前目录查找
        if not os.path.exists(csv_file_path):
            print(f"文件 {csv_file_path} 不存在，尝试查找CSV文件...")
            data_processor = EfficientMarketDataProcessor()
            data = data_processor.fetch_data()
        else:
            print(f"使用CSV文件: {csv_file_path}")
            data_processor = EfficientMarketDataProcessor(csv_file_path)
            data = data_processor.fetch_data(csv_file_path)

        if data is None or len(data) == 0:
            print("数据加载失败，退出程序")
            exit(1)

        print(f"数据加载成功，共{len(data)}条记录")

        # 2. 计算基础指标
        data_with_indicators = data_processor.calculate_basic_indicators(data)

        # 3. 计算9转序列信号
        data_with_signals = data_processor.calculate_nine_sequence_signals(data_with_indicators)

        print(f"数据准备完成，共{len(data_with_signals)}个数据点")

        # 4. 参数优化（可选）
        optimizer = StrategyOptimizer(data_processor)

        # 优化9转序列策略（使用部分数据进行优化以加快速度）
        print("\n开始参数优化...")
        opt_result = optimizer.optimize_nine_sequence_strategy(
            data_with_signals[:min(500, len(data_with_signals))]  # 使用前500个点进行优化
        )

        # 5. 使用优化后的参数运行回测
        print("\n" + "=" * 60)
        print("运行9转序列策略回测...")
        print('=' * 60)

        if opt_result['best_params']:
            params = opt_result['best_params']
            print(f"使用优化参数: {params}")
        else:
            params = {
                'initial_position_size': 1000,
                'max_multiplier': 5,
                'profit_target': 0.02,
                'stop_loss': 0.05,
                'sequence_length': 9
            }
            print(f"使用默认参数: {params}")

        # 重新计算9转信号（确保使用正确的sequence_length）
        final_data = data_processor.calculate_nine_sequence_signals(
            data_with_indicators,
            sequence_length=params['sequence_length']
        )

        backtester = RealisticBacktester(
            initial_capital=100000,
            strategy_type='nine_seq_martingale',
            **{k: v for k, v in params.items() if k != 'sequence_length'}
        )

        results = backtester.run_nine_sequence_backtest(final_data, use_rsi_filter=True)

        # 6. 绘制结果
        backtester.plot_backtest_results(final_data)

        # 7. 交易分析
        backtester.plot_trade_analysis()

        # 8. 策略总结
        print("\n" + "=" * 60)
        print("策略总结")
        print("=" * 60)

        # 计算一些额外统计
        total_signals = final_data['nine_seq_buy'].sum() + abs(final_data['nine_seq_sell'].sum())
        signal_hit_rate = results['profit_trades'] / total_signals if total_signals > 0 else 0

        print(f"总9转信号数量: {total_signals}")
        print(f"信号命中率: {signal_hit_rate:.2%}")

        # 计算平均持仓时间
        if backtester.trade_log:
            buy_trades = [t for t in backtester.trade_log if t['type'] == 'buy']
            sell_trades = [t for t in backtester.trade_log if t['type'] == 'sell']

            holding_periods = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                if sell_trades[i]['timestamp'] > buy_trades[i]['timestamp']:
                    period = (sell_trades[i]['timestamp'] - buy_trades[i]['timestamp']).days
                    holding_periods.append(period)

            if holding_periods:
                avg_holding_period = np.mean(holding_periods)
                print(f"平均持仓时间: {avg_holding_period:.1f}天")

        # 风险评估
        print("\n风险评估:")
        if results['max_drawdown'] > 0.2:
            print("⚠️  高风险: 最大回撤超过20%")
        elif results['max_drawdown'] > 0.1:
            print("⚠️  中等风险: 最大回撤超过10%")
        else:
            print("✅  低风险: 最大回撤低于10%")

        if results['profit_factor'] < 1.5:
            print("⚠️  盈利因子偏低: 建议优化策略参数")
        else:
            print("✅  盈利因子良好")

        if results['win_rate'] < 0.5:
            print("⚠️  胜率低于50%: 建议优化入场时机")
        else:
            print("✅  胜率良好")

        # 建议
        print("\n策略建议:")
        if results['max_consecutive_losses'] >= params['max_multiplier']:
            print(f"1. 当前最大连续亏损{results['max_consecutive_losses']}次已达到限制，建议减小初始仓位")

        if results['avg_loss'] > results['avg_profit'] * 1.5:
            print("2. 平均亏损远大于平均盈利，建议调整止损位置")

        if signal_hit_rate < 0.3:
            print("3. 信号命中率偏低，建议结合其他指标过滤信号")

        # 9. 保存结果
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # 保存交易记录
        if backtester.trade_log:
            trade_df = pd.DataFrame(backtester.trade_log)
            trade_file = os.path.join(output_dir, "trade_records.csv")
            trade_df.to_csv(trade_file, index=False)
            print(f"\n交易记录已保存到: {trade_file}")

        # 保存权益曲线
        equity_df = pd.DataFrame({
            'date': final_data.index[:len(backtester.equity_curve)],
            'equity': backtester.equity_curve
        })
        equity_file = os.path.join(output_dir, "equity_curve.csv")
        equity_df.to_csv(equity_file, index=False)
        print(f"权益曲线已保存到: {equity_file}")

    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback

        traceback.print_exc()
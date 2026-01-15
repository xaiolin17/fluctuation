# -*- coding: utf-8 -*-
"""
高收益时序因子策略回测系统 - 自动化优化版本
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class StrategyConfig:
    """策略配置类"""
    # 基础参数
    initial_capital: float = 100000
    initial_position_size: float = 3000

    # 时序因子参数
    autocorr_lookback: int = 5
    volatility_lookback: int = 20
    volume_confirm_lookback: int = 10

    # 信号阈值参数
    autocorr_threshold: float = 0.05
    volume_ratio_threshold: float = 1.2
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    # 止盈止损参数
    base_profit_target: float = 0.12
    base_stop_loss: float = 0.05
    use_trailing_stop: bool = True
    trailing_stop_activation: float = 0.06
    trailing_stop_distance: float = 0.03

    # 机器学习参数
    use_ml_signal: bool = True
    ml_lookback: int = 60

    # 风险管理
    max_position_ratio: float = 0.25
    daily_loss_limit: float = 0.02
    max_consecutive_losses: int = 3

    # 多因子权重
    autocorr_weight: float = 0.4
    volume_weight: float = 0.3
    momentum_weight: float = 0.3

    # 数据增强
    use_data_augmentation: bool = True
    augmentation_factor: float = 0.2


class DataLoader:
    """数据加载器 - 支持德文列名版本"""

    def __init__(self, csv_file_path: str = None):
        self.csv_file_path = csv_file_path

    def load_and_clean_data(self) -> pd.DataFrame:
        """加载并清理数据 - 专门处理德文列名"""
        try:
            print(f"正在加载CSV文件: {self.csv_file_path}")

            # 读取CSV文件
            df = pd.read_csv(self.csv_file_path)

            print(f"原始数据形状: {df.shape}")
            print(f"原始列名: {list(df.columns)}")

            # 重命名德文列名为英文
            german_to_english = {
                'datum': 'Date',
                'schluß': 'Close',
                'schluss': 'Close',
                'letzter': 'Close',
                'schluß/letzter': 'Close',
                'schluss/letzter': 'Close',
                'eröffnung': 'Open',
                'eröffnungskurs': 'Open',
                'eroeffnung': 'Open',
                'hoch': 'High',
                'tief': 'Low',
                'volumen': 'Volume',
                'umsatz': 'Volume'
            }

            # 创建列映射
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).strip().lower()
                found = False
                for german_key, english_name in german_to_english.items():
                    if german_key in col_lower:
                        column_mapping[col] = english_name
                        print(f"映射列: '{col}' -> '{english_name}'")
                        found = True
                        break
                if not found:
                    print(f"警告: 未识别的列 '{col}'，保持原样")
                    column_mapping[col] = col

            # 应用列名映射
            df = df.rename(columns=column_mapping)

            print(f"\n映射后列名: {list(df.columns)}")

            # 确保有Date列
            if 'Date' not in df.columns:
                date_candidates = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in ['date', 'datum', 'time', 'zeit']):
                        date_candidates.append(col)
                if date_candidates:
                    df = df.rename(columns={date_candidates[0]: 'Date'})
                    print(f"使用 '{date_candidates[0]}' 作为日期列")
                else:
                    df['Date'] = pd.date_range(start='2015-01-01', periods=len(df), freq='D')
                    print("创建日期列")

            # 确保价格列存在
            price_columns = ['Close', 'Open', 'High', 'Low']
            for col in price_columns:
                if col not in df.columns:
                    if col == 'Close':
                        candidates = [c for c in df.columns if any(x in str(c).lower() for x in
                                                                   ['close', 'closing', 'schluss', 'letzter', 'price'])]
                        if candidates:
                            df[col] = df[candidates[0]]
                    elif col == 'Open':
                        candidates = [c for c in df.columns if any(x in str(c).lower() for x in
                                                                   ['open', 'opening', 'eröffnung'])]
                        if candidates:
                            df[col] = df[candidates[0]]
                    elif col == 'High':
                        candidates = [c for c in df.columns if any(x in str(c).lower() for x in
                                                                   ['high', 'hoch', 'maximum'])]
                        if candidates:
                            df[col] = df[candidates[0]]
                    elif col == 'Low':
                        candidates = [c for c in df.columns if any(x in str(c).lower() for x in
                                                                   ['low', 'tief', 'minimum'])]
                        if candidates:
                            df[col] = df[candidates[0]]

            # 如果没有找到所有价格列，尝试推断
            missing_price_cols = [col for col in price_columns if col not in df.columns]
            if missing_price_cols:
                print(f"缺失价格列: {missing_price_cols}")
                if 'Close' in df.columns:
                    for col in missing_price_cols:
                        df[col] = df['Close']
                else:
                    raise ValueError("无法找到价格数据")

            # 确保Date列是datetime类型
            df['Date'] = pd.to_datetime(df['Date'])

            # 处理价格数据 - 确保是数值类型
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

            # 处理Volume列
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            else:
                # 添加模拟成交量数据
                price_changes = df['Close'].diff().abs()
                df['Volume'] = 1000000 * (1 + price_changes * 10)
                df['Volume'] = df['Volume'].fillna(1000000)
                print("添加模拟成交量列")

            # 设置索引并排序
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # 确保数据顺序正确
            if df.index[0] > df.index[-1]:
                df = df.sort_index()
                print("数据已按日期排序")

            # 填充缺失值
            df = df.ffill().bfill()

            # 确保数据质量
            print(f"\n清理后数据形状: {df.shape}")
            print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")

            # 只保留需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_columns if col in df.columns]]

            print(f"最终数据列: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            import traceback
            traceback.print_exc()



class DataAugmentor:
    """数据增强器 - 增加样本多样性"""

    @staticmethod
    def augment_data(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """增强数据 - 增加样本量"""
        if not config.use_data_augmentation or len(data) > 500:
            return data

        print(f"数据增强: 原始数据 {len(data)} 条")

        # 创建增强数据
        augmented_data = data.copy()

        # 方法1: 添加噪声
        noise_factor = config.augmentation_factor
        for col in ['Open', 'High', 'Low', 'Close']:
            noise = np.random.randn(len(data)) * noise_factor * data[col].std()
            augmented_data[col] = data[col] + noise

        # 方法2: 时间序列变换
        n_augment = min(100, int(len(data) * 0.5))
        for i in range(n_augment):
            # 随机选择一段数据
            start_idx = np.random.randint(0, len(data) - 20)
            length = np.random.randint(10, min(30, len(data) - start_idx))

            # 随机缩放
            scale = 1.0 + np.random.randn() * 0.1

            # 创建新的数据片段
            new_segment = data.iloc[start_idx:start_idx + length].copy()
            for col in ['Open', 'High', 'Low', 'Close']:
                new_segment[col] = new_segment[col] * scale

            # 调整日期
            date_offset = pd.DateOffset(days=np.random.randint(-100, 100))
            new_segment.index = new_segment.index + date_offset

            # 添加到增强数据
            augmented_data = pd.concat([augmented_data, new_segment])

        # 重新排序
        augmented_data = augmented_data.sort_index()
        augmented_data = augmented_data[~augmented_data.index.duplicated(keep='first')]

        print(f"数据增强完成: {len(augmented_data)} 条")
        return augmented_data


class AdvancedSignalGenerator:
    """高级信号生成器 - 基于时序因子和统计特征"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全面的技术指标"""
        result = df.copy()

        if 'Close' not in result.columns:
            raise ValueError("数据必须包含Close列")

        print(f"计算技术指标，数据长度: {len(result)}")

        # 基本价格指标
        result['Returns'] = result['Close'].pct_change()
        result['Log_Returns'] = np.log(result['Close'] / result['Close'].shift(1))

        # 移动平均线
        result['SMA_10'] = result['Close'].rolling(window=10, min_periods=1).mean()
        result['SMA_20'] = result['Close'].rolling(window=20, min_periods=1).mean()
        result['SMA_50'] = result['Close'].rolling(window=50, min_periods=1).mean()
        result['EMA_12'] = result['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        result['EMA_26'] = result['Close'].ewm(span=26, adjust=False, min_periods=1).mean()

        # 价格位置指标
        result['Price_Rank_20'] = result['Close'].rolling(window=20).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )

        # MACD
        result['MACD'] = result['EMA_12'] - result['EMA_26']
        result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']

        # RSI
        delta = result['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        result['RSI'] = 100 - (100 / (1 + rs))
        result['RSI'] = result['RSI'].fillna(50)

        # 布林带
        result['BB_Middle'] = result['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = result['Close'].rolling(window=20, min_periods=1).std()
        result['BB_Upper'] = result['BB_Middle'] + 2 * bb_std
        result['BB_Lower'] = result['BB_Middle'] - 2 * bb_std
        result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']

        # 波动率指标
        result['Volatility_10'] = result['Returns'].rolling(window=10, min_periods=1).std() * np.sqrt(252)
        result['Volatility_20'] = result['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # ATR
        high_low = result['High'] - result['Low']
        high_close = np.abs(result['High'] - result['Close'].shift())
        low_close = np.abs(result['Low'] - result['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        result['ATR_Pct'] = result['ATR'] / result['Close']

        # 成交量指标
        if 'Volume' in result.columns:
            result['Volume_SMA_20'] = result['Volume'].rolling(window=20, min_periods=1).mean()
            result['Volume_Ratio'] = result['Volume'] / result['Volume_SMA_20']
            result['Volume_Ratio'] = result['Volume_Ratio'].fillna(1.0)
            result['OBV'] = (np.sign(result['Returns']) * result['Volume']).cumsum()

        # 动量指标
        result['Momentum_5'] = result['Close'].pct_change(5)
        result['Momentum_10'] = result['Close'].pct_change(10)
        result['Momentum_20'] = result['Close'].pct_change(20)

        # 加速度指标
        result['Acceleration_5'] = result['Momentum_5'].diff(3)

        print("技术指标计算完成")
        return result

    def calculate_timing_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算时序因子 - 核心策略逻辑
        """
        result = df.copy()

        # 确保有收益率数据
        if 'Returns' not in result.columns:
            result['Returns'] = result['Close'].pct_change()

        # 1. 收益率自相关因子 (rtnfoc)
        result['Autocorr_Factor'] = 0.0

        for i in range(self.config.autocorr_lookback, len(result)):
            window_returns = result['Returns'].iloc[i - self.config.autocorr_lookback:i].values
            if len(window_returns) > 1:
                # 计算一阶自相关性
                autocorr = np.corrcoef(window_returns[:-1], window_returns[1:])[0, 1]
                result.loc[result.index[i], 'Autocorr_Factor'] = autocorr if not np.isnan(autocorr) else 0.0

        # 2. 成交量确认因子
        if 'Volume_Ratio' in result.columns:
            result['Volume_Confirm_Factor'] = result['Volume_Ratio'].rolling(
                window=self.config.volume_confirm_lookback, min_periods=1
            ).mean()
        else:
            result['Volume_Confirm_Factor'] = 1.0

        # 3. 动量-反转切换因子
        result['Momentum_Reversal_Factor'] = 0.0
        result['Abs_Autocorr'] = result['Autocorr_Factor'].abs()

        # 4. 波动率调整因子
        if 'Volatility_20' in result.columns:
            vol_scaled = result['Volatility_20'] / result['Volatility_20'].rolling(window=100, min_periods=1).mean()
            result['Volatility_Adjustment'] = 1.0 / (1.0 + vol_scaled)
        else:
            result['Volatility_Adjustment'] = 1.0

        # 5. 价格位置因子
        if 'Price_Rank_20' in result.columns:
            result['Price_Position_Factor'] = result['Price_Rank_20']
        else:
            result['Price_Position_Factor'] = 0.5

        # 6. RSI状态因子
        if 'RSI' in result.columns:
            result['RSI_State'] = 0
            result.loc[result['RSI'] < self.config.rsi_oversold, 'RSI_State'] = 1
            result.loc[result['RSI'] > self.config.rsi_overbought, 'RSI_State'] = -1
        else:
            result['RSI_State'] = 0

        print("时序因子计算完成")
        return result

    def generate_timing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于时序因子生成择时信号
        """
        result = df.copy()
        result['timing_signal'] = 0
        result['signal_strength'] = 0.0
        result['signal_components'] = ""

        # 计算综合信号强度
        min_lookback = max(self.config.autocorr_lookback, 20, self.config.ml_lookback)

        for i in range(min_lookback, len(result)):
            current = result.iloc[i]

            # 初始化信号强度
            signal_strength = 0.0
            components = []

            # 1. 自相关因子信号
            autocorr_signal = 0
            if current['Autocorr_Factor'] > self.config.autocorr_threshold:
                autocorr_signal = 1
                signal_strength += self.config.autocorr_weight * min(abs(current['Autocorr_Factor']) * 3, 1.0)
                components.append(f"AC+{current['Autocorr_Factor']:.3f}")
            elif current['Autocorr_Factor'] < -self.config.autocorr_threshold:
                autocorr_signal = -1
                signal_strength += self.config.autocorr_weight * min(abs(current['Autocorr_Factor']) * 2, 0.8)
                components.append(f"AC-{abs(current['Autocorr_Factor']):.3f}")

            # 2. 成交量确认
            volume_signal = 0
            if 'Volume_Confirm_Factor' in current and current[
                'Volume_Confirm_Factor'] > self.config.volume_ratio_threshold:
                volume_signal = 1
                signal_strength += self.config.volume_weight * 0.8
                components.append(f"V+{current['Volume_Confirm_Factor']:.2f}")
            elif 'Volume_Confirm_Factor' in current and current['Volume_Confirm_Factor'] < 0.8:
                volume_signal = -1
                signal_strength -= self.config.volume_weight * 0.3
                components.append(f"V-{current['Volume_Confirm_Factor']:.2f}")

            # 3. 动量因子
            momentum_signal = 0
            if 'Momentum_10' in current:
                if current['Momentum_10'] > 0.02:
                    momentum_signal = 1
                    signal_strength += self.config.momentum_weight * min(current['Momentum_10'] * 10, 0.8)
                    components.append(f"M+{current['Momentum_10']:.3f}")
                elif current['Momentum_10'] < -0.02:
                    momentum_signal = -1
                    signal_strength += self.config.momentum_weight * min(abs(current['Momentum_10']) * 8, 0.6)
                    components.append(f"M-{abs(current['Momentum_10']):.3f}")

            # 4. RSI状态调整
            rsi_adjustment = 1.0
            if current.get('RSI_State', 0) == 1:
                rsi_adjustment = 1.3
                components.append("RSI_超卖")
            elif current.get('RSI_State', 0) == -1:
                rsi_adjustment = 0.7
                components.append("RSI_超买")

            # 5. 价格位置调整
            position_adjustment = 1.0
            if 'Price_Position_Factor' in current:
                if current['Price_Position_Factor'] < 0.3:
                    position_adjustment = 1.2
                    components.append("价格低位")
                elif current['Price_Position_Factor'] > 0.7:
                    position_adjustment = 0.8
                    components.append("价格高位")

            # 6. 波动率调整
            volatility_adjustment = current.get('Volatility_Adjustment', 1.0)
            signal_strength *= volatility_adjustment * rsi_adjustment * position_adjustment

            # 确定最终信号方向
            final_signal = 0
            if signal_strength > 0.3:
                positive_signals = sum([1 for s in [autocorr_signal, volume_signal, momentum_signal] if s > 0])
                negative_signals = sum([1 for s in [autocorr_signal, volume_signal, momentum_signal] if s < 0])

                if positive_signals > negative_signals:
                    final_signal = 1
                elif negative_signals > positive_signals:
                    final_signal = -1
                else:
                    final_signal = autocorr_signal

            # 保存结果
            result.loc[result.index[i], 'timing_signal'] = final_signal
            result.loc[result.index[i], 'signal_strength'] = min(signal_strength, 1.0)
            result.loc[result.index[i], 'signal_components'] = "|".join(components) if components else "无"

        # 信号统计
        buy_signals = (result['timing_signal'] == 1).sum()
        sell_signals = (result['timing_signal'] == -1).sum()

        print(f"\n时序因子信号生成完成:")
        print(f"  买入信号: {buy_signals} 个")
        print(f"  卖出信号: {sell_signals} 个")

        if buy_signals > 0:
            avg_strength = result[result['timing_signal'] == 1]['signal_strength'].mean()
            print(f"  平均买入信号强度: {avg_strength:.3f}")

        return result


class AutoOptimizer:
    """自动优化器 - 自动寻找最佳参数"""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def optimize_configuration(self) -> StrategyConfig:
        """自动优化策略配置"""
        print("\n" + "=" * 60)
        print("开始自动参数优化")
        print("=" * 60)

        # 分析数据特征
        data_features = self.analyze_data_features()

        # 根据数据特征选择初始配置
        base_config = self.select_initial_config(data_features)

        # 测试多个配置
        tested_configs = self.test_configurations(base_config)

        # 选择最佳配置
        best_config = self.select_best_config(tested_configs)

        print("\n" + "=" * 60)
        print("自动优化完成")
        print("=" * 60)

        return best_config

    def analyze_data_features(self) -> Dict[str, Any]:
        """分析数据特征"""
        features = {}

        if len(self.data) > 0:
            # 计算基本统计
            returns = self.data['Close'].pct_change().dropna()

            features['data_length'] = len(self.data)
            features['avg_return'] = returns.mean()
            features['volatility'] = returns.std()
            features['skewness'] = returns.skew()
            features['kurtosis'] = returns.kurtosis()

            # 判断市场状态
            features['trend_strength'] = self.calculate_trend_strength()
            features['volatility_regime'] = self.determine_volatility_regime(features['volatility'])

            print(f"数据特征分析:")
            print(f"  数据长度: {features['data_length']}")
            print(f"  平均收益率: {features['avg_return']:.4%}")
            print(f"  波动率: {features['volatility']:.4%}")
            print(f"  偏度: {features['skewness']:.3f}")
            print(f"  峰度: {features['kurtosis']:.3f}")
            print(f"  趋势强度: {features['trend_strength']:.3f}")
            print(f"  波动率状态: {features['volatility_regime']}")

        return features

    def calculate_trend_strength(self) -> float:
        """计算趋势强度"""
        if len(self.data) < 20:
            return 0.5

        prices = self.data['Close'].values
        if len(prices) >= 50:
            # 使用线性回归判断趋势
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            trend_strength = abs(slope) / prices.mean()
            return min(trend_strength * 100, 1.0)
        return 0.5

    def determine_volatility_regime(self, volatility: float) -> str:
        """确定波动率状态"""
        if volatility < 0.01:
            return "低波动"
        elif volatility < 0.02:
            return "中波动"
        else:
            return "高波动"

    def select_initial_config(self, features: Dict[str, Any]) -> StrategyConfig:
        """根据数据特征选择初始配置"""
        config = StrategyConfig()

        # 根据数据长度调整参数
        if features['data_length'] < 100:
            config.autocorr_lookback = 3
            config.volatility_lookback = 10
            config.ml_lookback = 30
            config.use_ml_signal = False
        elif features['data_length'] < 500:
            config.autocorr_lookback = 5
            config.ml_lookback = 50

        # 根据波动率调整参数
        if features['volatility_regime'] == "高波动":
            config.base_profit_target = 0.15
            config.base_stop_loss = 0.06
            config.trailing_stop_distance = 0.04
            config.autocorr_threshold = 0.06
        elif features['volatility_regime'] == "低波动":
            config.base_profit_target = 0.08
            config.base_stop_loss = 0.03
            config.trailing_stop_distance = 0.02
            config.autocorr_threshold = 0.03

        # 根据趋势强度调整参数
        if features['trend_strength'] > 0.7:
            # 强趋势市场
            config.autocorr_weight = 0.5
            config.momentum_weight = 0.4
            config.volume_weight = 0.1
            config.use_trailing_stop = True
        elif features['trend_strength'] < 0.3:
            # 震荡市场
            config.autocorr_weight = 0.3
            config.momentum_weight = 0.2
            config.volume_weight = 0.5
            config.use_trailing_stop = False

        return config

    def test_configurations(self, base_config: StrategyConfig) -> List[Tuple[StrategyConfig, Dict[str, Any]]]:
        """测试多个配置"""
        tested_configs = []

        # 创建多个配置变体
        config_variants = self.create_config_variants(base_config)

        # 限制测试数量
        max_tests = min(5, len(config_variants))
        config_variants = config_variants[:max_tests]

        print(f"\n测试 {len(config_variants)} 个配置变体")

        for i, config in enumerate(config_variants):
            print(f"\n测试配置 {i + 1}/{len(config_variants)}")

            try:
                # 生成信号
                signal_generator = AdvancedSignalGenerator(config)
                data_with_indicators = signal_generator.calculate_technical_indicators(self.data)
                data_with_factors = signal_generator.calculate_timing_factors(data_with_indicators)
                data_with_signals = signal_generator.generate_timing_signals(data_with_factors)

                # 运行回测
                backtester = HighReturnBacktester(config)
                result = backtester.run_backtest(data_with_signals)

                # 计算综合评分
                score = self.calculate_config_score(result, config)
                result['score'] = score

                tested_configs.append((config, result))

                print(f"  评分: {score:.2f}")
                print(f"  年化收益: {result['annual_return']:.2%}")
                print(f"  最大回撤: {result['max_drawdown']:.2%}")

            except Exception as e:
                print(f"  配置测试失败: {e}")
                continue

        return tested_configs

    def create_config_variants(self, base_config: StrategyConfig) -> List[StrategyConfig]:
        """创建配置变体"""
        variants = []

        # 基础配置
        variants.append(base_config)

        # 变体1: 保守策略
        config1 = StrategyConfig(
            initial_position_size=base_config.initial_position_size * 0.8,
            autocorr_lookback=max(3, base_config.autocorr_lookback - 2),
            autocorr_threshold=base_config.autocorr_threshold * 1.2,
            base_profit_target=base_config.base_profit_target * 0.8,
            base_stop_loss=base_config.base_stop_loss * 0.8,
            use_trailing_stop=True,
            autocorr_weight=0.3,
            volume_weight=0.4,
            momentum_weight=0.3,
            use_ml_signal=base_config.use_ml_signal
        )
        variants.append(config1)

        # 变体2: 激进策略
        config2 = StrategyConfig(
            initial_position_size=base_config.initial_position_size * 1.2,
            autocorr_lookback=min(10, base_config.autocorr_lookback + 3),
            autocorr_threshold=base_config.autocorr_threshold * 0.8,
            base_profit_target=base_config.base_profit_target * 1.2,
            base_stop_loss=base_config.base_stop_loss * 1.2,
            use_trailing_stop=True,
            autocorr_weight=0.5,
            volume_weight=0.2,
            momentum_weight=0.3,
            use_ml_signal=base_config.use_ml_signal
        )
        variants.append(config2)

        # 变体3: 动量策略
        config3 = StrategyConfig(
            initial_position_size=base_config.initial_position_size,
            autocorr_lookback=base_config.autocorr_lookback,
            autocorr_threshold=base_config.autocorr_threshold,
            base_profit_target=base_config.base_profit_target * 1.1,
            base_stop_loss=base_config.base_stop_loss * 0.9,
            use_trailing_stop=True,
            autocorr_weight=0.2,
            volume_weight=0.2,
            momentum_weight=0.6,
            use_ml_signal=base_config.use_ml_signal
        )
        variants.append(config3)

        return variants

    def calculate_config_score(self, result: Dict[str, Any], config: StrategyConfig) -> float:
        """计算配置评分"""
        score = 0

        # 年化收益权重最高
        if result['annual_return'] > 0:
            score += result['annual_return'] * 100

        # 夏普比率
        if result['sharpe_ratio'] > 0:
            score += result['sharpe_ratio'] * 10

        # 盈利因子
        if result['profit_factor'] > 1:
            score += min(result['profit_factor'] - 1, 3) * 5

        # 惩罚大回撤
        if result['max_drawdown'] > 0:
            score -= result['max_drawdown'] * 200

        # 鼓励合理交易次数
        if result['total_trades'] > 5:
            score += min(result['total_trades'] * 0.1, 5)

        return score

    def select_best_config(self, tested_configs: List[Tuple[StrategyConfig, Dict[str, Any]]]) -> StrategyConfig:
        """选择最佳配置"""
        if not tested_configs:
            return StrategyConfig()

        # 按评分排序
        tested_configs.sort(key=lambda x: x[1]['score'], reverse=True)

        best_config, best_result = tested_configs[0]

        print(f"\n最佳配置选择:")
        print(f"  评分: {best_result['score']:.2f}")
        print(f"  年化收益: {best_result['annual_return']:.2%}")
        print(f"  最大回撤: {best_result['max_drawdown']:.2%}")
        print(f"  夏普比率: {best_result['sharpe_ratio']:.2f}")

        return best_config


class HighReturnBacktester:
    """高收益回测系统"""

    def __init__(self, config: StrategyConfig):
        self.config = config

        # 交易记录
        self.trade_log = []
        self.equity_curve = []
        self.signals_log = []

        # 策略状态
        self.current_position = 0
        self.entry_price = 0
        self.entry_date = None
        self.consecutive_losses = 0
        self.trailing_stop = 0

        # 性能统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_equity = 0

    def calculate_position_size(self, current_capital: float, signal_strength: float = 1.0) -> float:
        """计算仓位大小 - 考虑信号强度和风险控制"""

        base_size = self.config.initial_position_size

        # 信号强度调整 (0.5-1.5倍)
        strength_multiplier = 0.5 + signal_strength

        # 连续亏损调整
        martingale_multiplier = 1.0
        if self.consecutive_losses > 0:
            martingale_multiplier = min(1.2 ** self.consecutive_losses, 2.0)

        # 计算仓位
        position = base_size * strength_multiplier * martingale_multiplier

        # 仓位限制
        max_position = current_capital * self.config.max_position_ratio
        position = min(position, max_position)

        # 最小仓位
        min_position = current_capital * 0.01
        position = max(position, min_position)

        return position

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行回测"""
        print("\n" + "=" * 60)
        print("开始高收益时序因子策略回测")
        print("=" * 60)

        capital = self.config.initial_capital
        self.equity_curve = [capital]
        self.peak_equity = capital

        dates = data.index
        prices = data['Close'].values

        print(f"回测数据长度: {len(data)}")
        print(f"初始资金: ${capital:,.2f}")

        for i in range(1, len(data)):
            current_date = dates[i]
            current_price = prices[i]

            # 获取当前行数据
            if i < len(data):
                current_row = data.iloc[i]
            else:
                continue

            # 计算当前权益
            position_value = self.current_position * current_price if self.current_position > 0 else 0
            current_equity = capital + position_value
            self.equity_curve.append(current_equity)

            # 更新最大回撤
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)

            # 检查每日亏损限制
            if len(self.equity_curve) > 1 and self.equity_curve[-2] > 0:
                daily_return = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
                if daily_return < -self.config.daily_loss_limit and self.current_position > 0:
                    # 强制平仓
                    trade_value = self.current_position * current_price
                    capital += trade_value

                    self.trade_log.append({
                        'date': current_date,
                        'type': 'forced_exit',
                        'price': current_price,
                        'shares': self.current_position,
                        'profit': trade_value - (self.current_position * self.entry_price),
                        'reason': f"日内亏损超过{self.config.daily_loss_limit:.1%}限制",
                        'equity': capital
                    })

                    self.current_position = 0
                    self.entry_price = 0
                    self.entry_date = None
                    self.trailing_stop = 0

            # 如果有持仓，检查退出条件
            if self.current_position > 0 and self.entry_price > 0:
                profit_pct = (current_price - self.entry_price) / self.entry_price

                exit_trade = False
                exit_reason = ""

                # 检查移动止盈
                if self.config.use_trailing_stop and self.trailing_stop > 0:
                    if current_price <= self.trailing_stop:
                        exit_trade = True
                        exit_reason = f"移动止盈 @ {current_price:.2f}"

                # 检查止盈
                elif profit_pct >= self.config.base_profit_target:
                    exit_trade = True
                    exit_reason = f"固定止盈 @ {current_price:.2f} (+{profit_pct:.1%})"

                # 检查止损
                elif profit_pct <= -self.config.base_stop_loss:
                    exit_trade = True
                    exit_reason = f"固定止损 @ {current_price:.2f} ({profit_pct:.1%})"

                # 检查卖出信号
                elif current_row.get('timing_signal', 0) == -1:
                    exit_trade = True
                    exit_reason = f"时序因子卖出信号"

                # 更新移动止盈
                if self.config.use_trailing_stop and profit_pct >= self.config.trailing_stop_activation:
                    new_stop = current_price * (1 - self.config.trailing_stop_distance)
                    if new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop

                # 执行退出
                if exit_trade:
                    # 计算交易结果
                    trade_value = self.current_position * current_price
                    entry_value = self.current_position * self.entry_price
                    profit = trade_value - entry_value

                    # 更新资金
                    capital += trade_value

                    # 记录交易
                    self.trade_log.append({
                        'date': current_date,
                        'type': 'exit',
                        'price': current_price,
                        'shares': self.current_position,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'reason': exit_reason,
                        'equity': current_equity,
                        'consecutive_losses': self.consecutive_losses
                    })

                    # 更新策略状态
                    if profit > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1

                    self.total_trades += 1
                    self.current_position = 0
                    self.entry_price = 0
                    self.entry_date = None
                    self.trailing_stop = 0

            # 检查买入信号
            if current_row.get('timing_signal', 0) == 1 and capital > self.config.initial_position_size * 2:
                # 确保没有持仓或允许开新仓
                if self.current_position == 0:
                    # 计算信号强度
                    signal_strength = current_row.get('signal_strength', 0.5)

                    # 计算仓位
                    position_size = self.calculate_position_size(capital, signal_strength)

                    # 确保有足够资金
                    if position_size <= capital:
                        # 计算股票数量
                        shares = position_size / current_price

                        # 更新资金和持仓
                        capital -= position_size
                        self.current_position = shares
                        self.entry_price = current_price
                        self.entry_date = current_date

                        # 设置初始移动止盈
                        if self.config.use_trailing_stop:
                            self.trailing_stop = current_price * (1 - self.config.trailing_stop_distance)

                        # 记录交易
                        self.trade_log.append({
                            'date': current_date,
                            'type': 'entry',
                            'price': current_price,
                            'shares': shares,
                            'position_size': position_size,
                            'signal_strength': signal_strength,
                            'signal_components': current_row.get('signal_components', ''),
                            'equity': capital + shares * current_price
                        })

        # 最终平仓
        if self.current_position > 0:
            final_value = self.current_position * prices[-1]
            capital += final_value

            profit = final_value - (self.current_position * self.entry_price)
            profit_pct = profit / (
                        self.current_position * self.entry_price) if self.current_position * self.entry_price > 0 else 0

            self.trade_log.append({
                'date': dates[-1],
                'type': 'final_exit',
                'price': prices[-1],
                'shares': self.current_position,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': "最终平仓",
                'equity': capital
            })

            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            self.total_trades += 1

        print(f"\n回测完成，总交易次数: {self.total_trades}")
        print(f"最终资金: ${capital:,.2f}")

        # 计算绩效指标
        results = self.calculate_performance_metrics(capital)

        return results

    def calculate_performance_metrics(self, final_capital: float) -> Dict[str, Any]:
        """计算绩效指标"""

        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital

        # 计算年化收益率
        n_days = len(self.equity_curve)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        # 计算收益率序列
        equity_array = np.array(self.equity_curve)
        if len(equity_array) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            annual_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        else:
            annual_volatility = 0

        # 夏普比率
        risk_free_rate = 0.02
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0

        # 交易统计
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # 计算盈利因子
        total_profit = 0
        total_loss = 0

        for trade in self.trade_log:
            if trade['type'] in ['exit', 'forced_exit', 'switch', 'final_exit']:
                profit = trade.get('profit', 0)
                if profit > 0:
                    total_profit += profit
                else:
                    total_loss += abs(profit)

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 计算平均盈利/平均亏损
        winning_profits = []
        losing_profits = []

        for trade in self.trade_log:
            if trade['type'] in ['exit', 'forced_exit', 'switch', 'final_exit']:
                profit = trade.get('profit', 0)
                if profit > 0:
                    winning_profits.append(profit)
                else:
                    losing_profits.append(abs(profit))

        avg_win = np.mean(winning_profits) if winning_profits else 0
        avg_loss = np.mean(losing_profits) if losing_profits else 0
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # 计算最大连续亏损次数
        max_consecutive_losses = 0
        current_consecutive = 0

        for trade in self.trade_log:
            if trade['type'] in ['exit', 'forced_exit', 'switch', 'final_exit']:
                profit = trade.get('profit', 0)
                if profit < 0:
                    current_consecutive += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
                else:
                    current_consecutive = 0

        results = {
            'initial_capital': self.config.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_ratio': profit_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'consecutive_losses': self.consecutive_losses,
            'equity_curve_length': len(self.equity_curve)
        }

        self.print_results(results)
        return results

    def print_results(self, results: Dict[str, Any]):
        """打印结果"""
        print("\n" + "=" * 60)
        print("回测结果总结")
        print("=" * 60)
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annual_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"盈利交易: {results['winning_trades']}次")
        print(f"亏损交易: {results['losing_trades']}次")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"盈利因子: {results['profit_factor']:.2f}")
        print(f"平均盈利: ${results['avg_win']:.2f}")
        print(f"平均亏损: ${results['avg_loss']:.2f}")
        print(f"盈亏比: {results['profit_ratio']:.2f}")
        print(f"最大连续亏损次数: {results['max_consecutive_losses']}")
        print(f"权益曲线长度: {results['equity_curve_length']}")
        print("=" * 60)

    def plot_results(self, data: pd.DataFrame):
        """绘制结果图表"""
        if len(self.equity_curve) == 0:
            print("没有权益数据可绘制")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1])

        # 1. 价格和信号图
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(data.index, data['Close'], label='价格', linewidth=1, color='black', alpha=0.7)

        # 标记买入信号
        buy_signals = data[data['timing_signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'],
                        color='green', s=80, label='买入信号', marker='^', alpha=0.8, zorder=5)

        # 标记卖出信号
        sell_signals = data[data['timing_signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                        color='red', s=80, label='卖出信号', marker='v', alpha=0.8, zorder=5)

        # 标记交易
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            entry_trades = trade_df[trade_df['type'] == 'entry']
            exit_trades = trade_df[trade_df['type'].isin(['exit', 'forced_exit', 'final_exit'])]

            if not entry_trades.empty:
                ax1.scatter(entry_trades['date'], entry_trades['price'],
                            color='blue', s=100, label='买入', marker='>', alpha=0.9, zorder=6)

            if not exit_trades.empty:
                ax1.scatter(exit_trades['date'], exit_trades['price'],
                            color='orange', s=100, label='卖出', marker='<', alpha=0.9, zorder=6)

        ax1.set_title('价格走势与交易信号', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 权益曲线
        ax2 = plt.subplot(gs[1, :])
        equity_dates = data.index[:len(self.equity_curve)]
        ax2.plot(equity_dates, self.equity_curve, label='权益曲线',
                 color='blue', linewidth=2)

        ax2.axhline(y=self.config.initial_capital, color='black',
                    linestyle='--', alpha=0.5, label='初始资金')

        # 标记关键点
        if len(self.equity_curve) > 0:
            max_equity = max(self.equity_curve)
            min_equity = min(self.equity_curve)
            max_idx = self.equity_curve.index(max_equity)
            min_idx = self.equity_curve.index(min_equity)

            if max_idx < len(equity_dates) and min_idx < len(equity_dates):
                ax2.scatter([equity_dates[max_idx]], [max_equity],
                            color='green', s=100, marker='o', label=f'最高: ${max_equity:,.0f}')
                ax2.scatter([equity_dates[min_idx]], [min_equity],
                            color='red', s=100, marker='o', label=f'最低: ${min_equity:,.0f}')

        ax2.set_title('权益曲线', fontsize=14, fontweight='bold')
        ax2.set_ylabel('资金 ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. 回撤曲线
        ax3 = plt.subplot(gs[2, 0])
        if len(self.equity_curve) > 0:
            peak = np.maximum.accumulate(np.array(self.equity_curve))
            drawdown = (peak - self.equity_curve) / peak

            ax3.fill_between(equity_dates[:len(drawdown)], 0, drawdown * 100,
                             color='red', alpha=0.3)
            ax3.plot(equity_dates[:len(drawdown)], drawdown * 100,
                     color='red', linewidth=1)

            ax3.set_title('回撤曲线', fontsize=14, fontweight='bold')
            ax3.set_ylabel('回撤 (%)')
            ax3.grid(True, alpha=0.3)

        # 4. 信号强度分布
        ax4 = plt.subplot(gs[2, 1])
        if 'signal_strength' in data.columns:
            signal_strengths = data[data['timing_signal'] == 1]['signal_strength']
            if len(signal_strengths) > 0:
                ax4.hist(signal_strengths, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax4.axvline(x=signal_strengths.mean(), color='red', linestyle='--',
                            label=f'平均强度: {signal_strengths.mean():.3f}')
                ax4.set_title('买入信号强度分布', fontsize=14, fontweight='bold')
                ax4.set_xlabel('信号强度')
                ax4.set_ylabel('信号数量')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        # 5. 自相关因子与价格
        ax5 = plt.subplot(gs[3, 0])
        if 'Autocorr_Factor' in data.columns:
            # 双坐标轴
            ax5_twin = ax5.twinx()

            # 价格
            ax5.plot(data.index, data['Close'], color='black', alpha=0.3, linewidth=0.5)
            ax5.set_ylabel('价格', color='black')
            ax5.tick_params(axis='y', labelcolor='black')

            # 自相关因子
            ax5_twin.plot(data.index, data['Autocorr_Factor'],
                          color='blue', alpha=0.7, linewidth=1)
            ax5_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax5_twin.axhline(y=self.config.autocorr_threshold, color='green',
                             linestyle=':', alpha=0.5, label=f'阈值({self.config.autocorr_threshold})')
            ax5_twin.axhline(y=-self.config.autocorr_threshold, color='red',
                             linestyle=':', alpha=0.5)
            ax5_twin.set_ylabel('自相关因子', color='blue')
            ax5_twin.tick_params(axis='y', labelcolor='blue')

            ax5.set_title('价格与自相关因子', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)

        # 6. 交易盈利分布
        ax6 = plt.subplot(gs[3, 1])
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            exit_trades = trade_df[trade_df['type'].isin(['exit', 'forced_exit', 'final_exit', 'switch'])]

            if not exit_trades.empty and 'profit' in exit_trades.columns:
                profits = exit_trades['profit'].values

                if len(profits) > 0:
                    ax6.hist(profits, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='盈亏平衡线')
                    ax6.axvline(x=np.mean(profits), color='green', linestyle='--',
                                label=f'平均: ${np.mean(profits):.2f}')
                    ax6.set_title('单笔交易盈利分布', fontsize=14, fontweight='bold')
                    ax6.set_xlabel('盈利 ($)')
                    ax6.set_ylabel('交易次数')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """主函数 - 完全自动化版本"""
    print("高收益时序因子策略回测系统 - 自动化优化版本")
    print("=" * 60)

    # 加载数据
    csv_file = "./data_cache/HistoricalData_1765973879677.csv"
    if not os.path.exists(csv_file):
        csv_file = "HistoricalData_1765973879677.csv"

    # 如果没有数据文件，使用用户提供的数据
    if not os.path.exists(csv_file):
        raise "未找到数据文件，使用用户提供的数据..."


    data_loader = DataLoader(csv_file)
    data = data_loader.load_and_clean_data()

    print(f"\n数据基本信息:")
    print(f"  数据条数: {len(data)}")
    if len(data) > 0:
        print(f"  时间范围: {data.index[0].date()} 到 {data.index[-1].date()}")
        if 'Close' in data.columns:
            print(f"  价格范围: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"  平均价格: ${data['Close'].mean():.2f}")

    # 数据增强（如果数据量小）
    config = StrategyConfig()
    if len(data) < 100:
        augmentor = DataAugmentor()
        data = augmentor.augment_data(data, config)

    print(f"\n最终数据条数: {len(data)}")

    # 自动优化参数
    print("\n开始自动参数优化...")
    optimizer = AutoOptimizer(data)
    best_config = optimizer.optimize_configuration()

    print(f"\n使用优化后的配置:")
    print(f"  初始仓位: ${best_config.initial_position_size:,.0f}")
    print(f"  自相关回溯期: {best_config.autocorr_lookback}")
    print(f"  自相关阈值: {best_config.autocorr_threshold:.3f}")
    print(f"  止盈: {best_config.base_profit_target:.1%}")
    print(f"  止损: {best_config.base_stop_loss:.1%}")
    print(f"  移动止盈: {best_config.use_trailing_stop}")
    print(f"  因子权重: 自相关{best_config.autocorr_weight:.1f}, "
          f"成交量{best_config.volume_weight:.1f}, "
          f"动量{best_config.momentum_weight:.1f}")
    print(f"  使用ML信号: {best_config.use_ml_signal}")

    # 生成信号
    print("\n生成技术指标和时序因子...")
    signal_generator = AdvancedSignalGenerator(best_config)
    data_with_indicators = signal_generator.calculate_technical_indicators(data)
    data_with_factors = signal_generator.calculate_timing_factors(data_with_indicators)

    print("生成时序因子信号...")
    data_with_signals = signal_generator.generate_timing_signals(data_with_factors)

    # 运行回测
    print("\n运行回测...")
    final_backtester = HighReturnBacktester(best_config)
    final_results = final_backtester.run_backtest(data_with_signals)

    # 绘制图表
    print("\n生成图表...")
    final_backtester.plot_results(data_with_signals)

    # 保存结果
    output_dir = "./timing_factor_output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存交易记录
    if final_backtester.trade_log:
        trade_df = pd.DataFrame(final_backtester.trade_log)
        trade_file = os.path.join(output_dir, "trades.csv")
        trade_df.to_csv(trade_file, index=False)
        print(f"\n交易记录保存至: {trade_file}")

    # 保存权益曲线
    if final_backtester.equity_curve:
        equity_df = pd.DataFrame({
            'date': data.index[:len(final_backtester.equity_curve)],
            'equity': final_backtester.equity_curve
        })
        equity_file = os.path.join(output_dir, "equity.csv")
        equity_df.to_csv(equity_file, index=False)
        print(f"权益曲线保存至: {equity_file}")

    # 保存信号数据
    signal_cols = ['timing_signal', 'signal_strength', 'signal_components',
                   'Autocorr_Factor', 'Volume_Confirm_Factor', 'RSI']
    signal_cols = [col for col in signal_cols if col in data_with_signals.columns]

    if signal_cols:
        signal_df = data_with_signals[signal_cols].copy()
        signal_df['Close'] = data_with_signals['Close']
        signal_file = os.path.join(output_dir, "signals.csv")
        signal_df.to_csv(signal_file)
        print(f"信号数据保存至: {signal_file}")

    # 详细分析
    if final_backtester.trade_log:
        trade_df = pd.DataFrame(final_backtester.trade_log)
        exit_trades = trade_df[trade_df['type'].isin(['exit', 'forced_exit', 'final_exit', 'switch'])]

        if not exit_trades.empty:
            print("\n" + "=" * 60)
            print("交易详情分析")
            print("=" * 60)

            # 按退出原因统计
            reason_stats = exit_trades.groupby('reason').agg({
                'profit': ['count', 'sum', 'mean', 'std']
            }).round(2)

            print("\n按退出原因统计:")
            print(reason_stats)

            # 最佳和最差交易
            if 'profit' in exit_trades.columns and len(exit_trades) > 0:
                best_trade = exit_trades.loc[exit_trades['profit'].idxmax()]
                worst_trade = exit_trades.loc[exit_trades['profit'].idxmin()]

                print(f"\n最佳交易:")
                print(f"  盈利: ${best_trade['profit']:.2f}")
                print(f"  原因: {best_trade['reason']}")
                print(f"  日期: {best_trade['date'].date()}")

                print(f"\n最差交易:")
                print(f"  亏损: ${abs(worst_trade['profit']):.2f}")
                print(f"  原因: {worst_trade['reason']}")
                print(f"  日期: {worst_trade['date'].date()}")

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union
import datetime
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')


@dataclass
class PatternConfig:
    """形态检测配置"""
    # 通用参数
    min_data_points: int = 100
    min_swing_percent: float = 0.08
    min_distance_points: int = 5

    # 头肩形态参数
    hs_min_time_gap: int = 10
    hs_min_height_pct: float = 0.2
    hs_shoulder_diff_pct: float = 1.0
    hs_confidence_threshold: float = 0.6

    # 双重顶/底参数
    double_min_distance: int = 8
    double_price_tolerance: float = 0.1
    double_min_height_pct: float = 0.15

    # 三重顶/底参数
    triple_min_distance: int = 6
    triple_price_tolerance: float = 0.15
    triple_min_height_pct: float = 0.2

    # 楔形参数
    wedge_min_points: int = 5
    wedge_slope_ratio: float = 1.2
    wedge_min_confidence: float = 0.5


class BasePatternDetector:
    """模式检测器基类"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = None
        self.cursor = None
        self.data_loaded = False
        self.price_data = None
        self.config = PatternConfig()

    def connect(self) -> bool:
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"连接数据库失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def load_price_data(self) -> bool:
        """加载价格数据"""
        if not self.connect():
            return False

        try:
            print("加载价格数据...")
            query = f"""
            SELECT rowid, Time, Open, High, Low, Close 
            FROM {self.table_name} 
            ORDER BY Time
            """
            df = pd.read_sql_query(query, self.conn)

            if len(df) < self.config.min_data_points:
                print(f"数据量不足: {len(df)} 行，至少需要{self.config.min_data_points}行")
                return False

            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])

            # 填充缺失值
            with tqdm(total=4, desc="处理数据", leave=False) as pbar:
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    pbar.update(1)

            self.price_data = df
            self.data_loaded = True
            print(f"✓ 数据加载完成，共 {len(df)} 行")
            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_zigzag_points(self, prices: np.ndarray, min_swing_percent: float = None) -> List[Tuple[int, float, str]]:
        """寻找ZigZag转折点"""
        if min_swing_percent is None:
            min_swing_percent = self.config.min_swing_percent

        if len(prices) < 10:
            return []

        zigzag_points = []
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        current_trend = 0

        with tqdm(total=len(prices), desc="寻找转折点", leave=False) as pbar:
            i = 1
            while i < len(prices):
                price_change_pct = abs(prices[i] - last_pivot_price) / last_pivot_price * 100

                if current_trend == 0:
                    if prices[i] > last_pivot_price:
                        current_trend = 1
                    elif prices[i] < last_pivot_price:
                        current_trend = -1
                    else:
                        i += 1
                        pbar.update(1)
                        continue

                    last_pivot_idx = i
                    last_pivot_price = prices[i]

                elif current_trend == 1:
                    if prices[i] > last_pivot_price:
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                    elif price_change_pct >= min_swing_percent:
                        zigzag_points.append((last_pivot_idx, last_pivot_price, 'peak'))
                        current_trend = -1
                        last_pivot_idx = i
                        last_pivot_price = prices[i]

                elif current_trend == -1:
                    if prices[i] < last_pivot_price:
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                    elif price_change_pct >= min_swing_percent:
                        zigzag_points.append((last_pivot_idx, last_pivot_price, 'trough'))
                        current_trend = 1
                        last_pivot_idx = i
                        last_pivot_price = prices[i]

                i += 1
                pbar.update(1)

        # 添加最后一个点
        if last_pivot_idx != 0 and (len(zigzag_points) == 0 or zigzag_points[-1][0] != last_pivot_idx):
            if current_trend == 1:
                zigzag_points.append((last_pivot_idx, last_pivot_price, 'peak'))
            else:
                zigzag_points.append((last_pivot_idx, last_pivot_price, 'trough'))

        # 过滤相邻太近的点
        filtered_points = []
        if zigzag_points:
            filtered_points.append(zigzag_points[0])
            for i in range(1, len(zigzag_points)):
                idx_diff = zigzag_points[i][0] - zigzag_points[i - 1][0]
                price_diff_pct = abs(zigzag_points[i][1] - zigzag_points[i - 1][1]) / zigzag_points[i - 1][1] * 100

                if idx_diff >= self.config.min_distance_points and price_diff_pct >= 0.05:
                    filtered_points.append(zigzag_points[i])

        return filtered_points

    def find_extrema(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """寻找极值点"""
        with tqdm(total=2, desc="寻找极值点", leave=False) as pbar:
            peaks = argrelextrema(highs, np.greater, order=3)[0]
            pbar.update(1)
            troughs = argrelextrema(lows, np.less, order=3)[0]
            pbar.update(1)

        peaks_list = [(idx, highs[idx], 'peak') for idx in peaks]
        troughs_list = [(idx, lows[idx], 'trough') for idx in troughs]

        return peaks_list, troughs_list


class HeadShouldersDetector(BasePatternDetector):
    """头肩形态检测器"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        super().__init__(db_path, table_name)
        self.patterns = []
        self.label_column = 'HS_Pattern_Label'

    def add_label_column(self) -> bool:
        """添加标签列"""
        try:
            check_query = f"PRAGMA table_info({self.table_name})"
            self.cursor.execute(check_query)
            existing_columns = [col[1] for col in self.cursor.fetchall()]

            if self.label_column not in existing_columns:
                alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {self.label_column} TEXT"
                self.cursor.execute(alter_query)
                print(f"✓ 已添加 {self.label_column} 列")

            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加标签列时出错: {e}")
            return False

    def find_head_shoulders_patterns(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[Dict]:
        """检测头肩形态"""
        patterns = []

        # 寻找转折点
        peaks = self.find_zigzag_points(highs, min_swing_percent=0.08)
        troughs = self.find_zigzag_points(lows, min_swing_percent=0.08)

        print(f"找到 {len(peaks)} 个高点, {len(troughs)} 个低点")

        all_points = peaks + troughs
        all_points.sort(key=lambda x: x[0])

        # 检测头肩顶
        total_points = len(all_points)
        with tqdm(total=total_points, desc="检测头肩顶", leave=False) as pbar:
            for i in range(2, len(all_points) - 2):
                if (all_points[i - 2][2] == 'peak' and
                        all_points[i][2] == 'peak' and
                        all_points[i + 2][2] == 'peak'):

                    left_idx, left_price, _ = all_points[i - 2]
                    head_idx, head_price, _ = all_points[i]
                    right_idx, right_price, _ = all_points[i + 2]

                    head_height = head_price - min(left_price, right_price)
                    head_height_pct = head_height / head_price * 100
                    shoulder_diff_pct = abs(left_price - right_price) / ((left_price + right_price) / 2) * 100

                    if (head_price > left_price and
                            head_price > right_price and
                            shoulder_diff_pct < self.config.hs_shoulder_diff_pct and
                            head_height_pct >= self.config.hs_min_height_pct and
                            min(head_idx - left_idx, right_idx - head_idx) >= self.config.hs_min_time_gap):

                        neckline_points = []
                        if i - 1 < len(all_points) and all_points[i - 1][2] == 'trough':
                            neckline_points.append(all_points[i - 1])
                        if i + 1 < len(all_points) and all_points[i + 1][2] == 'trough':
                            neckline_points.append(all_points[i + 1])

                        if len(neckline_points) >= 2:
                            neckline_left_idx, neckline_left_price, _ = neckline_points[0]
                            neckline_right_idx, neckline_right_price, _ = neckline_points[1]

                            neckline_slope = (neckline_right_price - neckline_left_price) / (
                                    neckline_right_idx - neckline_left_idx)

                            # 寻找突破点
                            breakout_idx = -1
                            for j in range(right_idx + 1, min(right_idx + 50, len(closes))):
                                if j >= len(closes):
                                    break

                                neckline_level = neckline_left_price + neckline_slope * (j - neckline_left_idx)

                                if closes[j] < neckline_level - head_height * 0.5:
                                    breakout_idx = j
                                    break

                            confidence = self.calculate_hs_confidence(
                                left_price, head_price, right_price,
                                neckline_left_price, neckline_right_price,
                                head_height_pct, shoulder_diff_pct,
                                head_idx - left_idx, right_idx - head_idx
                            )

                            if confidence > self.config.hs_confidence_threshold:
                                pattern = {
                                    'type': 'head_shoulders',
                                    'subtype': 'top',
                                    'left_shoulder_idx': left_idx,
                                    'head_idx': head_idx,
                                    'right_shoulder_idx': right_idx,
                                    'neckline_break_idx': breakout_idx if breakout_idx > 0 else -1,
                                    'left_shoulder_price': left_price,
                                    'head_price': head_price,
                                    'right_shoulder_price': right_price,
                                    'neckline_left_price': neckline_left_price,
                                    'neckline_right_price': neckline_right_price,
                                    'pattern_height': head_height,
                                    'pattern_width': right_idx - left_idx,
                                    'confidence': confidence
                                }
                                patterns.append(pattern)
                pbar.update(1)

        # 检测头肩底
        with tqdm(total=total_points, desc="检测头肩底", leave=False) as pbar:
            for i in range(2, len(all_points) - 2):
                if (all_points[i - 2][2] == 'trough' and
                        all_points[i][2] == 'trough' and
                        all_points[i + 2][2] == 'trough'):

                    left_idx, left_price, _ = all_points[i - 2]
                    head_idx, head_price, _ = all_points[i]
                    right_idx, right_price, _ = all_points[i + 2]

                    head_depth = max(left_price, right_price) - head_price
                    head_depth_pct = head_depth / head_price * 100
                    shoulder_diff_pct = abs(left_price - right_price) / ((left_price + right_price) / 2) * 100

                    if (head_price < left_price and
                            head_price < right_price and
                            shoulder_diff_pct < self.config.hs_shoulder_diff_pct and
                            head_depth_pct >= self.config.hs_min_height_pct and
                            min(head_idx - left_idx, right_idx - head_idx) >= self.config.hs_min_time_gap):

                        neckline_points = []
                        if i - 1 < len(all_points) and all_points[i - 1][2] == 'peak':
                            neckline_points.append(all_points[i - 1])
                        if i + 1 < len(all_points) and all_points[i + 1][2] == 'peak':
                            neckline_points.append(all_points[i + 1])

                        if len(neckline_points) >= 2:
                            neckline_left_idx, neckline_left_price, _ = neckline_points[0]
                            neckline_right_idx, neckline_right_price, _ = neckline_points[1]

                            neckline_slope = (neckline_right_price - neckline_left_price) / (
                                    neckline_right_idx - neckline_left_idx)

                            breakout_idx = -1
                            for j in range(right_idx + 1, min(right_idx + 50, len(closes))):
                                if j >= len(closes):
                                    break

                                neckline_level = neckline_left_price + neckline_slope * (j - neckline_left_idx)

                                if closes[j] > neckline_level + head_depth * 0.5:
                                    breakout_idx = j
                                    break

                            confidence = self.calculate_hs_confidence(
                                left_price, head_price, right_price,
                                neckline_left_price, neckline_right_price,
                                head_depth_pct, shoulder_diff_pct,
                                head_idx - left_idx, right_idx - head_idx
                            )

                            if confidence > self.config.hs_confidence_threshold:
                                pattern = {
                                    'type': 'head_shoulders',
                                    'subtype': 'bottom',
                                    'left_shoulder_idx': left_idx,
                                    'head_idx': head_idx,
                                    'right_shoulder_idx': right_idx,
                                    'neckline_break_idx': breakout_idx if breakout_idx > 0 else -1,
                                    'left_shoulder_price': left_price,
                                    'head_price': head_price,
                                    'right_shoulder_price': right_price,
                                    'neckline_left_price': neckline_left_price,
                                    'neckline_right_price': neckline_right_price,
                                    'pattern_height': head_depth,
                                    'pattern_width': right_idx - left_idx,
                                    'confidence': confidence
                                }
                                patterns.append(pattern)
                pbar.update(1)

        return patterns

    def calculate_hs_confidence(self, left_price: float, head_price: float, right_price: float,
                                neckline_left: float, neckline_right: float,
                                pattern_height_pct: float, shoulder_diff_pct: float,
                                left_to_head_time: int, head_to_right_time: int) -> float:
        """计算头肩形态置信度"""
        confidence = 0.0

        if pattern_height_pct > 0.3:
            confidence += 0.25
        elif pattern_height_pct > 0.15:
            confidence += 0.15
        elif pattern_height_pct > 0.08:
            confidence += 0.08

        if shoulder_diff_pct < 1.0:
            confidence += 0.20
        elif shoulder_diff_pct < 2.0:
            confidence += 0.12
        elif shoulder_diff_pct < 3.0:
            confidence += 0.05

        if left_to_head_time > 0 and head_to_right_time > 0:
            time_ratio = min(left_to_head_time, head_to_right_time) / max(left_to_head_time, head_to_right_time)
            if time_ratio > 0.6:
                confidence += 0.20
            elif time_ratio > 0.4:
                confidence += 0.12
            elif time_ratio > 0.3:
                confidence += 0.05

        if neckline_left > 0 and neckline_right > 0:
            neckline_diff_pct = abs(neckline_right - neckline_left) / ((neckline_right + neckline_left) / 2) * 100
            if neckline_diff_pct < 1.0:
                confidence += 0.15
            elif neckline_diff_pct < 2.0:
                confidence += 0.10
            elif neckline_diff_pct < 3.0:
                confidence += 0.05

        min_time = min(left_to_head_time, head_to_right_time)
        if min_time >= 15:
            confidence += 0.20
        elif min_time >= 10:
            confidence += 0.12
        elif min_time >= 8:
            confidence += 0.08
        elif min_time >= 5:
            confidence += 0.04

        return min(confidence, 1.0)

    def detect_all_patterns(self):
        """检测所有头肩形态"""
        if not self.data_loaded:
            print("错误: 数据未加载")
            return False

        print("开始检测头肩形态...")

        try:
            df = self.price_data
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values

            patterns = self.find_head_shoulders_patterns(highs, lows, closes)
            self.patterns = self.filter_overlapping_patterns(patterns)

            print(f"✓ 检测到 {len(self.patterns)} 个头肩形态")

            if self.patterns:
                top_count = len([p for p in self.patterns if p['subtype'] == 'top'])
                bottom_count = len([p for p in self.patterns if p['subtype'] == 'bottom'])
                print(f"  头肩顶: {top_count}")
                print(f"  头肩底: {bottom_count}")

                if self.patterns:
                    avg_confidence = np.mean([p['confidence'] for p in self.patterns])
                    print(f"  平均置信度: {avg_confidence:.2f}")

            return True

        except Exception as e:
            print(f"形态检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def filter_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """过滤重叠形态"""
        if not patterns:
            return []

        with tqdm(total=len(patterns), desc="过滤重叠形态", leave=False) as pbar:
            patterns.sort(key=lambda x: x['confidence'], reverse=True)
            filtered = []
            used_indices = set()

            for pattern in patterns:
                pattern_indices = {
                    pattern['left_shoulder_idx'],
                    pattern['head_idx'],
                    pattern['right_shoulder_idx']
                }
                if pattern['neckline_break_idx'] > 0:
                    pattern_indices.add(pattern['neckline_break_idx'])

                overlap = False
                for idx in pattern_indices:
                    if idx in used_indices:
                        overlap = True
                        break

                if not overlap:
                    filtered.append(pattern)
                    used_indices.update(pattern_indices)

                pbar.update(1)

        return filtered

    def create_labels(self) -> pd.DataFrame:
        """创建标签"""
        if not self.patterns:
            print("没有检测到头肩形态")
            return None

        print("创建头肩形态标签...")

        df = self.price_data
        labels = [''] * len(df)

        with tqdm(total=len(self.patterns), desc="创建标签", leave=False) as pbar:
            for i, pattern in enumerate(self.patterns):
                pattern_id = f"HS_{i + 1}"

                # 标记左肩
                left_idx = pattern['left_shoulder_idx']
                if 0 <= left_idx < len(labels):
                    labels[left_idx] = f"{pattern_id}_左肩"

                # 标记头
                head_idx = pattern['head_idx']
                if 0 <= head_idx < len(labels):
                    labels[head_idx] = f"{pattern_id}_头"

                # 标记右肩
                right_idx = pattern['right_shoulder_idx']
                if 0 <= right_idx < len(labels):
                    labels[right_idx] = f"{pattern_id}_右肩"

                # 标记颈线突破
                break_idx = pattern['neckline_break_idx']
                if break_idx > 0 and break_idx < len(labels):
                    labels[break_idx] = f"{pattern_id}_颈线突破"

                pattern_type = "顶" if pattern['subtype'] == 'top' else "底"
                pbar.set_description(f"创建标签: 头肩{pattern_type}形 {i + 1}")
                pbar.update(1)

        result_df = pd.DataFrame({
            'rowid': df['rowid'].values,
            self.label_column: labels
        })

        labeled_count = sum(1 for label in labels if label)
        print(f"✓ 已标记 {labeled_count} 个头肩形态关键点")

        return result_df

    def update_database_labels(self) -> bool:
        """更新数据库标签"""
        try:
            if not self.add_label_column():
                return False

            labels_df = self.create_labels()
            if labels_df is None:
                print("没有头肩形态标签需要更新")
                return True

            print("正在更新数据库头肩形态标签...")

            batch_size = 100000
            total_rows = len(labels_df)

            with tqdm(total=total_rows, desc="更新头肩形态标签") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = labels_df.iloc[i:i + batch_size]

                    params = []
                    for _, row in batch.iterrows():
                        params.append((
                            row[self.label_column] if pd.notna(row[self.label_column]) else '',
                            row['rowid']
                        ))

                    update_query = f"""
                    UPDATE {self.table_name}
                    SET {self.label_column} = ?
                    WHERE rowid = ?
                    """

                    self.cursor.executemany(update_query, params)
                    self.conn.commit()

                    pbar.update(len(batch))

            print("✓ 头肩形态标签更新完成")
            return True

        except Exception as e:
            print(f"更新数据库失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_detection(self) -> bool:
        """运行检测"""
        print("=" * 60)
        print("头肩形态检测器")
        print("=" * 60)

        try:
            # 主进度条
            with tqdm(total=4, desc="头肩形态检测进度") as pbar:
                if not self.load_price_data():
                    return False
                pbar.update(1)

                if not self.detect_all_patterns():
                    return False
                pbar.update(1)

                if not self.update_database_labels():
                    return False
                pbar.update(1)

                self.display_results()
                pbar.update(1)

            return True

        except Exception as e:
            print(f"检测流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.close()

    def display_results(self):
        """显示结果"""
        if not self.patterns:
            print("未检测到头肩形态")
            return

        df = self.price_data

        print("\n头肩形态检测结果摘要:")
        print("-" * 60)

        for i, pattern in enumerate(self.patterns[:5]):
            pattern_id = f"HS_{i + 1}"
            pattern_type = "头肩顶" if pattern['subtype'] == 'top' else "头肩底"

            left_time = df.iloc[pattern['left_shoulder_idx']]['Time'] if 'Time' in df.columns else pattern[
                'left_shoulder_idx']
            head_time = df.iloc[pattern['head_idx']]['Time'] if 'Time' in df.columns else pattern['head_idx']
            right_time = df.iloc[pattern['right_shoulder_idx']]['Time'] if 'Time' in df.columns else pattern[
                'right_shoulder_idx']

            print(f"形态 {pattern_id} ({pattern_type}):")
            print(f"  左肩: 时间={left_time}, 价格={pattern['left_shoulder_price']:.2f}")
            print(f"  头: 时间={head_time}, 价格={pattern['head_price']:.2f}")
            print(f"  右肩: 时间={right_time}, 价格={pattern['right_shoulder_price']:.2f}")
            print(f"  颈线: {pattern['neckline_left_price']:.2f} -> {pattern['neckline_right_price']:.2f}")

            if pattern['neckline_break_idx'] > 0:
                break_time = df.iloc[pattern['neckline_break_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'neckline_break_idx']
                print(f"  突破: 时间={break_time}")

            print(f"  置信度: {pattern['confidence']:.2%}")
            print(f"  幅度: {pattern['pattern_height']:.2f}, 宽度: {pattern['pattern_width']}分钟")
            print("-" * 40)


class DoublePatternDetector(BasePatternDetector):
    """双重顶/底检测器"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        super().__init__(db_path, table_name)
        self.patterns = []
        self.label_column = 'Double_Pattern_Label'

    def add_label_column(self) -> bool:
        """添加标签列"""
        try:
            check_query = f"PRAGMA table_info({self.table_name})"
            self.cursor.execute(check_query)
            existing_columns = [col[1] for col in self.cursor.fetchall()]

            if self.label_column not in existing_columns:
                alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {self.label_column} TEXT"
                self.cursor.execute(alter_query)
                print(f"✓ 已添加 {self.label_column} 列")

            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加标签列时出错: {e}")
            return False

    def find_double_patterns(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[Dict]:
        """检测双重顶/底形态"""
        patterns = []

        # 寻找极值点
        peaks, troughs = self.find_extrema(highs, lows)

        # 检测双重顶
        total_peaks = len(peaks)
        if total_peaks > 1:
            with tqdm(total=total_peaks * (total_peaks - 1) // 2, desc="检测双重顶", leave=False) as pbar:
                for i in range(len(peaks) - 1):
                    for j in range(i + 1, len(peaks)):
                        idx1, price1, _ = peaks[i]
                        idx2, price2, _ = peaks[j]

                        if idx2 - idx1 < self.config.double_min_distance:
                            pbar.update(1)
                            continue

                        price_diff_pct = abs(price1 - price2) / ((price1 + price2) / 2) * 100
                        if price_diff_pct > self.config.double_price_tolerance:
                            pbar.update(1)
                            continue

                        # 寻找中间低点（颈线）
                        neckline_idx = -1
                        neckline_price = 0

                        for k in range(idx1 + 1, idx2):
                            if k < len(lows):
                                if neckline_idx == -1 or lows[k] < neckline_price:
                                    neckline_idx = k
                                    neckline_price = lows[k]

                        if neckline_idx == -1:
                            pbar.update(1)
                            continue

                        pattern_height = max(price1, price2) - neckline_price
                        pattern_height_pct = pattern_height / max(price1, price2) * 100

                        if pattern_height_pct >= self.config.double_min_height_pct:
                            # 寻找突破点
                            breakout_idx = -1
                            for k in range(idx2 + 1, min(idx2 + 30, len(closes))):
                                if closes[k] < neckline_price - pattern_height * 0.3:
                                    breakout_idx = k
                                    break

                            confidence = self.calculate_double_confidence(price1, price2, neckline_price, idx2 - idx1)

                            pattern = {
                                'type': 'double_pattern',
                                'subtype': 'top',
                                'left_peak_idx': idx1,
                                'right_peak_idx': idx2,
                                'left_peak_price': price1,
                                'right_peak_price': price2,
                                'neckline_idx': neckline_idx,
                                'neckline_price': neckline_price,
                                'neckline_break_idx': breakout_idx,
                                'pattern_height': pattern_height,
                                'pattern_width': idx2 - idx1,
                                'confidence': confidence
                            }
                            patterns.append(pattern)

                        pbar.update(1)

        # 检测双重底
        total_troughs = len(troughs)
        if total_troughs > 1:
            with tqdm(total=total_troughs * (total_troughs - 1) // 2, desc="检测双重底", leave=False) as pbar:
                for i in range(len(troughs) - 1):
                    for j in range(i + 1, len(troughs)):
                        idx1, price1, _ = troughs[i]
                        idx2, price2, _ = troughs[j]

                        if idx2 - idx1 < self.config.double_min_distance:
                            pbar.update(1)
                            continue

                        price_diff_pct = abs(price1 - price2) / ((price1 + price2) / 2) * 100
                        if price_diff_pct > self.config.double_price_tolerance:
                            pbar.update(1)
                            continue

                        # 寻找中间高点（颈线）
                        neckline_idx = -1
                        neckline_price = 0

                        for k in range(idx1 + 1, idx2):
                            if k < len(highs):
                                if neckline_idx == -1 or highs[k] > neckline_price:
                                    neckline_idx = k
                                    neckline_price = highs[k]

                        if neckline_idx == -1:
                            pbar.update(1)
                            continue

                        pattern_height = neckline_price - min(price1, price2)
                        pattern_height_pct = pattern_height / neckline_price * 100

                        if pattern_height_pct >= self.config.double_min_height_pct:
                            # 寻找突破点
                            breakout_idx = -1
                            for k in range(idx2 + 1, min(idx2 + 30, len(closes))):
                                if closes[k] > neckline_price + pattern_height * 0.3:
                                    breakout_idx = k
                                    break

                            confidence = self.calculate_double_confidence(price1, price2, neckline_price, idx2 - idx1)

                            pattern = {
                                'type': 'double_pattern',
                                'subtype': 'bottom',
                                'left_trough_idx': idx1,
                                'right_trough_idx': idx2,
                                'left_trough_price': price1,
                                'right_trough_price': price2,
                                'neckline_idx': neckline_idx,
                                'neckline_price': neckline_price,
                                'neckline_break_idx': breakout_idx,
                                'pattern_height': pattern_height,
                                'pattern_width': idx2 - idx1,
                                'confidence': confidence
                            }
                            patterns.append(pattern)

                        pbar.update(1)

        return patterns

    def calculate_double_confidence(self, price1: float, price2: float, neckline_price: float,
                                    width: int) -> float:
        """计算双重形态置信度"""
        confidence = 0.0

        # 价格相似性
        price_diff_pct = abs(price1 - price2) / ((price1 + price2) / 2) * 100
        if price_diff_pct < 0.5:
            confidence += 0.3
        elif price_diff_pct < 1.0:
            confidence += 0.2
        elif price_diff_pct < 1.5:
            confidence += 0.1

        # 形态高度
        if neckline_price > 0:
            pattern_height = abs(max(price1, price2) - neckline_price)
            height_pct = pattern_height / neckline_price * 100
            if height_pct > 0.3:
                confidence += 0.3
            elif height_pct > 0.2:
                confidence += 0.2
            elif height_pct > 0.15:
                confidence += 0.1

        # 形态宽度
        if width >= 10:
            confidence += 0.2
        elif width >= 8:
            confidence += 0.15
        elif width >= 5:
            confidence += 0.1

        # 时间对称性（占20%）
        confidence += 0.2

        return min(confidence, 1.0)

    def detect_all_patterns(self):
        """检测所有双重形态"""
        if not self.data_loaded:
            print("错误: 数据未加载")
            return False

        print("开始检测双重顶/底形态...")

        try:
            df = self.price_data
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values

            patterns = self.find_double_patterns(highs, lows, closes)
            self.patterns = self.filter_overlapping_patterns(patterns)

            print(f"✓ 检测到 {len(self.patterns)} 个双重形态")

            if self.patterns:
                top_count = len([p for p in self.patterns if p['subtype'] == 'top'])
                bottom_count = len([p for p in self.patterns if p['subtype'] == 'bottom'])
                print(f"  双重顶: {top_count}")
                print(f"  双重底: {bottom_count}")

                if self.patterns:
                    avg_confidence = np.mean([p['confidence'] for p in self.patterns])
                    print(f"  平均置信度: {avg_confidence:.2f}")

            return True

        except Exception as e:
            print(f"形态检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def filter_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """过滤重叠形态"""
        if not patterns:
            return []

        with tqdm(total=len(patterns), desc="过滤重叠形态", leave=False) as pbar:
            patterns.sort(key=lambda x: x['confidence'], reverse=True)
            filtered = []
            used_indices = set()

            for pattern in patterns:
                pattern_indices = set()

                if 'left_peak_idx' in pattern:
                    pattern_indices.add(pattern['left_peak_idx'])
                    pattern_indices.add(pattern['right_peak_idx'])
                elif 'left_trough_idx' in pattern:
                    pattern_indices.add(pattern['left_trough_idx'])
                    pattern_indices.add(pattern['right_trough_idx'])

                if pattern.get('neckline_break_idx', -1) > 0:
                    pattern_indices.add(pattern['neckline_break_idx'])

                overlap = False
                for idx in pattern_indices:
                    if idx in used_indices:
                        overlap = True
                        break

                if not overlap:
                    filtered.append(pattern)
                    used_indices.update(pattern_indices)

                pbar.update(1)

        return filtered

    def create_labels(self) -> pd.DataFrame:
        """创建标签"""
        if not self.patterns:
            print("没有检测到双重形态")
            return None

        print("创建双重形态标签...")

        df = self.price_data
        labels = [''] * len(df)

        with tqdm(total=len(self.patterns), desc="创建标签", leave=False) as pbar:
            for i, pattern in enumerate(self.patterns):
                pattern_id = f"Double_{i + 1}"

                if pattern['subtype'] == 'top':
                    # 双重顶
                    left_idx = pattern['left_peak_idx']
                    right_idx = pattern['right_peak_idx']
                    neckline_idx = pattern['neckline_idx']

                    if 0 <= left_idx < len(labels):
                        labels[left_idx] = f"{pattern_id}_左顶"

                    if 0 <= right_idx < len(labels):
                        labels[right_idx] = f"{pattern_id}_右顶"

                    if 0 <= neckline_idx < len(labels):
                        labels[neckline_idx] = f"{pattern_id}_颈线"

                else:
                    # 双重底
                    left_idx = pattern['left_trough_idx']
                    right_idx = pattern['right_trough_idx']
                    neckline_idx = pattern['neckline_idx']

                    if 0 <= left_idx < len(labels):
                        labels[left_idx] = f"{pattern_id}_左底"

                    if 0 <= right_idx < len(labels):
                        labels[right_idx] = f"{pattern_id}_右底"

                    if 0 <= neckline_idx < len(labels):
                        labels[neckline_idx] = f"{pattern_id}_颈线"

                # 标记突破点
                break_idx = pattern.get('neckline_break_idx', -1)
                if break_idx > 0 and break_idx < len(labels):
                    labels[break_idx] = f"{pattern_id}_突破"

                pbar.update(1)

        result_df = pd.DataFrame({
            'rowid': df['rowid'].values,
            self.label_column: labels
        })

        labeled_count = sum(1 for label in labels if label)
        print(f"✓ 已标记 {labeled_count} 个双重形态关键点")

        return result_df

    def update_database_labels(self) -> bool:
        """更新数据库标签"""
        try:
            if not self.add_label_column():
                return False

            labels_df = self.create_labels()
            if labels_df is None:
                print("没有双重形态标签需要更新")
                return True

            print("正在更新数据库双重形态标签...")

            batch_size = 1000
            total_rows = len(labels_df)

            with tqdm(total=total_rows, desc="更新双重形态标签") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = labels_df.iloc[i:i + batch_size]

                    params = []
                    for _, row in batch.iterrows():
                        params.append((
                            row[self.label_column] if pd.notna(row[self.label_column]) else '',
                            row['rowid']
                        ))

                    update_query = f"""
                    UPDATE {self.table_name}
                    SET {self.label_column} = ?
                    WHERE rowid = ?
                    """

                    self.cursor.executemany(update_query, params)
                    self.conn.commit()

                    pbar.update(len(batch))

            print("✓ 双重形态标签更新完成")
            return True

        except Exception as e:
            print(f"更新数据库失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_detection(self) -> bool:
        """运行检测"""
        print("=" * 60)
        print("双重顶/底形态检测器")
        print("=" * 60)

        try:
            # 主进度条
            with tqdm(total=3, desc="双重形态检测进度") as pbar:
                if not self.load_price_data():
                    return False
                pbar.update(1)

                # if not self.detect_all_patterns():
                #     return False
                # pbar.update(1)

                if not self.update_database_labels():
                    return False
                pbar.update(1)

                self.display_results()
                pbar.update(1)

            return True

        except Exception as e:
            print(f"检测流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.close()

    def display_results(self):
        """显示结果"""
        if not self.patterns:
            print("未检测到双重形态")
            return

        df = self.price_data

        print("\n双重形态检测结果摘要:")
        print("-" * 60)

        for i, pattern in enumerate(self.patterns[:5]):
            pattern_id = f"Double_{i + 1}"
            pattern_type = "双重顶" if pattern['subtype'] == 'top' else "双重底"

            if pattern['subtype'] == 'top':
                left_time = df.iloc[pattern['left_peak_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'left_peak_idx']
                right_time = df.iloc[pattern['right_peak_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'right_peak_idx']
                neckline_time = df.iloc[pattern['neckline_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'neckline_idx']

                print(f"形态 {pattern_id} ({pattern_type}):")
                print(f"  左顶: 时间={left_time}, 价格={pattern['left_peak_price']:.2f}")
                print(f"  右顶: 时间={right_time}, 价格={pattern['right_peak_price']:.2f}")
                print(f"  颈线: 时间={neckline_time}, 价格={pattern['neckline_price']:.2f}")
            else:
                left_time = df.iloc[pattern['left_trough_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'left_trough_idx']
                right_time = df.iloc[pattern['right_trough_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'right_trough_idx']
                neckline_time = df.iloc[pattern['neckline_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'neckline_idx']

                print(f"形态 {pattern_id} ({pattern_type}):")
                print(f"  左底: 时间={left_time}, 价格={pattern['left_trough_price']:.2f}")
                print(f"  右底: 时间={right_time}, 价格={pattern['right_trough_price']:.2f}")
                print(f"  颈线: 时间={neckline_time}, 价格={pattern['neckline_price']:.2f}")

            if pattern.get('neckline_break_idx', -1) > 0:
                break_time = df.iloc[pattern['neckline_break_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'neckline_break_idx']
                print(f"  突破: 时间={break_time}")

            print(f"  置信度: {pattern['confidence']:.2%}")
            print(f"  幅度: {pattern['pattern_height']:.2f}, 宽度: {pattern['pattern_width']}分钟")
            print("-" * 40)


class TriplePatternDetector(BasePatternDetector):
    """三重顶/底检测器"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        super().__init__(db_path, table_name)
        self.patterns = []
        self.label_column = 'Triple_Pattern_Label'

    def add_label_column(self) -> bool:
        """添加标签列"""
        try:
            check_query = f"PRAGMA table_info({self.table_name})"
            self.cursor.execute(check_query)
            existing_columns = [col[1] for col in self.cursor.fetchall()]

            if self.label_column not in existing_columns:
                alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {self.label_column} TEXT"
                self.cursor.execute(alter_query)
                print(f"✓ 已添加 {self.label_column} 列")

            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加标签列时出错: {e}")
            return False

    def find_triple_patterns(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[Dict]:
        """检测三重顶/底形态"""
        patterns = []

        # 寻找极值点
        peaks, troughs = self.find_extrema(highs, lows)

        # 检测三重顶
        total_peaks = len(peaks)
        if total_peaks >= 3:
            total_combinations = total_peaks * (total_peaks - 1) * (total_peaks - 2) // 6
            with tqdm(total=total_combinations, desc="检测三重顶", leave=False) as pbar:
                for i in range(len(peaks) - 2):
                    for j in range(i + 1, len(peaks) - 1):
                        for k in range(j + 1, len(peaks)):
                            idx1, price1, _ = peaks[i]
                            idx2, price2, _ = peaks[j]
                            idx3, price3, _ = peaks[k]

                            # 检查时间间隔
                            if (idx2 - idx1 < self.config.triple_min_distance or
                                    idx3 - idx2 < self.config.triple_min_distance):
                                pbar.update(1)
                                continue

                            # 检查价格相似性
                            avg_price = (price1 + price2 + price3) / 3
                            max_diff = max(abs(p - avg_price) for p in [price1, price2, price3])
                            if max_diff / avg_price * 100 > self.config.triple_price_tolerance:
                                pbar.update(1)
                                continue

                            # 寻找颈线
                            neckline_idx1, neckline_price1 = self.find_lowest_between(lows, idx1, idx2)
                            neckline_idx2, neckline_price2 = self.find_lowest_between(lows, idx2, idx3)

                            pattern_height = avg_price - (neckline_price1 + neckline_price2) / 2
                            pattern_height_pct = pattern_height / avg_price * 100

                            if pattern_height_pct >= self.config.triple_min_height_pct:
                                confidence = self.calculate_triple_confidence(
                                    [price1, price2, price3], [idx1, idx2, idx3],
                                    pattern_height
                                )

                                if confidence > 0.6:
                                    pattern = {
                                        'type': 'triple_pattern',
                                        'subtype': 'top',
                                        'peak_indices': [idx1, idx2, idx3],
                                        'peak_prices': [price1, price2, price3],
                                        'neckline_indices': [neckline_idx1, neckline_idx2],
                                        'neckline_prices': [neckline_price1, neckline_price2],
                                        'pattern_height': pattern_height,
                                        'pattern_width': idx3 - idx1,
                                        'confidence': confidence
                                    }
                                    patterns.append(pattern)

                            pbar.update(1)

        # 检测三重底
        total_troughs = len(troughs)
        if total_troughs >= 3:
            total_combinations = total_troughs * (total_troughs - 1) * (total_troughs - 2) // 6
            with tqdm(total=total_combinations, desc="检测三重底", leave=False) as pbar:
                for i in range(len(troughs) - 2):
                    for j in range(i + 1, len(troughs) - 1):
                        for k in range(j + 1, len(troughs)):
                            idx1, price1, _ = troughs[i]
                            idx2, price2, _ = troughs[j]
                            idx3, price3, _ = troughs[k]

                            if (idx2 - idx1 < self.config.triple_min_distance or
                                    idx3 - idx2 < self.config.triple_min_distance):
                                pbar.update(1)
                                continue

                            avg_price = (price1 + price2 + price3) / 3
                            max_diff = max(abs(p - avg_price) for p in [price1, price2, price3])
                            if max_diff / avg_price * 100 > self.config.triple_price_tolerance:
                                pbar.update(1)
                                continue

                            neckline_idx1, neckline_price1 = self.find_highest_between(highs, idx1, idx2)
                            neckline_idx2, neckline_price2 = self.find_highest_between(highs, idx2, idx3)

                            pattern_height = (neckline_price1 + neckline_price2) / 2 - avg_price
                            pattern_height_pct = pattern_height / avg_price * 100

                            if pattern_height_pct >= self.config.triple_min_height_pct:
                                confidence = self.calculate_triple_confidence(
                                    [price1, price2, price3], [idx1, idx2, idx3],
                                    pattern_height
                                )

                                if confidence > 0.6:
                                    pattern = {
                                        'type': 'triple_pattern',
                                        'subtype': 'bottom',
                                        'trough_indices': [idx1, idx2, idx3],
                                        'trough_prices': [price1, price2, price3],
                                        'neckline_indices': [neckline_idx1, neckline_idx2],
                                        'neckline_prices': [neckline_price1, neckline_price2],
                                        'pattern_height': pattern_height,
                                        'pattern_width': idx3 - idx1,
                                        'confidence': confidence
                                    }
                                    patterns.append(pattern)

                            pbar.update(1)

        return patterns

    def find_lowest_between(self, lows: np.ndarray, start_idx: int, end_idx: int) -> Tuple[int, float]:
        """寻找两个索引之间的最低点"""
        min_idx = start_idx + 1
        min_price = lows[min_idx]

        for i in range(start_idx + 2, end_idx):
            if i < len(lows) and lows[i] < min_price:
                min_price = lows[i]
                min_idx = i

        return min_idx, min_price

    def find_highest_between(self, highs: np.ndarray, start_idx: int, end_idx: int) -> Tuple[int, float]:
        """寻找两个索引之间的最高点"""
        max_idx = start_idx + 1
        max_price = highs[max_idx]

        for i in range(start_idx + 2, end_idx):
            if i < len(highs) and highs[i] > max_price:
                max_price = highs[i]
                max_idx = i

        return max_idx, max_price

    def calculate_triple_confidence(self, prices: List[float], indices: List[int],
                                    pattern_height: float) -> float:
        """计算三重形态置信度"""
        confidence = 0.0

        # 价格相似性
        avg_price = np.mean(prices)
        std_price = np.std(prices)
        cv = std_price / avg_price * 100

        if cv < 0.5:
            confidence += 0.3
        elif cv < 1.0:
            confidence += 0.2
        elif cv < 1.5:
            confidence += 0.1

        # 形态高度
        height_pct = pattern_height / avg_price * 100
        if height_pct > 0.4:
            confidence += 0.3
        elif height_pct > 0.3:
            confidence += 0.2
        elif height_pct > 0.2:
            confidence += 0.1

        # 时间对称性
        time_gap1 = indices[1] - indices[0]
        time_gap2 = indices[2] - indices[1]
        time_ratio = min(time_gap1, time_gap2) / max(time_gap1, time_gap2)

        if time_ratio > 0.7:
            confidence += 0.2
        elif time_ratio > 0.5:
            confidence += 0.15
        elif time_ratio > 0.3:
            confidence += 0.1

        # 总体宽度
        total_width = indices[2] - indices[0]
        if total_width >= 20:
            confidence += 0.2
        elif total_width >= 15:
            confidence += 0.15
        elif total_width >= 10:
            confidence += 0.1

        return min(confidence, 1.0)

    def detect_all_patterns(self):
        """检测所有三重形态"""
        if not self.data_loaded:
            print("错误: 数据未加载")
            return False

        print("开始检测三重顶/底形态...")

        try:
            df = self.price_data
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values

            patterns = self.find_triple_patterns(highs, lows, closes)
            self.patterns = self.filter_overlapping_patterns(patterns)

            print(f"✓ 检测到 {len(self.patterns)} 个三重形态")

            if self.patterns:
                top_count = len([p for p in self.patterns if p['subtype'] == 'top'])
                bottom_count = len([p for p in self.patterns if p['subtype'] == 'bottom'])
                print(f"  三重顶: {top_count}")
                print(f"  三重底: {bottom_count}")

                if self.patterns:
                    avg_confidence = np.mean([p['confidence'] for p in self.patterns])
                    print(f"  平均置信度: {avg_confidence:.2f}")

            return True

        except Exception as e:
            print(f"形态检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def filter_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """过滤重叠形态"""
        if not patterns:
            return []

        with tqdm(total=len(patterns), desc="过滤重叠形态", leave=False) as pbar:
            patterns.sort(key=lambda x: x['confidence'], reverse=True)
            filtered = []
            used_indices = set()

            for pattern in patterns:
                pattern_indices = set()

                if 'peak_indices' in pattern:
                    pattern_indices.update(pattern['peak_indices'])
                    pattern_indices.update(pattern['neckline_indices'])
                elif 'trough_indices' in pattern:
                    pattern_indices.update(pattern['trough_indices'])
                    pattern_indices.update(pattern['neckline_indices'])

                overlap = False
                for idx in pattern_indices:
                    if idx in used_indices:
                        overlap = True
                        break

                if not overlap:
                    filtered.append(pattern)
                    used_indices.update(pattern_indices)

                pbar.update(1)

        return filtered

    def create_labels(self) -> pd.DataFrame:
        """创建标签"""
        if not self.patterns:
            print("没有检测到三重形态")
            return None

        print("创建三重形态标签...")

        df = self.price_data
        labels = [''] * len(df)

        with tqdm(total=len(self.patterns), desc="创建标签", leave=False) as pbar:
            for i, pattern in enumerate(self.patterns):
                pattern_id = f"Triple_{i + 1}"

                if pattern['subtype'] == 'top':
                    # 三重顶
                    for j, idx in enumerate(pattern['peak_indices']):
                        if 0 <= idx < len(labels):
                            labels[idx] = f"{pattern_id}_顶{j + 1}"

                    for j, idx in enumerate(pattern['neckline_indices']):
                        if 0 <= idx < len(labels):
                            labels[idx] = f"{pattern_id}_颈线{j + 1}"
                else:
                    # 三重底
                    for j, idx in enumerate(pattern['trough_indices']):
                        if 0 <= idx < len(labels):
                            labels[idx] = f"{pattern_id}_底{j + 1}"

                    for j, idx in enumerate(pattern['neckline_indices']):
                        if 0 <= idx < len(labels):
                            labels[idx] = f"{pattern_id}_颈线{j + 1}"

                pbar.update(1)

        result_df = pd.DataFrame({
            'rowid': df['rowid'].values,
            self.label_column: labels
        })

        labeled_count = sum(1 for label in labels if label)
        print(f"✓ 已标记 {labeled_count} 个三重形态关键点")

        return result_df

    def update_database_labels(self) -> bool:
        """更新数据库标签"""
        try:
            if not self.add_label_column():
                return False

            labels_df = self.create_labels()
            if labels_df is None:
                print("没有三重形态标签需要更新")
                return True

            print("正在更新数据库三重形态标签...")

            batch_size = 1000
            total_rows = len(labels_df)

            with tqdm(total=total_rows, desc="更新三重形态标签") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = labels_df.iloc[i:i + batch_size]

                    params = []
                    for _, row in batch.iterrows():
                        params.append((
                            row[self.label_column] if pd.notna(row[self.label_column]) else '',
                            row['rowid']
                        ))

                    update_query = f"""
                    UPDATE {self.table_name}
                    SET {self.label_column} = ?
                    WHERE rowid = ?
                    """

                    self.cursor.executemany(update_query, params)
                    self.conn.commit()

                    pbar.update(len(batch))

            print("✓ 三重形态标签更新完成")
            return True

        except Exception as e:
            print(f"更新数据库失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_detection(self) -> bool:
        """运行检测"""
        print("=" * 60)
        print("三重顶/底形态检测器")
        print("=" * 60)

        try:
            # 主进度条
            with tqdm(total=4, desc="三重形态检测进度") as pbar:
                if not self.load_price_data():
                    return False
                pbar.update(1)

                if not self.detect_all_patterns():
                    return False
                pbar.update(1)

                if not self.update_database_labels():
                    return False
                pbar.update(1)

                self.display_results()
                pbar.update(1)

            return True

        except Exception as e:
            print(f"检测流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.close()

    def display_results(self):
        """显示结果"""
        if not self.patterns:
            print("未检测到三重形态")
            return

        df = self.price_data

        print("\n三重形态检测结果摘要:")
        print("-" * 60)

        for i, pattern in enumerate(self.patterns[:3]):
            pattern_id = f"Triple_{i + 1}"
            pattern_type = "三重顶" if pattern['subtype'] == 'top' else "三重底"

            print(f"形态 {pattern_id} ({pattern_type}):")

            if pattern['subtype'] == 'top':
                for j, idx in enumerate(pattern['peak_indices']):
                    price = pattern['peak_prices'][j]
                    time = df.iloc[idx]['Time'] if 'Time' in df.columns else idx
                    print(f"  顶{j + 1}: 时间={time}, 价格={price:.2f}")
            else:
                for j, idx in enumerate(pattern['trough_indices']):
                    price = pattern['trough_prices'][j]
                    time = df.iloc[idx]['Time'] if 'Time' in df.columns else idx
                    print(f"  底{j + 1}: 时间={time}, 价格={price:.2f}")

            for j, idx in enumerate(pattern['neckline_indices']):
                price = pattern['neckline_prices'][j]
                time = df.iloc[idx]['Time'] if 'Time' in df.columns else idx
                print(f"  颈线{j + 1}: 时间={time}, 价格={price:.2f}")

            print(f"  置信度: {pattern['confidence']:.2%}")
            print(f"  幅度: {pattern['pattern_height']:.2f}, 宽度: {pattern['pattern_width']}分钟")
            print("-" * 40)


class ComprehensiveReversalDetector:
    """综合反转形态检测器"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        self.db_path = db_path
        self.table_name = table_name

        # 初始化各形态检测器
        self.detectors = {
            'head_shoulders': HeadShouldersDetector(db_path, table_name),
            # 'double_pattern': DoublePatternDetector(db_path, table_name),
            # 'triple_pattern': TriplePatternDetector(db_path, table_name)
        }

        self.all_patterns = {}
        self.label_columns = [
            'HS_Pattern_Label',
            # 'Double_Pattern_Label',
            # 'Triple_Pattern_Label'
        ]

        # 性能统计
        self.execution_times = {}

    def add_all_label_columns(self) -> bool:
        """添加所有标签列"""
        try:
            # 连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查表结构
            check_query = f"PRAGMA table_info({self.table_name})"
            cursor.execute(check_query)
            existing_columns = [col[1] for col in cursor.fetchall()]

            # 添加缺失的列
            added_columns = []
            with tqdm(total=len(self.label_columns), desc="检查/添加标签列", leave=False) as pbar:
                for column in self.label_columns:
                    if column not in existing_columns:
                        alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {column} TEXT"
                        cursor.execute(alter_query)
                        added_columns.append(column)
                    pbar.update(1)

            conn.commit()
            conn.close()

            if added_columns:
                print(f"✓ 已添加 {len(added_columns)} 个标签列: {', '.join(added_columns)}")
            else:
                print("✓ 所有标签列已存在")

            return True

        except Exception as e:
            print(f"添加标签列时出错: {e}")
            return False

    def run_all_detections(self) -> bool:
        """运行所有形态检测"""
        print("=" * 60)
        print("综合反转形态检测器 - 开始运行")
        print("=" * 60)

        start_time = time.time()

        # 添加所有标签列
        print("\n步骤 1/4: 准备数据库...")
        if not self.add_all_label_columns():
            print("错误: 无法添加标签列")
            return False

        success_count = 0
        detector_names = {
            'head_shoulders': '头肩形态检测',
            'double_pattern': '双重顶底检测',
            'triple_pattern': '三重顶底检测'
        }

        # 依次运行各个检测器
        print("\n步骤 2/4: 运行形态检测...")
        with tqdm(total=len(self.detectors), desc="综合检测进度") as main_pbar:
            for name, detector in self.detectors.items():
                detector_start = time.time()
                detector_name = detector_names.get(name, name)

                main_pbar.set_description(f"正在检测: {detector_name}")

                print(f"\n{'=' * 40}")
                print(f"运行 {detector_name}...")
                print(f"{'=' * 40}")

                if detector.run_detection():
                    success_count += 1
                    self.all_patterns[name] = detector.patterns
                else:
                    print(f"警告: {detector_name} 运行失败")

                detector_time = time.time() - detector_start
                self.execution_times[name] = detector_time
                print(f"完成 {detector_name}，耗时: {detector_time:.1f}秒")

                main_pbar.update(1)

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"综合检测完成")
        print(f"{'=' * 60}")

        # 显示性能统计
        self.display_performance_stats(total_time)

        # 显示摘要
        self.display_summary()

        return success_count > 0

    def display_performance_stats(self, total_time: float):
        """显示性能统计"""
        print("\n性能统计:")
        print("-" * 40)
        print(f"总耗时: {total_time:.1f}秒")
        print(f"平均每个检测器: {total_time / len(self.detectors):.1f}秒")

        if self.execution_times:
            print("\n各检测器耗时:")
            for name, exec_time in self.execution_times.items():
                detector_name = {
                    'head_shoulders': '头肩形态',
                    'double_pattern': '双重顶底',
                    'triple_pattern': '三重顶底'
                }.get(name, name)
                print(f"  {detector_name}: {exec_time:.1f}秒")

    def display_summary(self):
        """显示综合摘要"""
        print("\n综合形态检测摘要:")
        print("-" * 60)

        total_patterns = 0
        pattern_summary = []

        for name, patterns in self.all_patterns.items():
            if patterns:
                count = len(patterns)
                total_patterns += count

                # 翻译检测器名称
                detector_name = {
                    'head_shoulders': '头肩形态',
                    'double_pattern': '双重顶底',
                    'triple_pattern': '三重顶底'
                }.get(name, name)

                if patterns:
                    avg_confidence = np.mean([p['confidence'] for p in patterns])
                    pattern_summary.append({
                        'name': detector_name,
                        'count': count,
                        'avg_confidence': avg_confidence
                    })
                else:
                    pattern_summary.append({
                        'name': detector_name,
                        'count': 0,
                        'avg_confidence': 0
                    })

        # 按数量排序
        pattern_summary.sort(key=lambda x: x['count'], reverse=True)

        for summary in pattern_summary:
            if summary['count'] > 0:
                print(f"  {summary['name']:10} : {summary['count']:3d} 个，平均置信度: {summary['avg_confidence']:.2%}")
            else:
                print(f"  {summary['name']:10} : 0 个")

        print(f"\n  总形态数: {total_patterns}")

        # 显示SQL查询示例
        print(f"\n数据库查询示例:")
        for column in self.label_columns:
            column_name = {
                'HS_Pattern_Label': '头肩形态',
                'Double_Pattern_Label': '双重形态',
                'Triple_Pattern_Label': '三重形态'
            }.get(column, column)
            print(f"  -- 查询{column_name}:")
            print(f"  SELECT Time, Close, {column} FROM {self.table_name} WHERE {column} != '' LIMIT 5;")

    def get_pattern_statistics(self) -> Dict:
        """获取形态统计信息"""
        stats = {}

        for name, patterns in self.all_patterns.items():
            if patterns:
                stats[name] = {
                    'count': len(patterns),
                    'avg_confidence': np.mean([p['confidence'] for p in patterns]) if patterns else 0,
                    'avg_height': np.mean([p.get('pattern_height', 0) for p in patterns]) if patterns else 0,
                    'avg_width': np.mean([p.get('pattern_width', 0) for p in patterns]) if patterns else 0
                }

        return stats


# 使用示例
if __name__ == "__main__":
    # 配置参数
    DB_PATH = "./identifier.sqlite"  # 数据库路径
    TABLE_NAME = "PriceData"  # 表名

    print("综合反转形态检测器")
    print("=" * 60)
    print(f"数据库: {DB_PATH}")
    print(f"数据表: {TABLE_NAME}")
    print("=" * 60)

    # 创建综合检测器
    detector = ComprehensiveReversalDetector(DB_PATH, TABLE_NAME)

    # 运行所有检测
    success = detector.run_all_detections()

    if success:
        print("\n" + "=" * 60)
        print("✓ 检测成功！数据库已更新。")
        print("=" * 60)

        print("\n您可以使用以下SQL查询查看所有结果:")
        print(f"  SELECT Time, Close, HS_Pattern_Label, Double_Pattern_Label, Triple_Pattern_Label")
        print(f"  FROM {TABLE_NAME} ")
        print(f"  WHERE HS_Pattern_Label != '' OR Double_Pattern_Label != '' OR Triple_Pattern_Label != ''")
        print(f"  ORDER BY Time;")

        # 显示统计信息
        stats = detector.get_pattern_statistics()
        if stats:
            print(f"\n详细形态统计:")
            print("-" * 40)
            for pattern_type, data in stats.items():
                pattern_name = {
                    'head_shoulders': '头肩形态',
                    'double_pattern': '双重形态',
                    'triple_pattern': '三重形态'
                }.get(pattern_type, pattern_type)
                print(f"  {pattern_name}:")
                print(f"    数量: {data['count']}个")
                print(f"    平均置信度: {data['avg_confidence']:.2%}")
                print(f"    平均幅度: {data['avg_height']:.2f}")
                print(f"    平均宽度: {data['avg_width']:.1f}分钟")
    else:
        print("\n" + "=" * 60)
        print("✗ 检测失败或部分失败，请检查错误信息。")
        print("=" * 60)

    print("\n程序结束。")
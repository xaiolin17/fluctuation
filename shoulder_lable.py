import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from typing import List, Dict, Tuple, Optional
import datetime

warnings.filterwarnings('ignore')


class HeadShouldersDetector:
    """头肩形态检测器 - 专为1分钟数据优化"""

    def __init__(self, db_path: str, table_name: str = 'Price'):
        """
        初始化头肩形态检测器

        Args:
            db_path: SQLite数据库路径
            table_name: 数据表名
        """
        self.db_path = db_path
        self.table_name = table_name
        self.conn = None
        self.cursor = None
        self.data_loaded = False
        self.price_data = None
        self.hs_patterns = []

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

    def add_label_column(self) -> bool:
        """添加头肩形态标签列"""
        try:
            # 检查表结构
            check_query = f"PRAGMA table_info({self.table_name})"
            self.cursor.execute(check_query)
            existing_columns = [col[1] for col in self.cursor.fetchall()]

            # 添加标签列
            if 'HS_Pattern_Label' not in existing_columns:
                alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN HS_Pattern_Label TEXT"
                self.cursor.execute(alter_query)
                print("✓ 已添加 HS_Pattern_Label 列")

            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加标签列时出错: {e}")
            return False

    def load_price_data(self) -> bool:
        """从数据库加载价格数据并预处理"""
        if not self.connect():
            return False

        try:
            print("正在加载价格数据...")

            # 读取数据，确保按时间排序
            query = f"""
            SELECT rowid, Time, Open, High, Low, Close 
            FROM {self.table_name} 
            ORDER BY Time
            """
            df = pd.read_sql_query(query, self.conn)

            if len(df) < 100:  # 至少需要100个数据点进行有效分析
                print(f"数据量不足: {len(df)} 行，至少需要100行")
                return False

            # 处理时间列
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])

            # 数据质量检查
            print(f"数据统计:")
            print(f"  时间范围: {df['Time'].min()} 到 {df['Time'].max()}")
            print(f"  数据行数: {len(df)}")
            print(f"  数据列: {df.columns.tolist()}")

            # 检查缺失值
            missing_values = df[['Open', 'High', 'Low', 'Close']].isnull().sum()
            if missing_values.sum() > 0:
                print(f"发现缺失值: {missing_values.to_dict()}")
                # 使用前后均值填充缺失值
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                print("✓ 已填充缺失值")

            # 验证价格数据合理性
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if df[col].min() <= 0:
                    print(f"警告: {col}列包含非正值")

            # 计算基本统计
            print("价格统计信息:")
            for col in price_columns:
                print(f"  {col}: 均值={df[col].mean():.2f}, 标准差={df[col].std():.2f}")

            self.price_data = df
            self.data_loaded = True
            print(f"✓ 数据加载完成，共 {len(df)} 行")
            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_zigzag_points(self, prices: np.ndarray, min_swing_percent: float = 0.1) -> List[Tuple[int, float, str]]:
        """
        使用ZigZag算法寻找关键转折点

        Args:
            prices: 价格序列
            min_swing_percent: 最小波动百分比

        Returns:
            List of (index, price, type) where type is 'peak' or 'trough'
        """
        if len(prices) < 10:
            return []

        zigzag_points = []
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        current_trend = 0  # 0:未定义, 1:上升, -1:下降

        i = 1
        while i < len(prices):
            # 计算价格变动百分比
            price_change_pct = abs(prices[i] - last_pivot_price) / last_pivot_price * 100

            if current_trend == 0:
                # 初始化趋势
                if prices[i] > last_pivot_price:
                    current_trend = 1
                elif prices[i] < last_pivot_price:
                    current_trend = -1
                else:
                    i += 1
                    continue

                last_pivot_idx = i
                last_pivot_price = prices[i]

            elif current_trend == 1:  # 上升趋势
                if prices[i] > last_pivot_price:
                    # 更新高点
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
                elif price_change_pct >= min_swing_percent:
                    # 记录高点，趋势反转
                    zigzag_points.append((last_pivot_idx, last_pivot_price, 'peak'))
                    current_trend = -1
                    last_pivot_idx = i
                    last_pivot_price = prices[i]

            elif current_trend == -1:  # 下降趋势
                if prices[i] < last_pivot_price:
                    # 更新低点
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
                elif price_change_pct >= min_swing_percent:
                    # 记录低点，趋势反转
                    zigzag_points.append((last_pivot_idx, last_pivot_price, 'trough'))
                    current_trend = 1
                    last_pivot_idx = i
                    last_pivot_price = prices[i]

            i += 1

        # 添加最后一个点
        if last_pivot_idx != 0 and (len(zigzag_points) == 0 or zigzag_points[-1][0] != last_pivot_idx):
            if current_trend == 1:
                zigzag_points.append((last_pivot_idx, last_pivot_price, 'peak'))
            else:
                zigzag_points.append((last_pivot_idx, last_pivot_price, 'trough'))

        # 过滤掉相邻太近的点
        filtered_points = []
        if zigzag_points:
            filtered_points.append(zigzag_points[0])
            for i in range(1, len(zigzag_points)):
                idx_diff = zigzag_points[i][0] - zigzag_points[i - 1][0]
                price_diff_pct = abs(zigzag_points[i][1] - zigzag_points[i - 1][1]) / zigzag_points[i - 1][1] * 100

                if idx_diff >= 5 and price_diff_pct >= 0.05:  # 至少5个时间单位，0.05%的价格变化
                    filtered_points.append(zigzag_points[i])

        return filtered_points

    def find_head_shoulders_patterns(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[Dict]:
        """
        检测头肩形态

        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列

        Returns:
            头肩形态列表
        """
        patterns = []

        # 寻找ZigZag转折点
        peaks = self.find_zigzag_points(highs, min_swing_percent=0.08)  # 寻找高点
        troughs = self.find_zigzag_points(lows, min_swing_percent=0.08)  # 寻找低点

        print(f"找到 {len(peaks)} 个高点, {len(troughs)} 个低点")

        # 合并所有转折点并按索引排序
        all_points = peaks + troughs
        all_points.sort(key=lambda x: x[0])

        # 检测头肩顶形态 (头肩形态的反转形态)
        for i in range(2, len(all_points) - 2):
            # 检查是否符合头肩顶模式: 左肩 < 头 > 右肩
            if (all_points[i - 2][2] == 'peak' and  # 左肩是高点
                    all_points[i][2] == 'peak' and  # 头是高点
                    all_points[i + 2][2] == 'peak'):  # 右肩是高点

                left_idx, left_price, _ = all_points[i - 2]
                head_idx, head_price, _ = all_points[i]
                right_idx, right_price, _ = all_points[i + 2]

                # 计算形态参数
                head_height = head_price - min(left_price, right_price)
                head_height_pct = head_height / head_price * 100

                shoulder_diff_pct = abs(left_price - right_price) / ((left_price + right_price) / 2) * 100

                # 形态条件:
                # 1. 头必须高于两个肩膀
                # 2. 两个肩膀高度大致相等（差异小于1%）
                # 3. 头的高度至少为价格的0.2%
                # 4. 时间间距合理（肩膀之间至少10个时间单位）
                if (head_price > left_price and
                        head_price > right_price and
                        shoulder_diff_pct < 1.0 and  # 肩膀高度差异小于1%
                        head_height_pct >= 0.2 and  # 头的高度至少0.2%
                        min(head_idx - left_idx, right_idx - head_idx) >= 10):  # 时间间距

                    # 计算颈线（连接左右肩之间的最低点）
                    neckline_points = []
                    if i - 1 < len(all_points) and all_points[i - 1][2] == 'trough':
                        neckline_points.append(all_points[i - 1])
                    if i + 1 < len(all_points) and all_points[i + 1][2] == 'trough':
                        neckline_points.append(all_points[i + 1])

                    if len(neckline_points) >= 2:
                        neckline_left_idx, neckline_left_price, _ = neckline_points[0]
                        neckline_right_idx, neckline_right_price, _ = neckline_points[1]

                        # 计算颈线斜率和水平
                        neckline_slope = (neckline_right_price - neckline_left_price) / (
                                    neckline_right_idx - neckline_left_idx)

                        # 寻找颈线突破点（在右肩之后，收盘价跌破颈线）
                        breakout_idx = -1
                        for j in range(right_idx + 1, min(right_idx + 50, len(closes))):
                            if j >= len(closes):
                                break

                            # 计算该点的颈线水平
                            neckline_level = neckline_left_price + neckline_slope * (j - neckline_left_idx)

                            if closes[j] < neckline_level - head_height * 0.5:  # 突破颈线以下一定幅度
                                breakout_idx = j
                                break

                        # 计算形态置信度
                        confidence = self.calculate_pattern_confidence(
                            left_price, head_price, right_price,
                            neckline_left_price, neckline_right_price,
                            head_height_pct, shoulder_diff_pct,
                            head_idx - left_idx, right_idx - head_idx
                        )

                        if confidence > 0.6:  # 置信度阈值
                            pattern = {
                                'type': 'head_shoulders',
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
                                'confidence': confidence,
                                'is_inverse': False  # 头肩顶（常规）
                            }
                            patterns.append(pattern)

        # 检测头肩底形态（头肩形态的反向）
        for i in range(2, len(all_points) - 2):
            # 检查是否符合头肩底模式: 左肩 > 头 < 右肩
            if (all_points[i - 2][2] == 'trough' and  # 左肩是低点
                    all_points[i][2] == 'trough' and  # 头是低点
                    all_points[i + 2][2] == 'trough'):  # 右肩是低点

                left_idx, left_price, _ = all_points[i - 2]
                head_idx, head_price, _ = all_points[i]
                right_idx, right_price, _ = all_points[i + 2]

                # 计算形态参数
                head_depth = max(left_price, right_price) - head_price
                head_depth_pct = head_depth / head_price * 100

                shoulder_diff_pct = abs(left_price - right_price) / ((left_price + right_price) / 2) * 100

                # 形态条件:
                # 1. 头必须低于两个肩膀
                # 2. 两个肩膀高度大致相等
                # 3. 头的深度至少为价格的0.2%
                # 4. 时间间距合理
                if (head_price < left_price and
                        head_price < right_price and
                        shoulder_diff_pct < 1.0 and
                        head_depth_pct >= 0.2 and
                        min(head_idx - left_idx, right_idx - head_idx) >= 10):

                    # 计算颈线（连接左右肩之间的最高点）
                    neckline_points = []
                    if i - 1 < len(all_points) and all_points[i - 1][2] == 'peak':
                        neckline_points.append(all_points[i - 1])
                    if i + 1 < len(all_points) and all_points[i + 1][2] == 'peak':
                        neckline_points.append(all_points[i + 1])

                    if len(neckline_points) >= 2:
                        neckline_left_idx, neckline_left_price, _ = neckline_points[0]
                        neckline_right_idx, neckline_right_price, _ = neckline_points[1]

                        # 计算颈线斜率和水平
                        neckline_slope = (neckline_right_price - neckline_left_price) / (
                                    neckline_right_idx - neckline_left_idx)

                        # 寻找颈线突破点（在右肩之后，收盘价突破颈线）
                        breakout_idx = -1
                        for j in range(right_idx + 1, min(right_idx + 50, len(closes))):
                            if j >= len(closes):
                                break

                            # 计算该点的颈线水平
                            neckline_level = neckline_left_price + neckline_slope * (j - neckline_left_idx)

                            if closes[j] > neckline_level + head_depth * 0.5:  # 突破颈线以上一定幅度
                                breakout_idx = j
                                break

                        # 计算形态置信度
                        confidence = self.calculate_pattern_confidence(
                            left_price, head_price, right_price,
                            neckline_left_price, neckline_right_price,
                            head_depth_pct, shoulder_diff_pct,
                            head_idx - left_idx, right_idx - head_idx
                        )

                        if confidence > 0.6:  # 置信度阈值
                            pattern = {
                                'type': 'head_shoulders',
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
                                'confidence': confidence,
                                'is_inverse': True  # 头肩底（反向）
                            }
                            patterns.append(pattern)

        return patterns

    def calculate_pattern_confidence(self, left_price: float, head_price: float, right_price: float,
                                     neckline_left: float, neckline_right: float,
                                     pattern_height_pct: float, shoulder_diff_pct: float,
                                     left_to_head_time: int, head_to_right_time: int) -> float:
        """
        计算头肩形态的置信度 - 针对黄金1分钟数据优化的宽松版本

        Returns:
            置信度得分 0-1
        """
        confidence = 0.0

        # 1. 头肩高度/深度 (25%) - 降低要求
        if pattern_height_pct > 0.3:  # 原为0.5
            confidence += 0.25
        elif pattern_height_pct > 0.15:  # 原为0.2
            confidence += 0.15
        elif pattern_height_pct > 0.08:  # 原为0.1
            confidence += 0.08

        # 2. 左右肩对称性 (20%) - 降低要求
        if shoulder_diff_pct < 1.0:  # 原为0.5
            confidence += 0.20
        elif shoulder_diff_pct < 2.0:  # 原为1.0
            confidence += 0.12
        elif shoulder_diff_pct < 3.0:  # 新增
            confidence += 0.05

        # 3. 时间对称性 (20%) - 更宽松
        if left_to_head_time > 0 and head_to_right_time > 0:
            time_ratio = min(left_to_head_time, head_to_right_time) / max(left_to_head_time, head_to_right_time)
            if time_ratio > 0.6:  # 原为0.8
                confidence += 0.20
            elif time_ratio > 0.4:  # 原为0.6
                confidence += 0.12
            elif time_ratio > 0.3:  # 原为0.4
                confidence += 0.05

        # 4. 颈线质量 (15%) - 降低要求
        if neckline_left > 0 and neckline_right > 0:
            neckline_diff_pct = abs(neckline_right - neckline_left) / ((neckline_right + neckline_left) / 2) * 100
            if neckline_diff_pct < 1.0:  # 原为0.5
                confidence += 0.15
            elif neckline_diff_pct < 2.0:  # 原为1.0
                confidence += 0.10
            elif neckline_diff_pct < 3.0:  # 原为2.0
                confidence += 0.05

        # 5. 形态大小 (20%) - 改为更重要的因子，降低时间要求
        min_time = min(left_to_head_time, head_to_right_time)
        if min_time >= 15:  # 原为20
            confidence += 0.20
        elif min_time >= 10:  # 原为15
            confidence += 0.12
        elif min_time >= 8:  # 原为10
            confidence += 0.08
        elif min_time >= 5:  # 新增，允许更小的形态
            confidence += 0.04

        return min(confidence, 1.0)

    def detect_all_patterns(self):
        """检测所有头肩形态"""
        if not self.data_loaded:
            print("错误: 数据未加载")
            return False

        print("开始检测头肩形态...")

        try:
            # 提取价格数据
            df = self.price_data
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values

            # 检测头肩形态
            patterns = self.find_head_shoulders_patterns(highs, lows, closes)

            # 过滤重复和重叠的形态
            filtered_patterns = self.filter_overlapping_patterns(patterns)

            self.hs_patterns = filtered_patterns

            print(f"✓ 检测到 {len(filtered_patterns)} 个头肩形态")

            if filtered_patterns:
                print("形态统计:")
                regular_count = len([p for p in filtered_patterns if not p['is_inverse']])
                inverse_count = len([p for p in filtered_patterns if p['is_inverse']])
                print(f"  头肩顶: {regular_count}")
                print(f"  头肩底: {inverse_count}")

                avg_confidence = np.mean([p['confidence'] for p in filtered_patterns])
                print(f"  平均置信度: {avg_confidence:.2f}")

            return True

        except Exception as e:
            print(f"形态检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def filter_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """过滤重叠的头肩形态"""
        if not patterns:
            return []

        # 按置信度排序
        patterns.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        used_indices = set()

        for pattern in patterns:
            # 获取形态涉及的所有索引
            pattern_indices = {
                pattern['left_shoulder_idx'],
                pattern['head_idx'],
                pattern['right_shoulder_idx']
            }
            if pattern['neckline_break_idx'] > 0:
                pattern_indices.add(pattern['neckline_break_idx'])

            # 检查是否有重叠
            overlap = False
            for idx in pattern_indices:
                if idx in used_indices:
                    overlap = True
                    break

            # 如果没有重叠，添加该形态
            if not overlap:
                filtered.append(pattern)
                used_indices.update(pattern_indices)

        return filtered

    def create_labels(self) -> pd.DataFrame:
        """创建标签DataFrame"""
        if not self.hs_patterns:
            print("没有检测到头肩形态")
            return None

        print("创建标签...")

        df = self.price_data
        labels = [''] * len(df)

        for i, pattern in enumerate(self.hs_patterns):
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

            # 标记颈线突破点
            break_idx = pattern['neckline_break_idx']
            if break_idx > 0 and break_idx < len(labels):
                labels[break_idx] = f"{pattern_id}_颈线突破"

            # 在形态描述中添加置信度和类型信息
            pattern_type = "顶" if not pattern['is_inverse'] else "底"
            print(f"  形态 {pattern_id}: {pattern_type}形, "
                  f"置信度={pattern['confidence']:.2f}, "
                  f"幅度={pattern['pattern_height']:.2f}, "
                  f"宽度={pattern['pattern_width']}分钟")

        # 创建标签DataFrame
        result_df = pd.DataFrame({
            'rowid': df['rowid'].values,
            'HS_Pattern_Label': labels
        })

        labeled_count = sum(1 for label in labels if label)
        print(f"✓ 已标记 {labeled_count} 个关键点")

        return result_df

    def update_database_labels(self) -> bool:
        """更新数据库中的标签"""
        try:
            # 添加标签列
            if not self.add_label_column():
                return False

            # 创建标签
            labels_df = self.create_labels()
            if labels_df is None:
                print("没有标签需要更新")
                return True

            # 批量更新数据库
            print("正在更新数据库标签...")

            batch_size = 1000
            total_rows = len(labels_df)

            with tqdm(total=total_rows, desc="更新数据库") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch = labels_df.iloc[i:i + batch_size]

                    params = []
                    for _, row in batch.iterrows():
                        params.append((
                            row['HS_Pattern_Label'] if pd.notna(row['HS_Pattern_Label']) else '',
                            row['rowid']
                        ))

                    update_query = f"""
                    UPDATE {self.table_name}
                    SET HS_Pattern_Label = ?
                    WHERE rowid = ?
                    """

                    self.cursor.executemany(update_query, params)
                    self.conn.commit()

                    pbar.update(len(batch))

            print("✓ 数据库标签更新完成")
            return True

        except Exception as e:
            print(f"更新数据库失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_detection(self) -> bool:
        """运行完整的头肩形态检测流程"""
        print("=" * 60)
        print("头肩形态检测器 - 1分钟数据")
        print("=" * 60)

        try:
            # 1. 加载数据
            if not self.load_price_data():
                return False

            # 2. 检测头肩形态
            if not self.detect_all_patterns():
                return False

            # 3. 更新数据库标签
            if not self.update_database_labels():
                return False

            # 4. 显示检测结果
            self.display_detection_results()

            print("\n" + "=" * 60)
            print("头肩形态检测完成！")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"检测流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.close()

    def display_detection_results(self):
        """显示检测结果"""
        if not self.hs_patterns:
            print("未检测到头肩形态")
            return

        df = self.price_data

        print("\n检测结果摘要:")
        print("-" * 60)

        for i, pattern in enumerate(self.hs_patterns[:5]):  # 只显示前5个
            pattern_id = f"HS_{i + 1}"
            pattern_type = "头肩顶" if not pattern['is_inverse'] else "头肩底"

            left_time = df.iloc[pattern['left_shoulder_idx']]['Time'] if 'Time' in df.columns else pattern[
                'left_shoulder_idx']
            head_time = df.iloc[pattern['head_idx']]['Time'] if 'Time' in df.columns else pattern['head_idx']
            right_time = df.iloc[pattern['right_shoulder_idx']]['Time'] if 'Time' in df.columns else pattern[
                'right_shoulder_idx']

            print(f"形态 {pattern_id} ({pattern_type}):")
            print(f"  左肩: 时间={left_time}, 价格={pattern['left_shoulder_price']:.2f}")
            print(f"  头  : 时间={head_time}, 价格={pattern['head_price']:.2f}")
            print(f"  右肩: 时间={right_time}, 价格={pattern['right_shoulder_price']:.2f}")
            print(f"  颈线: {pattern['neckline_left_price']:.2f} -> {pattern['neckline_right_price']:.2f}")

            if pattern['neckline_break_idx'] > 0:
                break_time = df.iloc[pattern['neckline_break_idx']]['Time'] if 'Time' in df.columns else pattern[
                    'neckline_break_idx']
                print(f"  突破: 时间={break_time}")

            print(f"  置信度: {pattern['confidence']:.2%}")
            print(f"  幅度: {pattern['pattern_height']:.2f}, 宽度: {pattern['pattern_width']}分钟")
            print("-" * 40)

        # 统计信息
        total_patterns = len(self.hs_patterns)
        regular_patterns = len([p for p in self.hs_patterns if not p['is_inverse']])
        inverse_patterns = len([p for p in self.hs_patterns if p['is_inverse']])

        print(f"\n统计信息:")
        print(f"  总形态数: {total_patterns}")
        print(f"  头肩顶数: {regular_patterns}")
        print(f"  头肩底数: {inverse_patterns}")

        if total_patterns > 0:
            avg_confidence = np.mean([p['confidence'] for p in self.hs_patterns])
            print(f"  平均置信度: {avg_confidence:.2%}")

            avg_height = np.mean([p['pattern_height'] for p in self.hs_patterns])
            print(f"  平均幅度: {avg_height:.2f}")

            avg_width = np.mean([p['pattern_width'] for p in self.hs_patterns])
            print(f"  平均宽度: {avg_width:.1f}分钟")

        # 查询示例
        print(f"\n数据库查询示例:")
        print(
            f"  SELECT Time, Close, HS_Pattern_Label FROM {self.table_name} WHERE HS_Pattern_Label LIKE '%左肩%' LIMIT 5;")
        print(
            f"  SELECT Time, Close, HS_Pattern_Label FROM {self.table_name} WHERE HS_Pattern_Label LIKE '%颈线突破%' LIMIT 5;")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    DB_PATH = "./identifier.sqlite"  # 数据库路径
    TABLE_NAME = "PriceData"  # 表名

    print("头肩形态检测器")
    print("=" * 60)

    # 创建检测器实例
    detector = HeadShouldersDetector(DB_PATH, TABLE_NAME)

    # 运行检测
    success = detector.run_detection()

    if success:
        print("\n检测成功！数据库已更新。")
        print("您可以使用以下SQL查询查看结果:")
        print(f"  SELECT Time, Close, HS_Pattern_Label FROM {TABLE_NAME} WHERE HS_Pattern_Label != '' ORDER BY Time;")
    else:
        print("\n检测失败，请检查错误信息。")

    print("\n程序结束。")
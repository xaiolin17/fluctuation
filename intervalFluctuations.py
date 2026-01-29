import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm


class PriceDataCalculatorOptimized:
    def __init__(self, _db, table_name='Price'):
        self.db_path = _db
        self.table_name = table_name
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()

    def create_calculated_columns(self):
        columns_to_add = [
            ("IntervalFluctuations5m", "REAL"),
            ("IntervalFluctuations15m", "REAL"),
            ("IntervalFluctuations1h", "REAL"),
            ("IntervalFluctuationsPeak5m", "REAL"),
            ("IntervalFluctuationsPeak15m", "REAL"),
            ("IntervalFluctuationsPeak1h", "REAL")
        ]

        for col_name, col_type in columns_to_add:
            try:
                check_query = f"PRAGMA table_info({self.table_name})"
                self.cursor.execute(check_query)
                existing_columns = [col[1] for col in self.cursor.fetchall()]

                if col_name not in existing_columns:
                    alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {col_name} {col_type}"
                    self.cursor.execute(alter_query)
                    print(f"已添加列: {col_name}")
            except Exception as e:
                print(f"添加列 {col_name} 时出错: {e}")

        self.conn.commit()

    def calculate_fluctuations_vectorized(self):
        """使用向量化操作计算波动率"""
        print("读取数据...")
        query = f"SELECT rowid, Open, High, Low, Close FROM {self.table_name} ORDER BY Time"
        df = pd.read_sql_query(query, self.conn)

        if len(df) < 60:  # 至少需要60行来计算1小时窗口
            print("数据不足，至少需要60行数据")
            return

        print("使用向量化计算波动率指标...")

        # 转换为numpy数组
        opens = df['Open'].values
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        rowids = df['rowid'].values

        # 初始化结果数组
        n = len(df)
        results = {
            'rowid': rowids,
            'fluct_5m': np.full(n, np.nan),
            'fluct_15m': np.full(n, np.nan),
            'fluct_1h': np.full(n, np.nan),
            'peak_5m': np.full(n, np.nan),
            'peak_15m': np.full(n, np.nan),
            'peak_1h': np.full(n, np.nan)
        }

        # 计算各窗口的指标
        windows = [
            (5, 'fluct_5m', 'peak_5m'),
            (15, 'fluct_15m', 'peak_15m'),
            (60, 'fluct_1h', 'peak_1h')
        ]

        for window_size, fluct_col, peak_col in windows:
            print(f"计算 {window_size} 分钟窗口...")

            # 为每个窗口创建滑动窗口索引
            for i in tqdm(range(window_size - 1, n), desc=f"{window_size}分钟"):
                start_idx = i - (window_size - 1)

                # 普通波动率
                if not np.isnan(opens[start_idx]):
                    results[fluct_col][i] = round(
                        (opens[start_idx] - closes[i]) / opens[start_idx], 8
                    )

                # 峰值波动率
                window_highs = highs[start_idx:i + 1]
                window_lows = lows[start_idx:i + 1]

                if not np.isnan(opens[start_idx]) and len(window_highs) > 0 and len(window_lows) > 0:
                    max_high = np.max(window_highs)
                    min_low = np.min(window_lows)
                    results[peak_col][i] = round(
                        (max_high - min_low) / opens[start_idx], 8
                    )

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        # 更新数据库
        print("更新数据库...")
        self._batch_update_database(results_df)

    def _batch_update_database(self, results_df):
        """批量更新数据库"""
        # 分批处理，避免内存问题
        batch_size = 10000
        n_batches = (len(results_df) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="批量更新"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(results_df))
            batch = results_df.iloc[start_idx:end_idx]

            # 创建参数列表
            params = []
            for _, row in batch.iterrows():
                params.append((
                    row['fluct_5m'], row['fluct_15m'], row['fluct_1h'],
                    row['peak_5m'], row['peak_15m'], row['peak_1h'],
                    row['rowid']
                ))

            # 批量执行UPDATE
            update_query = f"""
            UPDATE {self.table_name}
            SET 
                IntervalFluctuations5m = ?,
                IntervalFluctuations15m = ?,
                IntervalFluctuations1h = ?,
                IntervalFluctuationsPeak5m = ?,
                IntervalFluctuationsPeak15m = ?,
                IntervalFluctuationsPeak1h = ?
            WHERE rowid = ?
            """

            self.cursor.executemany(update_query, params)
            self.conn.commit()

    def calculate_all(self):
        try:
            self.connect()
            print("已连接到数据库")

            print("检查并添加计算列...")
            self.create_calculated_columns()

            print("开始计算波动率指标...")
            self.calculate_fluctuations_vectorized()

            print("计算完成！")

        except Exception as e:
            print(f"计算过程中出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.close()


# 使用示例
if __name__ == "__main__":
    db_path = "./identifier.sqlite"
    calculator = PriceDataCalculatorOptimized(db_path, table_name='PriceData')
    calculator.calculate_all()
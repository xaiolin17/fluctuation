import sqlite3
import pandas as pd
from tqdm import tqdm


class PriceDataCalculator:
    def __init__(self, _db, table_name='Price'):
        """
        初始化计算器

        Args:
            _db: 数据库文件路径
            table_name: 表名
        """
        self.db_path = _db
        self.table_name = table_name
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接到数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def get_total_rows(self):
        """获取总行数"""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    def create_calculated_columns(self):
        """在数据库中添加计算列（如果不存在）"""
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
                # 检查列是否存在
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

    def calculate_fluctuations(self, batch_size=5000):
        """
        批量计算波动率指标

        Args:
            batch_size: 每批处理的行数
        """
        # 获取总行数用于进度条
        total_rows = self.get_total_rows()

        # 使用窗口函数分批计算
        offset = 0
        with tqdm(total=total_rows, desc="计算波动率") as pbar:
            while offset < total_rows:
                # 获取当前批次数据
                query = f"""
                SELECT rowid, Time, Open, High, Low, Close FROM {self.table_name}
                ORDER BY Time
                LIMIT {batch_size} OFFSET {offset}
                """
                df = pd.read_sql_query(query, self.conn)

                if df.empty:
                    break

                # 为每个批次创建临时表并填充
                temp_table_name = f"temp_{self.table_name}"
                df.to_sql(temp_table_name, self.conn, if_exists='replace', index=False)

                # 在临时表中添加计算列
                self._add_columns_to_temp_table(temp_table_name)

                # 使用窗口函数计算
                self._calculate_window_metrics(temp_table_name)

                # 更新主表
                self._update_main_table(temp_table_name, offset)

                # 删除临时表
                self.cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

                offset += batch_size
                pbar.update(len(df))

        self.conn.commit()

    def _add_columns_to_temp_table(self, temp_table_name):
        """在临时表中添加计算列"""
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
                # 检查列是否已存在
                check_query = f"PRAGMA table_info({temp_table_name})"
                self.cursor.execute(check_query)
                existing_columns = [col[1] for col in self.cursor.fetchall()]

                if col_name not in existing_columns:
                    # 添加列
                    alter_query = f"ALTER TABLE {temp_table_name} ADD COLUMN {col_name} {col_type}"
                    self.cursor.execute(alter_query)
            except Exception as e:
                print(f"在临时表添加列 {col_name} 时出错: {e}")

        self.conn.commit()

    def _calculate_window_metrics(self, temp_table_name):
        """使用窗口函数计算指标"""
        # 获取临时表中的所有行ID
        select_query = f"SELECT rowid FROM {temp_table_name} ORDER BY rowid"
        self.cursor.execute(select_query)
        rows = self.cursor.fetchall()

        # 逐行计算并更新
        for row in rows:
            rowid = row[0]

            # 计算5分钟窗口
            if rowid >= 5:
                # 计算普通波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (first.Open - last.Close) * 1.0 / first.Open,
                        8
                    )
                    FROM (
                        SELECT Open, Close
                        FROM {temp_table_name}
                        WHERE rowid = ?
                    ) AS last
                    CROSS JOIN (
                        SELECT Open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 4
                    ) AS first
                """, (rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuations5m = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

                # 计算5分钟峰值波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (max_high - min_low) * 1.0 / first_open,
                        8
                    )
                    FROM (
                        SELECT MAX(High) as max_high, MIN(Low) as min_low
                        FROM {temp_table_name}
                        WHERE rowid BETWEEN ? - 4 AND ?
                    ) AS max_min
                    CROSS JOIN (
                        SELECT Open as first_open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 4
                    ) AS first
                """, (rowid, rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuationsPeak5m = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

            # 计算15分钟窗口
            if rowid >= 15:
                # 计算普通波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (first.Open - last.Close) * 1.0 / first.Open,
                        8
                    )
                    FROM (
                        SELECT Open, Close
                        FROM {temp_table_name}
                        WHERE rowid = ?
                    ) AS last
                    CROSS JOIN (
                        SELECT Open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 14
                    ) AS first
                """, (rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuations15m = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

                # 计算15分钟峰值波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (max_high - min_low) * 1.0 / first_open,
                        8
                    )
                    FROM (
                        SELECT MAX(High) as max_high, MIN(Low) as min_low
                        FROM {temp_table_name}
                        WHERE rowid BETWEEN ? - 14 AND ?
                    ) AS max_min
                    CROSS JOIN (
                        SELECT Open as first_open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 14
                    ) AS first
                """, (rowid, rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuationsPeak15m = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

            # 计算1小时窗口
            if rowid >= 60:
                # 计算普通波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (first.Open - last.Close) * 1.0 / first.Open,
                        8
                    )
                    FROM (
                        SELECT Open, Close
                        FROM {temp_table_name}
                        WHERE rowid = ?
                    ) AS last
                    CROSS JOIN (
                        SELECT Open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 59
                    ) AS first
                """, (rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuations1h = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

                # 计算1小时峰值波动率
                self.cursor.execute(f"""
                    SELECT ROUND(
                        (max_high - min_low) * 1.0 / first_open,
                        8
                    )
                    FROM (
                        SELECT MAX(High) as max_high, MIN(Low) as min_low
                        FROM {temp_table_name}
                        WHERE rowid BETWEEN ? - 59 AND ?
                    ) AS max_min
                    CROSS JOIN (
                        SELECT Open as first_open
                        FROM {temp_table_name}
                        WHERE rowid = ? - 59
                    ) AS first
                """, (rowid, rowid, rowid))

                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    self.cursor.execute(f"""
                        UPDATE {temp_table_name}
                        SET IntervalFluctuationsPeak1h = ?
                        WHERE rowid = ?
                    """, (result[0], rowid))

    def _update_main_table(self, temp_table_name, offset):
        """将计算结果更新到主表"""
        # 获取临时表中的计算结果
        query = f"""
        SELECT rowid, 
               IntervalFluctuations5m,
               IntervalFluctuations15m,
               IntervalFluctuations1h,
               IntervalFluctuationsPeak5m,
               IntervalFluctuationsPeak15m,
               IntervalFluctuationsPeak1h
        FROM {temp_table_name}
        """

        df_results = pd.read_sql_query(query, self.conn)

        # 更新主表
        for _, row in df_results.iterrows():
            main_rowid = offset + row['rowid']

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

            self.cursor.execute(update_query, (
                row['IntervalFluctuations5m'],
                row['IntervalFluctuations15m'],
                row['IntervalFluctuations1h'],
                row['IntervalFluctuationsPeak5m'],
                row['IntervalFluctuationsPeak15m'],
                row['IntervalFluctuationsPeak1h'],
                main_rowid
            ))

    def calculate_all(self):
        """执行所有计算"""
        try:
            self.connect()
            print("已连接到数据库")

            # 添加计算列
            print("检查并添加计算列...")
            self.create_calculated_columns()

            # 计算指标
            print("开始计算波动率指标...")
            self.calculate_fluctuations(batch_size=5000)

            print("计算完成！")

        except Exception as e:
            print(f"计算过程中出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.close()


# 使用示例
if __name__ == "__main__":
    # 配置数据库路径
    db_path = "./identifier.sqlite"

    # 创建计算器并执行计算
    calculator = PriceDataCalculator(db_path, table_name='Price')
    calculator.calculate_all()
import numpy as np

def zigzag_improved(df, depth=20, deviation=0.2, backstep=5):
    """ZigZag

    参数：
    - depth: 深度，数量
    - deviation: 最小变化百分比（默认0.2%）
    - backstep: 回退步数
    """
    df = df.copy()
    n = len(df)

    # 初始化
    zz_pivots = np.zeros(n)  # 0:无，1:高点，2:低点
    zz_values = np.zeros(n)  # 存储对应价格

    # 第一个点
    last_pivot_idx = -1
    last_pivot_value = 0
    last_pivot_type = 0  # 0:无，1:高点，2:低点

    i = depth
    while i < n - depth:
        # 寻找潜在的高点
        if last_pivot_type != 1:  # 上一个不是高点
            # 在当前窗口内寻找最高点
            window = df['High'].iloc[i - depth:i + depth + 1]
            max_idx = window.idxmax()
            max_val = window.max()

            # 检查是否满足偏离条件
            if last_pivot_type == 2:  # 上一个点是低点
                price_change = (max_val - last_pivot_value) / last_pivot_value
                if price_change > deviation / 100 and max_idx > last_pivot_idx:
                    zz_pivots[max_idx] = 1
                    zz_values[max_idx] = max_val
                    last_pivot_idx = max_idx
                    last_pivot_value = max_val
                    last_pivot_type = 1
                    i = max_idx + backstep
                    continue
            elif last_pivot_type == 0:  # 第一个点
                zz_pivots[max_idx] = 1
                zz_values[max_idx] = max_val
                last_pivot_idx = max_idx
                last_pivot_value = max_val
                last_pivot_type = 1
                i = max_idx + backstep
                continue

        # 寻找潜在的低点
        if last_pivot_type != 2:  # 上一个不是低点
            # 在当前窗口内寻找最低点
            window = df['Low'].iloc[i - depth:i + depth + 1]
            min_idx = window.idxmin()
            min_val = window.min()

            # 检查是否满足偏离条件
            if last_pivot_type == 1:  # 上一个点是高点
                price_change = (last_pivot_value - min_val) / last_pivot_value
                if price_change > deviation / 100 and min_idx > last_pivot_idx:
                    zz_pivots[min_idx] = 2
                    zz_values[min_idx] = min_val
                    last_pivot_idx = min_idx
                    last_pivot_value = min_val
                    last_pivot_type = 2
                    i = min_idx + backstep
                    continue
            elif last_pivot_type == 0:  # 第一个点
                zz_pivots[min_idx] = 2
                zz_values[min_idx] = min_val
                last_pivot_idx = min_idx
                last_pivot_value = min_val
                last_pivot_type = 2
                i = min_idx + backstep
                continue

        i += 1

    # 应用标签
    df['Lable'] = zz_pivots

    # 后处理：确保交替出现高点和低点
    df = ensure_alternating_pivots(df)

    return df


def ensure_alternating_pivots(df):
    """
    确保高点和低点交替出现
    """
    labels = df['Lable'].values
    n = len(labels)

    # 找到所有标记点
    pivot_indices = np.where(labels != 0)[0]

    if len(pivot_indices) < 2:
        return df

    # 检查并修复连续的相同类型标记
    for i in range(1, len(pivot_indices)):
        idx1 = pivot_indices[i - 1]
        idx2 = pivot_indices[i]

        # 如果连续两个标记类型相同
        if labels[idx1] == labels[idx2]:
            # 移除变化较小的那个
            if labels[idx1] == 1:  # 都是高点
                if df['High'].iloc[idx1] > df['High'].iloc[idx2]:
                    labels[idx2] = 0
                else:
                    labels[idx1] = 0
            else:  # 都是低点
                if df['Low'].iloc[idx1] < df['Low'].iloc[idx2]:
                    labels[idx2] = 0
                else:
                    labels[idx1] = 0

    df['Lable'] = labels
    return df
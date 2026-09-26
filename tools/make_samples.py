"""生成 ``samples/`` 目录下的示例 CSV 数据。

运行::

    .venv\\Scripts\\python.exe tools\\make_samples.py

所有文件都以 ``utf-8-sig`` 保存，方便 Excel 直接双击打开而不乱码。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "samples")
RNG = np.random.default_rng(20240607)


def sensor_timeseries() -> pd.DataFrame:
    """单通道传感器时序：适合折线图 / 面积图 / 误差棒。"""
    t = np.linspace(0.0, 24.0, 240)
    temperature = 22 + 6 * np.sin(2 * np.pi * t / 24 - 1.1) + RNG.normal(0, 0.25, t.size)
    humidity = 55 + 18 * np.cos(2 * np.pi * t / 24 + 0.4) + RNG.normal(0, 0.8, t.size)
    pressure = 101.3 + 1.4 * np.sin(2 * np.pi * t / 12) + RNG.normal(0, 0.12, t.size)
    voltage = 3.3 * np.exp(-t / 9.0) * np.sin(2 * np.pi * t / 3.0) + RNG.normal(0, 0.02, t.size)
    return pd.DataFrame(
        {
            "时间(h)": np.round(t, 3),
            "温度(℃)": np.round(temperature, 3),
            "湿度(%)": np.round(humidity, 3),
            "气压(kPa)": np.round(pressure, 4),
            "电压(V)": np.round(voltage, 5),
            "温度误差": np.round(np.full(t.size, 0.25), 3),
        }
    )


def monthly_sales() -> pd.DataFrame:
    """离散分类数据：适合柱形图 / 条形图 / 饼图。"""
    months = [f"{m}月" for m in range(1, 13)]
    base = np.array([82, 74, 96, 108, 121, 134, 142, 138, 126, 118, 104, 152], dtype=float)
    return pd.DataFrame(
        {
            "月份": months,
            "华东": np.round(base * 1.00 + RNG.normal(0, 4, 12), 1),
            "华北": np.round(base * 0.78 + RNG.normal(0, 4, 12), 1),
            "华南": np.round(base * 0.62 + RNG.normal(0, 3, 12), 1),
            "目标": np.round(base * 0.9, 1),
        }
    )


def vector_field() -> pd.DataFrame:
    """二维向量场：x、y 为起点，u、v 为分量，可直接画向量图。"""
    xs = np.linspace(-2.0, 2.0, 15)
    ys = np.linspace(-2.0, 2.0, 15)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.ravel()
    y = grid_y.ravel()
    u = -y
    v = x
    return pd.DataFrame(
        {
            "x": np.round(x, 4),
            "y": np.round(y, 4),
            "u": np.round(u, 4),
            "v": np.round(v, 4),
            "强度": np.round(np.hypot(u, v), 4),
        }
    )


def surface_3d() -> pd.DataFrame:
    """规则网格三维数据：适合曲面 / 线框 / 三维等高线。"""
    xs = np.linspace(-3.0, 3.0, 60)
    ys = np.linspace(-3.0, 3.0, 60)
    grid_x, grid_y = np.meshgrid(xs, ys)
    r = np.hypot(grid_x, grid_y)
    z = np.sin(r) / (r + 0.35) * 3.0
    return pd.DataFrame(
        {
            "x": np.round(grid_x.ravel(), 4),
            "y": np.round(grid_y.ravel(), 4),
            "z": np.round(z.ravel(), 6),
        }
    )


def student_scores() -> pd.DataFrame:
    """分组样本：适合箱线图 / 直方图 / 散点图。"""
    groups = ["一班", "二班", "三班", "四班"]
    rows: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        count = 28 + index * 3
        chinese = np.clip(RNG.normal(88 - index * 2, 7.5, count), 40, 100)
        maths = np.clip(RNG.normal(92 - index * 3.5, 9.0, count), 35, 100)
        english = np.clip(RNG.normal(85 + index, 8.0, count), 40, 100)
        for i in range(count):
            rows.append(
                {
                    "班级": group,
                    "语文": round(float(chinese[i]), 1),
                    "数学": round(float(maths[i]), 1),
                    "英语": round(float(english[i]), 1),
                    "学号": f"{index + 1}-{i + 1:02d}",
                }
            )
    return pd.DataFrame(rows)


def matrix_sample() -> pd.DataFrame:
    """矩阵样表：用于「从数据集列导入矩阵」。"""
    values = np.array(
        [
            [4.0, -2.0, 1.0],
            [3.0, 6.0, -1.0],
            [2.0, 1.0, 8.0],
        ]
    )
    return pd.DataFrame(values, columns=["列1", "列2", "列3"])


def damping_curve() -> pd.DataFrame:
    """带 LaTeX 公式含义的曲线族：适合演示公式标题。"""
    t = np.linspace(0.0, 20.0, 400)
    frame: dict[str, np.ndarray] = {"t": np.round(t, 4)}
    for zeta in (0.05, 0.15, 0.3):
        omega = 2.0
        omega_d = omega * np.sqrt(1 - zeta**2)
        frame[f"ζ={zeta:g}"] = np.round(np.exp(-zeta * omega * t) * np.cos(omega_d * t), 6)
    return pd.DataFrame(frame)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    datasets = {
        "传感器时序.csv": sensor_timeseries(),
        "月度销售.csv": monthly_sales(),
        "二维向量场.csv": vector_field(),
        "三维曲面.csv": surface_3d(),
        "学生成绩.csv": student_scores(),
        "矩阵样例.csv": matrix_sample(),
        "阻尼振荡.csv": damping_curve(),
    }
    for name, frame in datasets.items():
        path = os.path.join(OUT, name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name:20s} {frame.shape[0]:5d} 行 × {frame.shape[1]:2d} 列  ->  {path}")
    print(f"\n共生成 {len(datasets)} 个示例文件，输出目录：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

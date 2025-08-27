import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional

# ===================== 对齐函数 =====================
@dataclass
class AlignedCurves:
    x_common: np.ndarray
    y1_on_common: np.ndarray
    y2_on_common: np.ndarray
def align_curves(
    x1, y1, x2, y2,
    grid: Literal["union", "x1", "x2", "linspace"] = "union",
    n_points: int = 1000,
    method: Literal["linear", "nearest"] = "linear",
    left: Optional[float] = np.nan,
    right: Optional[float] = np.nan,
) -> AlignedCurves:
    x1 = np.asarray(x1).astype(float)
    y1 = np.asarray(y1).astype(float)
    x2 = np.asarray(x2).astype(float)
    y2 = np.asarray(y2).astype(float)

    idx1 = np.argsort(x1)
    x1, y1 = x1[idx1], y1[idx1]
    idx2 = np.argsort(x2)
    x2, y2 = x2[idx2], y2[idx2]

    if grid == "union":
        x_common = np.unique(np.concatenate([x1, x2]))
    elif grid == "x1":
        x_common = x1.copy()
    elif grid == "x2":
        x_common = x2.copy()
    elif grid == "linspace":
        xmin = min(x1.min(), x2.min())
        xmax = max(x1.max(), x2.max())
        x_common = np.linspace(xmin, xmax, n_points)
    else:
        raise ValueError("Unknown grid option")

    if method == "linear":
        y1c = np.interp(x_common, x1, y1, left=left, right=right)
        y2c = np.interp(x_common, x2, y2, left=left, right=right)
    else:
        raise ValueError("Only linear interpolation implemented here")

    return AlignedCurves(x_common=x_common, y1_on_common=y1c, y2_on_common=y2c)


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 读取 Excel
    df = pd.read_excel("data.xlsx")

    # 假设结构：第1列 x1，第3列 y1，第4列 x2，第5列 y2
    x1 = df.iloc[:, 0].dropna().to_numpy()
    y1 = df.iloc[:, 2].dropna().to_numpy()
    x2 = df.iloc[:, 3].dropna().to_numpy()
    y2 = df.iloc[:, 4].dropna().to_numpy()

    # 对齐
    aligned = align_curves(x1, y1, x2, y2, grid="union", method="linear")

    # 保存到新 Excel
    out_df = pd.DataFrame({
        "x_common": aligned.x_common,
        "y1_aligned": aligned.y1_on_common,
        "y2_aligned": aligned.y2_on_common
    })
    out_df.to_excel("aligned_output.xlsx", index=False)

    print("对齐完成，结果已保存到 aligned_output.xlsx")


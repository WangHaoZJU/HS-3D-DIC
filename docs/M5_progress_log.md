# M5 阶段推进日志

## 记录信息

- 项目：HS-3D-DIC / StereoDIC
- 日期：2026-11-23
- 基线：M4 提交 `09e0930`
- 当前分支：`main`
- 阶段主题：不确定度传播、蒙特卡洛校验和噪声底统计

## 已完成内容

### 三角化不确定度传播

新增 `stereodic/uncertainty/propagation.py`：

- 使用原始畸变像素模型计算三角化结果对四个像素观测的数值雅可比
- 接受单个或逐点二维像素协方差
- 输出三维协方差、标准差和雅可比
- 对三角化角度、深度和无效观测保留有效掩膜

### 共享参考帧差分协方差

新增 `displacement_covariance`：

```text
Cov(U) = Cov(X_t) + Cov(X_0) - Cov(X_t, X_0) - Cov(X_0, X_t)
```

接口保留动态帧和固定参考帧之间的交叉协方差输入，便于后续使用真实重复实验估计共享误差。

### 蒙特卡洛校验

新增 `stereodic/uncertainty/monte_carlo.py`：

- 按二维像素协方差采样左右相机观测
- 重复执行非线性三角化
- 输出经验均值、协方差、标准差和有效样本数
- 支持固定随机种子，保证报告可复现

### 静态噪声底

新增 `stereodic/quality/noise_floor.py`：

- 每个 POI、每个分量的均值
- 标准差和 RMS
- 有效样本数
- NaN 和有效掩膜传播

### CLI 报告

新增：

```bash
uv run python scripts/report_uncertainty.py \
  --config configs/experiments/<name>.yaml \
  --pixel-sigma 0.01 \
  --monte-carlo 1000
```

结果写入 HDF5 的 `uncertainty/` 组，并记录 `progress` 中的阶段哈希。

## 测试结果

新增 `tests/unit/test_uncertainty.py`，覆盖：

- 一阶三角化协方差传播
- 位移差分协方差
- 蒙特卡洛标准差量级
- 静态噪声底统计

结果：

```text
3 passed
```

M5 代码建立在观测噪声协方差已知的前提上。当前尚未把相机内外参协方差、温漂、曝光模糊和同步误差纳入总协方差。

## 待完成事项

1. 从 OpenCV 标定结果导出参数协方差并传播到三维坐标。
2. 使用 E1 静态噪声实验校准二维协方差模型。
3. 使用 E2 位移台重复实验验证三维 Bias、STD、RMSE 和覆盖概率。
4. 对固定参考帧引入时间相关误差模型。
5. 在真实实验数据上输出完整不确定度预算。

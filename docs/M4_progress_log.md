# M4 阶段推进日志

## 记录信息

- 项目：HS-3D-DIC / StereoDIC
- 日期：2026-11-23
- 基线提交：`041c390`，M3 已合并到 `main`
- 当前开发分支：`main`
- 阶段目标：振幅、相位、ODS、FDD、FRF、阻尼、MAC 和刚体补偿

## 本阶段完成内容

### 1. 谐波拟合

新增 `stereodic/vibration/harmonic.py`，实现：

- 基于实际时间戳的最小二乘拟合
- 余弦和正弦系数
- 幅值和相位
- 偏置和线性趋势项
- 多次谐波配置
- 每个 POI/分量独立处理 NaN 缺测
- 拟合残差和有效样本数

相位采用约定：

```text
u(t) = A cos(omega * t + phase)
```

拟合内部对时间做中心化，输出相位已还原到原始时间零点。

### 2. ODS

新增 `stereodic/vibration/ods.py`，输出：

- 复数 ODS 响应
- 幅值场
- 相位场
- 残差场
- 有效掩膜

复数响应约定为：

```text
u(t) = Re(response * exp(i * omega * t))
```

### 3. FDD

新增 `stereodic/vibration/fdd.py`，实现输出响应的频域分解：

- 分段 FFT
- 输出谱矩阵
- 频率方向的特征值排序
- 复数模态向量
- 多通道输入，支持 `(T, N, C)` 数据展平

实现按频率逐点分解谱矩阵，避免为完整数据构造超大互谱矩阵。

### 4. FRF 和相干函数

新增 `stereodic/vibration/frf.py`，实现：

- H1 型频响函数 `G_yF / G_FF`
- 多响应通道
- 输入力缺测插值
- 相干函数
- 激励谱零点的除零保护

### 5. 阻尼和 MAC

新增：

- `stereodic/vibration/damping.py`
- `stereodic/vibration/mac.py`

支持：

- 半功率带宽阻尼比
- 单对模态 MAC
- 模态对之间的 MAC 矩阵
- 实数和复数模态向量

### 6. 刚体运动补偿

新增 `stereodic/recon/rbm.py`，实现：

- 加权 Kabsch/SVD 刚体运动拟合
- 旋转矩阵和位移向量
- 逐帧刚体残差
- `(T, N, 3)` 三维轨迹补偿
- 无效点和 NaN 传播

`stereodic/pipeline.py` 新增 `run_rbm`，`scripts/run_dic.py` 新增：

```bash
uv run python scripts/run_dic.py \
  --config configs/experiments/<name>.yaml \
  --stage rbm
```

### 7. 分析命令

扩展 `scripts/analyze.py`：

- `--frequency` 执行指定频率的谐波/ODS 拟合
- `--n-harmonics` 指定谐波数量
- `--fdd` 写入 FDD 结果
- `--force-file` 写入 FRF 和相干函数
- 自动尝试半功率阻尼估计
- `--use-rbm` 分析刚体补偿位移
- 写入 `progress` 组中的 analyze 阶段完成信息

示例：

```bash
uv run python scripts/analyze.py \
  --config configs/experiments/<name>.yaml \
  --direction 2 \
  --frequency 1000 \
  --fdd \
  --force-file force.csv
```

## 测试结果

新增 `tests/unit/test_vibration_m4.py`，包含 6 项测试：

- 不规则时间戳下的幅值和相位恢复
- ODS 复数响应约定
- FDD 主频识别
- FRF 幅值和相干函数
- 半功率阻尼和 MAC
- 刚体运动估计与轨迹补偿

测试结果：

```text
6 passed
```

M3 集成测试也加入了 `run_rbm` 阶段，结果：

```text
7 passed
```

## 当前限制

以下内容仍需要实验数据或额外环境：

1. 真实叶片与 LDV 的幅值、相位和频率对比。
2. FEM 模态向量对齐和实际 MAC 验证。
3. 基于真实宽带激励的阻尼比评估。
4. CUDA 环境上的完整振动分析性能验证。
5. `viz/animate.py` 振型和 ODS 动画。

数值模块使用合成信号和 M3 合成双目动态链路验证，不能替代 E3/E4 实验验收。

## 推荐运行顺序

```bash
uv run pytest
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage stereo
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage temporal
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage reconstruct
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage rbm
uv run python scripts/analyze.py --config configs/experiments/<name>.yaml --direction 2 --fdd
```

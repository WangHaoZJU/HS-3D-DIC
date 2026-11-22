# M3 阶段推进日志

## 记录信息

- 项目：HS-3D-DIC / StereoDIC
- 日期：2026-11-22
- 基线提交：`95872db`，M2：ChArUco 标定、二阶 IC-GN、立体匹配与三角化
- 当前分支：`main`
- Python：3.12.15
- 当前计算后端：Numba CPU

## 推进前基线

从远程仓库拉取最新 `main` 后，仓库状态为 M2 已完成。M2 相关测试通过，工作区在开始 M3 实现前保持干净。

环境复核结果如下：

- NumPy：2.5.5
- Numba：0.68.1，16 个线程
- CUDA：不可用，当前环境未安装 CuPy 且没有可用 NVIDIA GPU
- `auto` 后端：`numba`

## 本次完成内容

### 1. 固定参考时序 DIC

新增 `stereodic/dic/temporal.py`，实现：

- 参考子区只准备一次
- 每一帧相对固定参考帧求解
- 上一帧结果只作为当前帧初值
- 支持 0 阶、1 阶和 2 阶形函数配置
- ZNSSD 残差阈值和时间跳变状态
- 保持稳定的 POI ID 和状态码

### 2. 时序流水线和断点续算

扩展 `stereodic/pipeline.py`：

- 新增 `run_temporal`
- 新增 `run_reconstruct`
- 新增左右相机逐帧 DIC
- 新增 `progress/dic_frames_done`
- 新增 `progress/reconstruction_done`
- 每帧写入后 flush，支持中断后继续
- 写入左右时间戳、平均时间轴和同步时间差

扩展 `scripts/run_dic.py`，支持：

```bash
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage temporal
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage reconstruct
```

### 3. 时间同步检查

新增 `stereodic/io/sync.py` 和 `scripts/check_sync.py`，支持：

- `.npy`、CSV 和文本时间戳
- 严格递增检查
- 左右相机最近时间戳配对
- 平均同步偏差和最大同步偏差统计
- 无时间戳时按帧率生成明确的规则时间轴

### 4. 频率识别

新增 `stereodic/vibration/spectrum.py` 和 `scripts/analyze.py`，支持：

- 单边 FFT 幅值谱
- Welch 功率谱
- 主频峰值识别
- 不规则时间轴拒绝 FFT/Welch，避免掩盖丢帧问题

运行示例：

```bash
uv run python scripts/analyze.py \
  --config configs/experiments/<name>.yaml \
  --direction 2
```

### 5. CUDA 后端

新增 `stereodic/dic/backends/cuda.py`，并接入 `ICGNSolver`：

- 使用 CuPy 在 GPU 上执行固定参考 IC-GN
- 当前覆盖 M3 默认的一阶时序形函数和 0 阶平移
- 使用 FP64 保持与 NumPy/Numba 参考实现的一致性
- 目标图像样条系数在一个目标帧内复用
- 当前机器没有 GPU，因此尚未执行真实 CUDA 一致性测试

## 测试结果

### 完整测试

```text
98 passed, 1 skipped, 2 deselected
```

唯一跳过项是 CUDA 测试，原因是当前环境没有 CuPy/GPU。

### M3 集成测试

新增 `tests/integration/test_m3_temporal.py`，覆盖：

- 合成双目动态序列
- `stereo → temporal → reconstruct` 完整链路
- 左右相机时序匹配
- 逐帧三角化
- HDF5 阶段完成标记
- 二次运行时的阶段缓存复用

结果：

```text
3 passed
```

### 新增单元测试

- `tests/unit/test_temporal.py`
- `tests/unit/test_sync.py`
- `tests/unit/test_spectrum.py`

固定参考合成平移测试中，4 帧、20 个 POI 的最大二维位移误差约为 `0.00086 px`。

## 当前文件状态

本次 M3 实现已写入本地工作区，当前远程 `main` 仍停留在 `95872db`。由于当前 Git 元数据目录不可写，改动尚未提交到远程仓库。

## 待完成验证

1. 在 Linux + RTX A6000 环境执行 `uv sync --extra cuda`。
2. 运行 CUDA 后端一致性测试，确认与 NumPy/Numba 的位移差满足 P99 目标。
3. 使用真实高速相机数据验证帧号、时间戳、曝光和丢帧处理。
4. 完成 E3 刚性靶或悬臂梁振动实验。
5. 完成 E4 实际叶片频率与参考仪器对比。
6. 根据真实数据补充同步阈值、质量阈值和频率识别参数。

## 推荐运行顺序

```bash
uv run pytest
uv run python scripts/check_env.py
uv run python scripts/calibrate.py --config configs/experiments/<name>.yaml -v
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage stereo -v
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage temporal -v
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage reconstruct -v
uv run python scripts/analyze.py --config configs/experiments/<name>.yaml --direction 2
```

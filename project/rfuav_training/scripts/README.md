# 辅助脚本

## pluto_usb_benchmark.py

Pluto USB 传输能力基准测试脚本。

**依赖**：pip install pyadi-iio numpy

**用法**：
```bash
python pluto_usb_benchmark.py
```

---

## Pluto SDR 能力清单（基于 pyadi-iio 0.0.20 实测）

### 属性对照表

| collector 配置 key | pyadi-iio 0.0.20 属性 | 说明 |
|---|---|---|
| `center_frequency` | `rx_lo` | 中心频率（LO），单位 Hz |
| `sample_rate` | `sample_rate` | 采样率，单位 Hz |
| `gain` | `rx_hardwaregain_chan0` | 增益，单位 dB |
| `bandwidth` | `rx_rf_bandwidth` | 带宽，单位 Hz |

> ⚠️ 注意：`center_freq` / `gain` / `bandwidth` / `filter_bandwidth` 等属性名在 pyadi-iio 0.0.20 中不存在，使用会报错。

### 硬件能力

| 参数 | 值 |
|---|---|
| 频段范围 | 325 MHz - **5.9 GHz**（已破解）|
| 采样率上限 | 61.44 MHz，建议 **60 MHz** |
| 带宽 | `rx_rf_bandwidth`，默认值 ~18 MHz，建议设 **56 MHz** |
| 增益范围 | 0 - 60 dB，建议 **20 dB** |
| ADC 分辨率 | 12 bits |
| buffer_size | 默认 1024，建议 **2048** |

### USB 传输性能（实测）

| 指标 | 值 |
|---|---|
| 实际速率 | **14.5 MB/s**（稳定）|
| 协议 | USB 2.0 |
| 每次 rx() | 1024 samples（buffer_size=2048 时翻倍）|

### Burst 配置与等效窗口

| burst 数 | 等效窗口 | 5 频点轮询周期 | 2 秒内轮询次数 |
|---|---|---|---|
| 10 | 0.34 ms | 51.7 ms | 38.7 次 |
| 50 | 1.71 ms | 58.5 ms | 34.2 次 |
| 100 | 3.41 ms | 67.1 ms | 29.8 次 |
| 200 | 6.83 ms | 84.1 ms | 23.8 次 |
| **500** | **17.07 ms** | **135.3 ms** | **14.8 次** |

> 注：等效窗口 = burst数 × buffer_size / sample_rate

### 推荐推理配置

```yaml
collector:
  type: pluto
  uri: ip:192.168.2.1
  config:
    sample_rate: 60000000      # 60 MHz
    center_frequency: 5800000000  # 5800 MHz（可改为5760/5775/5825/5850）
    gain: 20                   # dB
    bandwidth: 56000000        # 56 MHz

inference:
  burst_count: 100             # 等效窗口 ~3.4 ms
  buffer_size: 2048
  poll_interval: 0            # 轮询间隔（ms）
```

### 训练数据采集配置

```yaml
collector:
  type: pluto
  config:
    sample_rate: 60000000
    center_frequency: 5800000000
    gain: 20
    bandwidth: 56000000

acquisition:
  burst_count: 500            # 与 scan_20260508 一致（17.07 ms 等效窗口）
  buffer_size: 2048
```

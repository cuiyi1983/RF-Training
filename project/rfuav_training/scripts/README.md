# 辅助脚本

## pluto_usb_benchmark.py

Pluto USB 传输能力基准测试脚本。

**用途**：在 PC 上（Pluto USB 直连）运行，验证实际传输能力，为轮询方案提供数据支撑。

**用法**：
```bash
python pluto_usb_benchmark.py
```

**依赖**：
```bash
pip install pyadi-iio numpy
```

**输出**：
- 不同 burst 配置下的传输速度
- 持续采集稳定性测试
- 轮询周期估算（5 频点）

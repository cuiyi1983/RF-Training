#!/usr/bin/env python3
"""
Pluto USB 传输能力基准测试
============================
测试 Pluto 通过 USB 连接时的实际数据传输能力。

用法：
    python pluto_usb_benchmark.py

输出：
  - 不同 burst 配置下的传输速度
  - 持续采集稳定性测试
  - 推荐 burst 数 / 轮询周期建议
"""

import adi
import numpy as np
import time
import sys
import os

# ===================== 配置 =====================
PLUTO_URI = None  # 自动发现，或手动指定，如 "usb:2.6.5"

# 测试配置
TEST_CONFIGS = [
    {"name": "小burst（10次）", "bursts": 10, "buffer_size": 2048},
    {"name": "中burst（50次）", "bursts": 50, "buffer_size": 2048},
    {"name": "大burst（100次）", "bursts": 100, "buffer_size": 2048},
    {"name": "超大burst（200次）", "bursts": 200, "buffer_size": 2048},
    {"name": "极限burst（500次）", "bursts": 500, "buffer_size": 2048},
]

SUSTAINED_TEST = {
    "duration_sec": 5,       # 持续采集 5 秒
    "bursts_per_read": 50,
    "buffer_size": 2048,
}

# 采样率固定
SAMPLE_RATE = 60e6  # 60 MHz

# ===================== 工具函数 =====================
def format_bytes(n):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

def measure_burst_throughput(uri, bursts, buffer_size, fs=60e6):
    """测量 burst 采集的实际吞吐量"""
    sdr = adi.Pluto(uri)
    sdr.sample_rate = int(fs)
    sdr.rx_buffer_size = buffer_size

    # 预热
    for _ in range(3):
        _ = sdr.rx()

    # 测试采集
    total_samples = 0
    times = []

    for i in range(bursts):
        t0 = time.perf_counter()
        chunk = sdr.rx()
        t1 = time.perf_counter()

        if len(chunk) == 0:
            print(f"  [!] Burst {i} 返回空数据")
            continue

        total_samples += len(chunk)
        times.append(t1 - t0)

    # 统计
    avg_time = np.mean(times)
    std_time = np.std(times)
    total_time = sum(times)
    total_bytes = total_samples * 8  # complex64 = 8 bytes

    throughput_mbps = (total_bytes / 1e6) / total_time if total_time > 0 else 0

    del sdr

    return {
        "total_samples": total_samples,
        "total_bytes": total_bytes,
        "total_time": total_time,
        "avg_burst_time": avg_time,
        "std_burst_time": std_time,
        "throughput_mbps": throughput_mbps,
    }

def sustained_acquisition_test(uri, duration_sec, bursts_per_read, buffer_size, fs=60e6):
    """持续采集稳定性测试"""
    sdr = adi.Pluto(uri)
    sdr.sample_rate = int(fs)
    sdr.rx_buffer_size = buffer_size

    # 预热
    for _ in range(3):
        _ = sdr.rx()

    samples_list = []
    start_time = time.perf_counter()
    last_report = start_time
    report_interval = 1.0  # 每秒报告一次
    total_bytes = 0
    burst_count = 0

    while time.perf_counter() - start_time < duration_sec:
        t0 = time.perf_counter()
        chunk = sdr.rx()
        t1 = time.perf_counter()

        if len(chunk) > 0:
            samples_list.append(len(chunk))
            total_bytes += len(chunk) * 8
            burst_count += 1

        # 每秒报告
        if time.perf_counter() - last_report >= report_interval:
            elapsed = time.perf_counter() - start_time
            rate_mbps = (total_bytes / 1e6) / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] 累计 {burst_count} bursts, "
                  f"{total_bytes/1e6:.1f} MB, 实时速率 {rate_mbps:.1f} MB/s")
            last_report = time.perf_counter()

    total_time = time.perf_counter() - start_time
    total_samples = sum(samples_list)
    rate_mbps = (total_bytes / 1e6) / total_time if total_time > 0 else 0

    del sdr

    return {
        "total_samples": total_samples,
        "total_bytes": total_bytes,
        "total_time": total_time,
        "total_bursts": burst_count,
        "avg_rate_mbps": rate_mbps,
    }

# ===================== 主测试 =====================
def main():
    print("=" * 60)
    print("Pluto USB 传输能力基准测试")
    print("=" * 60)

    # 连接 Pluto
    print("\n[*] 连接 Pluto...")
    if PLUTO_URI:
        uri = PLUTO_URI
        print(f"    使用指定 URI: {uri}")
    else:
        # 自动搜索
        try:
            sdr = adi.Pluto()
            uri = None
            print("    [✓] 自动发现 Pluto（USB）")
        except Exception as e:
            print(f"    [✗] 连接失败: {e}")
            sys.exit(1)

    sdr_test = adi.Pluto(uri) if uri else adi.Pluto()
    try:
        sample_rate = sdr_test.sample_rate
        print(f"    采样率: {sample_rate/1e6:.0f} MHz")
        print(f"    中心频率: {sdr_test.center_freq/1e6:.0f} MHz")
        del sdr_test
    except Exception as e:
        print(f"    [!] 无法读取部分属性: {e}")
        del sdr_test

    print("\n" + "=" * 60)
    print("Part 1: Burst 传输测试")
    print("=" * 60)

    results = []
    for cfg in TEST_CONFIGS:
        print(f"\n[*] {cfg['name']} ({cfg['bursts']} bursts × {cfg['buffer_size']} samples)...")
        r = measure_burst_throughput(uri, cfg['bursts'], cfg['buffer_size'])
        results.append((cfg, r))

        total_ms = r['total_time'] * 1000
        total_mb = r['total_bytes'] / 1e6
        rate = r['throughput_mbps']
        avg_ms = r['avg_burst_time'] * 1000

        print(f"    总耗时: {total_ms:.1f} ms")
        print(f"    总数据: {total_mb:.2f} MB ({r['total_samples']:,} samples)")
        print(f"    平均速率: {rate:.1f} MB/s ({rate*8:.1f} Mb/s)")
        print(f"    单次 burst 平均: {avg_ms:.2f} ms")

        # 估算等效窗口
        window_ms = r['total_samples'] / 60e6 * 1000
        print(f"    等效采样窗口: {window_ms:.2f} ms")

    # 汇总
    print("\n" + "-" * 40)
    print("Burst 传输汇总:")
    print(f"{'配置':<25} {'耗时':>8} {'数据量':>10} {'速率':>10} {'等效窗口':>10}")
    print("-" * 40)
    for cfg, r in results:
        label = cfg['name']
        total_ms = r['total_time'] * 1000
        total_mb = r['total_bytes'] / 1e6
        rate = r['throughput_mbps']
        window_ms = r['total_samples'] / 60e6 * 1000
        print(f"{label:<25} {total_ms:>7.1f}ms {total_mb:>9.2f}MB {rate:>9.1f}MB/s {window_ms:>9.2f}ms")

    print("\n" + "=" * 60)
    print("Part 2: 持续采集稳定性测试")
    print("=" * 60)
    print(f"\n[*] 持续采集 {SUSTAINED_TEST['duration_sec']} 秒 "
          f"({SUSTAINED_TEST['bursts_per_read']} bursts/次)...")

    r = sustained_acquisition_test(
        uri,
        SUSTAINED_TEST['duration_sec'],
        SUSTAINED_TEST['bursts_per_read'],
        SUSTAINED_TEST['buffer_size']
    )

    print(f"\n    总耗时: {r['total_time']:.2f} s")
    print(f"    总数据: {r['total_bytes']/1e6:.2f} MB")
    print(f"    总 bursts: {r['total_bursts']}")
    print(f"    平均速率: {r['avg_rate_mbps']:.2f} MB/s ({r['avg_rate_mbps']*8:.1f} Mb/s)")

    # 轮询周期估算
    print("\n" + "=" * 60)
    print("轮询周期估算（5 频点）")
    print("=" * 60)
    print(f"\n假设检测门限: 1-2 秒内发现")
    print(f"\n{'配置':<25} {'等效窗口':>10} {'5频点轮询':>12} {'2秒内轮询次数':>14}")
    print("-" * 60)

    for cfg, r in results:
        label = cfg['name']
        window_ms = r['total_samples'] / 60e6 * 1000

        # 5频点轮询：每个频点采集 + 切换开销(假设10ms)
        per_freq_ms = window_ms + 10
        cycle_ms = per_freq_ms * 5
        cycles_in_2s = 2000 / cycle_ms if cycle_ms > 0 else 999

        print(f"{label:<25} {window_ms:>9.2f}ms {cycle_ms:>11.1f}ms {cycles_in_2s:>13.1f}次")

    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)
    print("""
  - 如果目标是 1-2 秒内检测：选择等效窗口 ≥10ms 的配置（100+ bursts）
  - 如果需要更高实时性：选择等效窗口 3-5ms（50 bursts）+ 多轮轮询
  - 训练数据采集：建议用 500 bursts（~17ms 等效窗口），与 scan_20260508 一致
  - 推理采集：可适当减少 bursts数以提高轮询频率
    """)

if __name__ == "__main__":
    main()

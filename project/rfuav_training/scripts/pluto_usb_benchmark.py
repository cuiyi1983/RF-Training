#!/usr/bin/env python3
"""
Pluto USB 传输能力基准测试
============================
测试 Pluto 通过 USB 连接时的实际数据传输能力。

依赖：pip install pyadi-iio numpy

用法：
    python pluto_usb_benchmark.py

注意：
    - 运行前先确认 Pluto USB 地址（如 usb:2.6.5）
    - 脚本会自动发现网络连接，但优先使用 USB
"""

import adi
import numpy as np
import time
import sys

# ===================== 可配置参数 =====================
# 手动指定 Pluto URI（留空则自动发现）
# 格式："usb:2.6.5" 或 "ip:192.168.2.1"
PLUTO_URI = ""

# 测试配置
TEST_CONFIGS = [
    {"name": "小burst（10次）",  "bursts": 10,  "buffer_size": 2048},
    {"name": "中burst（50次）",  "bursts": 50,  "buffer_size": 2048},
    {"name": "大burst（100次）", "bursts": 100, "buffer_size": 2048},
    {"name": "超大burst（200次）","bursts": 200, "buffer_size": 2048},
    {"name": "极限burst（500次）","bursts": 500, "buffer_size": 2048},
]

SUSTAINED_TEST = {
    "duration_sec": 5,
    "buffer_size": 2048,
}

# ===================== 连接函数 =====================
def connect_pluto():
    """连接 Pluto，优先网络 URI，回退 USB 自动发现"""
    if PLUTO_URI:
        try:
            sdr = adi.Pluto(PLUTO_URI)
            sdr.rx()
            print(f"    [✓] 使用指定 URI: {PLUTO_URI}")
            return PLUTO_URI
        except Exception as e:
            print(f"    [!] 指定 URI 失败: {e}")

    for ip in ["ip:192.168.2.1", "ip:192.168.2.10", "ip:192.168.1.10"]:
        try:
            sdr = adi.Pluto(ip)
            sdr.rx()
            print(f"    [✓] 网络连接成功: {ip}")
            return ip
        except Exception:
            pass

    try:
        sdr = adi.Pluto()
        sdr.rx()
        print("    [✓] USB 自动发现成功")
        return None
    except Exception as e:
        print(f"    [✗] 连接失败: {e}")
        sys.exit(1)

# ===================== 能力查询 =====================
ATTR_TESTS = [
    {
        "name": "sample_rate",
        "read": lambda sdr: getattr(sdr, "sample_rate", None),
        "write": lambda sdr, v: setattr(sdr, "sample_rate", int(v)),
        "test_value": 60e6,
        "unit": "Hz",
        "desc": "采样率"
    },
    {
        "name": "center_freq",
        "read": lambda sdr: getattr(sdr, "center_freq", None),
        "write": lambda sdr, v: setattr(sdr, "center_freq", int(v)),
        "test_value": 5800e6,
        "unit": "Hz",
        "desc": "中心频率"
    },
    {
        "name": "filter_bandwidth",
        "read": lambda sdr: getattr(sdr, "filter_bandwidth", None),
        "write": lambda sdr, v: setattr(sdr, "filter_bandwidth", int(v)),
        "test_value": 56e6,
        "unit": "Hz",
        "desc": "带宽"
    },
    {
        "name": "rx_buffer_size",
        "read": lambda sdr: getattr(sdr, "rx_buffer_size", None),
        "write": lambda sdr, v: setattr(sdr, "rx_buffer_size", int(v)),
        "test_value": 2048,
        "unit": "samples",
        "desc": "RX Buffer 大小"
    },
    {
        "name": "gain",
        "read": lambda sdr: getattr(sdr, "gain", None),
        "write": lambda sdr, v: setattr(sdr, "gain", float(v)),
        "test_value": 20.0,
        "unit": "dB",
        "desc": "增益"
    },
    {
        "name": "rx_hardwaregain",
        "read": lambda sdr: getattr(sdr, "rx_hardwaregain", None),
        "write": lambda sdr, v: setattr(sdr, "rx_hardwaregain", float(v)),
        "test_value": 20.0,
        "unit": "dB",
        "desc": "RX 增益"
    },
]

def query_capabilities(uri):
    """查询 Pluto 属性支持情况，读写测试"""
    sdr = adi.Pluto(uri) if uri else adi.Pluto()
    results = []

    for attr in ATTR_TESTS:
        name = attr["name"]
        result = {"name": name, "desc": attr["desc"], "readable": False, "writable": False,
                  "value": None, "error": None}

        # 读取测试
        try:
            val = attr["read"](sdr)
            result["readable"] = True
            result["value"] = val
        except Exception as e:
            result["error"] = str(e)[:60]

        # 写入测试（仅测可读的属性）
        if result["readable"]:
            try:
                orig = result["value"]
                attr["write"](sdr, attr["test_value"])
                after = attr["read"](sdr)
                result["writable"] = (after == attr["test_value"])
                # 恢复原值
                if orig is not None:
                    attr["write"](sdr, orig)
            except Exception as e:
                result["writable"] = False
                result["write_error"] = str(e)[:60]

        results.append(result)

    del sdr
    return results

# ===================== 测试函数 =====================
def measure_burst_throughput(uri, bursts, buffer_size):
    """测量 burst 采集的实际吞吐量"""
    sdr = adi.Pluto(uri) if uri else adi.Pluto()
    sdr.rx_buffer_size = buffer_size

    for _ in range(3):
        sdr.rx()

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

    total_time = sum(times)
    total_bytes = total_samples * 8
    throughput_mbps = (total_bytes / 1e6) / total_time if total_time > 0 else 0

    del sdr

    return {
        "total_samples": total_samples,
        "total_bytes": total_bytes,
        "total_time": total_time,
        "avg_burst_time": np.mean(times),
        "throughput_mbps": throughput_mbps,
    }

def sustained_test(uri, duration_sec, buffer_size):
    """持续采集稳定性测试"""
    sdr = adi.Pluto(uri) if uri else adi.Pluto()
    sdr.rx_buffer_size = buffer_size

    for _ in range(3):
        sdr.rx()

    total_bytes = 0
    burst_count = 0
    start_time = time.perf_counter()
    last_report = start_time

    while time.perf_counter() - start_time < duration_sec:
        chunk = sdr.rx()
        if len(chunk) > 0:
            total_bytes += len(chunk) * 8
            burst_count += 1

        if time.perf_counter() - last_report >= 1.0:
            elapsed = time.perf_counter() - start_time
            rate = (total_bytes / 1e6) / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] {burst_count} bursts, {total_bytes/1e6:.1f} MB, {rate:.1f} MB/s")
            last_report = time.perf_counter()

    total_time = time.perf_counter() - start_time
    del sdr

    return {
        "total_bytes": total_bytes,
        "total_time": total_time,
        "total_bursts": burst_count,
        "avg_rate_mbps": (total_bytes / 1e6) / total_time if total_time > 0 else 0,
    }

# ===================== 主测试 =====================
def main():
    print("=" * 60)
    print("Pluto USB 传输能力基准测试")
    print("=" * 60)

    print("\n[*] 连接 Pluto...")
    uri = connect_pluto()

    # Part 0: 属性读写测试
    print("\n" + "=" * 60)
    print("Part 0: Pluto 属性支持情况")
    print("=" * 60)

    attr_results = query_capabilities(uri)

    print(f"\n{'属性名':<20} {'说明':<12} {'读':>4} {'写':>4} {'当前值':<20} {'备注'}")
    print("-" * 80)

    actual_sr = 60e6
    for r in attr_results:
        readable = "✓" if r["readable"] else "✗"
        writable = "✓" if r["writable"] else "✗"
        val_str = ""
        if r["readable"] and r["value"] is not None:
            v = r["value"]
            if v > 1e9:
                val_str = f"{v/1e9:.2f} GHz"
            elif v > 1e6:
                val_str = f"{v/1e6:.2f} MHz"
            else:
                val_str = str(v)
            if r["name"] == "sample_rate":
                actual_sr = float(v)
        note = r.get("write_error", r.get("error", ""))
        if len(note) > 20:
            note = note[:18] + ".."
        print(f"{r['name']:<20} {r['desc']:<12} {readable:>4} {writable:>4} {val_str:<20} {note}")

    # 检查必要属性
    print("\n[*] 必要属性检查:")
    sr_ok = any(r["name"] == "sample_rate" and r["readable"] for r in attr_results)
    print(f"    sample_rate: {'✓ 可用' if sr_ok else '✗ 不可用'}")
    if not sr_ok:
        print("\n    [✗] sample_rate 不可用，burst 传输测试可能失败")
        actual_sr = 60e6  # 使用默认值

    # Part 1: Burst 传输测试
    print("\n" + "=" * 60)
    print("Part 1: Burst 传输测试")
    print("=" * 60)

    results = []
    for cfg in TEST_CONFIGS:
        print(f"\n[*] {cfg['name']} ({cfg['bursts']} bursts × {cfg['buffer_size']} samples)...")
        r = measure_burst_throughput(uri, cfg['bursts'], cfg['buffer_size'])
        results.append((cfg, r))

        window_ms = r['total_samples'] / actual_sr * 1000
        print(f"    总耗时: {r['total_time']*1000:.1f} ms")
        print(f"    总数据: {r['total_bytes']/1e6:.2f} MB ({r['total_samples']:,} samples)")
        print(f"    平均速率: {r['throughput_mbps']:.1f} MB/s")
        print(f"    等效采样窗口: {window_ms:.2f} ms")

    # 汇总
    print("\n" + "-" * 70)
    print(f"{'配置':<25} {'耗时':>8} {'数据量':>10} {'速率':>10} {'等效窗口':>10}")
    print("-" * 70)
    for cfg, r in results:
        window_ms = r['total_samples'] / actual_sr * 1000
        print(f"{cfg['name']:<25} {r['total_time']*1000:>7.1f}ms "
              f"{r['total_bytes']/1e6:>9.2f}MB {r['throughput_mbps']:>9.1f}MB/s "
              f"{window_ms:>9.2f}ms")

    # Part 2: 持续采集测试
    print("\n" + "=" * 60)
    print("Part 2: 持续采集稳定性测试（5 秒）")
    print("=" * 60)

    r = sustained_test(uri, SUSTAINED_TEST['duration_sec'], SUSTAINED_TEST['buffer_size'])
    print(f"\n    总耗时: {r['total_time']:.2f} s")
    print(f"    总数据: {r['total_bytes']/1e6:.2f} MB")
    print(f"    总 bursts: {r['total_bursts']}")
    print(f"    平均速率: {r['avg_rate_mbps']:.2f} MB/s ({r['avg_rate_mbps']*8:.1f} Mb/s)")

    # 轮询周期估算
    print("\n" + "=" * 60)
    print("轮询周期估算（5 频点）")
    print("=" * 60)
    print(f"\n{'配置':<25} {'等效窗口':>10} {'5频点周期':>12} {'2秒内次数':>12}")
    print("-" * 60)

    for cfg, r in results:
        window_ms = r['total_samples'] / actual_sr * 1000
        cycle_ms = (window_ms + 10) * 5  # 切换开销 10ms/频点
        cycles_in_2s = 2000 / cycle_ms if cycle_ms > 0 else 999
        print(f"{cfg['name']:<25} {window_ms:>9.2f}ms {cycle_ms:>11.1f}ms {cycles_in_2s:>11.1f}次")

    print("\n" + "=" * 60)
    print("结果解读")
    print("=" * 60)
    print("""
  - 等效窗口：单次 burst 累积的采样时长（ms）
  - 5频点周期：轮询完所有频点所需时间
  - 2秒内次数：在检测时限内能完成多少轮完整扫描
  - 推荐：2秒内至少完成 1 轮完整扫描
    """)

if __name__ == "__main__":
    main()

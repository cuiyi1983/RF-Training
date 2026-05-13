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

# ===================== Part 0: 属性探测 =====================
def probe_pluto_attributes(uri):
    """探测 Pluto 所有属性及配置能力"""
    sdr = adi.Pluto(uri) if uri else adi.Pluto()

    print("\n[*] 探测 Pluto 属性...")

    # 1. 所有属性列表
    print("\n    === 所有属性 ===")
    all_attrs = sorted([a for a in dir(sdr) if not a.startswith('_')])
    for a in all_attrs:
        print(f"      {a}")

    # 2. 关键属性读写测试
    print("\n    === 关键属性读写测试 ===")

    # 要测试的属性：名称、读取方式、写入值
    attr_tests = [
        # (属性名, 读取值, 写入值)
        ("sample_rate",     None,   int(60e6)),
        ("center_freq",     None,   int(5800e6)),
        ("center_frequency",None,   int(5800e6)),
        ("filter_bandwidth",None,   int(56e6)),
        ("bandwidth",       None,   int(56e6)),
        ("rx_hardwaregain", None,   float(20.0)),
        ("gain",            None,   float(20.0)),
        ("rx_buffer_size",  None,   int(2048)),
        ("tx_rf_bandwidth",None,   int(20e6)),
        ("lo_frequency",    None,   int(2000e6)),
        ("frequency",       None,   int(5800e6)),
    ]

    results = []
    for attr_name, _, test_val in attr_tests:
        r = {"name": attr_name, "exists": False, "readable": False, "writable": False,
             "read_val": None, "write_val": None, "error": None}
        results.append(r)

        if not hasattr(sdr, attr_name):
            continue
        r["exists"] = True

        # 读取
        try:
            r["read_val"] = getattr(sdr, attr_name)
            r["readable"] = True
        except Exception as e:
            r["error"] = f"read: {str(e)[:50]}"
            continue

        # 写入（恢复原值）
        try:
            orig = r["read_val"]
            setattr(sdr, attr_name, test_val)
            r["write_val"] = getattr(sdr, attr_name)
            r["writable"] = (r["write_val"] == test_val)
            # 恢复
            if orig is not None:
                try:
                    setattr(sdr, attr_name, orig)
                except Exception:
                    pass
        except Exception as e:
            r["error"] = f"write: {str(e)[:50]}"

    # 打印表格
    print(f"\n    {'属性名':<22} {'存在':>4} {'可读':>4} {'可写':>4} {'当前值':<22} {'错误'}")
    print("    " + "-" * 80)
    for r in results:
        if not r["exists"]:
            print(f"    {r['name']:<22} {'✗':>4} {'—':>4} {'—':>4} {'—':<22}")
            continue
        ex = "✓" if r["exists"] else "✗"
        rd = "✓" if r["readable"] else "✗"
        wr = "✓" if r["writable"] else "✗"
        val = ""
        if r["readable"] and r["read_val"] is not None:
            v = r["read_val"]
            if isinstance(v, float) and v > 1e6:
                val = f"{v/1e6:.2f} MHz"
            elif isinstance(v, int) and v > 1e6:
                val = f"{v/1e6:.2f} MHz"
            else:
                val = str(v)[:20]
        err = r["error"] or ""
        print(f"    {r['name']:<22} {ex:>4} {rd:>4} {wr:>4} {val:<22} {err[:25]}")

    # 3. 尝试 rx() 获取实际采样数据
    print("\n    === rx() 采数测试 ===")
    try:
        chunk = sdr.rx()
        print(f"      rx() 成功: {len(chunk)} samples, dtype={chunk.dtype}")
    except Exception as e:
        print(f"      rx() 失败: {e}")

    del sdr
    return results

# ===================== Part 1-2: 吞吐量测试 =====================
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

    # 探测采样率（用于后续计算）
    try:
        sdr_tmp = adi.Pluto(uri) if uri else adi.Pluto()
        actual_sr = float(sdr_tmp.sample_rate)
        del sdr_tmp
    except Exception:
        actual_sr = 60e6
        print(f"    [!] 无法读取 sample_rate，使用默认值 60 MHz")

    # Part 0: 属性探测
    print("\n" + "=" * 60)
    print("Part 0: Pluto 属性探测")
    print("=" * 60)
    probe_pluto_attributes(uri)

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
        cycle_ms = (window_ms + 10) * 5
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

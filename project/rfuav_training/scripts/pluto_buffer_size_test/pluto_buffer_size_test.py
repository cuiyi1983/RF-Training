#!/usr/bin/env python3
"""
Pluto 最大 rx_buffer_size 探测脚本
====================================
测试 Pluto 在 USB 和 IP 两种连接方式下，不同 rx_buffer_size 的稳定性。

用法：
    python pluto_buffer_size_test.py [--ip] [--usb]

参数：
    --ip   测试网络连接（ip:192.168.2.1）
    --usb  测试 USB 连接
    不带参数则测试两种连接方式

作者：RF-Drone-Platform
日期：2026-05-14
"""

import argparse
import sys
import time
import numpy as np

try:
    import adi
except ImportError:
    print("错误：需要安装 pyadi-iio")
    print("  pip install pyadi-iio")
    sys.exit(1)

# 测试的 buffer_size 列表
BUFFER_SIZES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]


def test_connection(uri: str, connection_name: str) -> dict:
    """测试指定连接的稳定性"""
    results = {
        "connection": connection_name,
        "uri": uri,
        "buffer_sizes": [],
        "max_stable_size": 0,
        "max_stable_ms": 0.0,
    }

    print(f"\n{'='*60}")
    print(f"测试 {connection_name} 连接: {uri}")
    print(f"{'='*60}")

    try:
        sdr = adi.Pluto(uri)
        sdr.rx_enabled_channels = [0]
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return results

    # 先测试一次最小 buffer 确保连接正常（带 warmup）
    sdr.rx_buffer_size = 1024
    for _ in range(3):  # warmup: 清空 DMA 过渡期数据
        _ = sdr.rx()

    test_data = sdr.rx()
    if len(test_data) != 1024:
        print(f"❌ 连接测试失败: 期望 1024, 实际 {len(test_data)}")
        return results
    print(f"✅ 连接正常\n")

    # 测试每个 buffer_size
    for size in BUFFER_SIZES:
        continuous_window_ms = size / 60e6 * 1000
        total_bytes = size * 8  # complex64 = 8 bytes

        print(f"测试 rx_buffer_size={size:>8} ({continuous_window_ms:.2f} ms 连续窗口)... ", end="", flush=True)

        try:
            sdr.rx_buffer_size = size

            # 改变 buffer_size 后需要 warmup：清空 DMA 过渡期数据
            for _ in range(3):
                _ = sdr.rx()

            # 连续测试 5 次
            success_count = 0
            sample_counts = []
            transfer_times = []

            for _ in range(5):
                t0 = time.perf_counter()
                data = sdr.rx()
                t1 = time.perf_counter()

                transfer_time = t1 - t0
                transfer_times.append(transfer_time)
                sample_counts.append(len(data))

                if len(data) == size:
                    success_count += 1

            avg_transfer_time = np.mean(transfer_times)
            throughput_mbps = (total_bytes / 1e6) / avg_transfer_time if avg_transfer_time > 0 else 0

            if success_count == 5:
                print(f"✅ 5/5 成功 | 传输 {avg_transfer_time*1000:.1f} ms | 吞吐 {throughput_mbps:.1f} MB/s")
                results["buffer_sizes"].append({
                    "size": size,
                    "window_ms": continuous_window_ms,
                    "success": True,
                    "avg_transfer_ms": avg_transfer_time * 1000,
                    "throughput_mbps": throughput_mbps,
                    "all_counts": sample_counts,
                })
                results["max_stable_size"] = size
                results["max_stable_ms"] = continuous_window_ms
            else:
                print(f"❌ 失败 {5-success_count}/5 | 返回 {sample_counts}")
                results["buffer_sizes"].append({
                    "size": size,
                    "window_ms": continuous_window_ms,
                    "success": False,
                    "sample_counts": sample_counts,
                })
                # 遇到第一个失败就停止
                break

        except Exception as e:
            print(f"❌ 异常: {e}")
            results["buffer_sizes"].append({
                "size": size,
                "window_ms": continuous_window_ms,
                "success": False,
                "error": str(e),
            })
            break

    # 恢复默认配置
    try:
        sdr.rx_buffer_size = 1024
    except:
        pass

    return results


def print_summary(results_list: list):
    """打印汇总结果"""
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")

    for results in results_list:
        conn = results["connection"]
        max_size = results["max_stable_size"]
        max_ms = results["max_stable_ms"]

        print(f"\n{conn} ({results['uri']}):")
        print(f"  最大稳定 buffer_size: {max_size:,} ({max_ms:.2f} ms)")

        for bs in results["buffer_sizes"]:
            status = "✅" if bs["success"] else "❌"
            if bs["success"]:
                print(f"    {status} {bs['size']:>8} → {bs['window_ms']:.2f} ms | {bs['throughput_mbps']:.1f} MB/s")
            else:
                print(f"    {status} {bs['size']:>8} → {bs['window_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Pluto rx_buffer_size 探测")
    parser.add_argument("--ip", action="store_true", help="仅测试 IP 连接")
    parser.add_argument("--usb", action="store_true", help="仅测试 USB 连接")
    args = parser.parse_args()

    results_list = []

    # 确定要测试的连接
    test_usb = not args.ip  # 默认测试 USB，除非明确指定只测 IP
    test_ip = args.usb or (not args.ip and not args.usb)  # 默认都测

    if test_usb:
        usb_results = test_connection("", "USB (自动发现)")
        results_list.append(usb_results)

    if test_ip:
        # 测试多个可能的 IP 地址
        for ip in ["ip:192.168.2.1", "ip:192.168.2.10", "ip:192.168.1.10"]:
            ip_results = test_connection(ip, f"IP ({ip})")
            if ip_results["buffer_sizes"]:  # 如果有结果（连接成功过）
                results_list.append(ip_results)
                break  # 只测试第一个成功的 IP

    # 打印汇总
    if results_list:
        print_summary(results_list)

        # 保存结果
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"buffer_test_results_{timestamp}.json"

        # 只保留可序列化部分
        serializable_results = []
        for r in results_list:
            sr = {
                "connection": r["connection"],
                "uri": r["uri"],
                "max_stable_size": r["max_stable_size"],
                "max_stable_ms": r["max_stable_ms"],
                "buffer_sizes": []
            }
            for bs in r.get("buffer_sizes", []):
                sb = {
                    "size": bs["size"],
                    "window_ms": bs["window_ms"],
                    "success": bs["success"]
                }
                if bs["success"]:
                    sb["throughput_mbps"] = bs.get("throughput_mbps", 0)
                else:
                    sb["error"] = bs.get("error", "unknown")
                sr["buffer_sizes"].append(sb)
            serializable_results.append(sr)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n结果已保存到: {filename}")

        # 推荐配置
        print(f"\n{'='*60}")
        print("推荐配置")
        print(f"{'='*60}")
        for r in results_list:
            if r["max_stable_size"] > 0:
                print(f"\n{r['connection']}:")
                print(f"  rx_buffer_size = {r['max_stable_size']}")
                print(f"  连续窗口 = {r['max_stable_ms']:.2f} ms")
                stft_frames = (r['max_stable_size'] - 1024) // 512 + 1
                print(f"  STFT 帧数 ≈ {stft_frames}")
    else:
        print("未测试任何连接")


if __name__ == "__main__":
    main()
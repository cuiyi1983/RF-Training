#!/usr/bin/env python3
"""
Pluto SDR 针对性采集脚本 - 专门捕获 DJI 5.8GHz 跳频信号
功能：在 5800MHz 附近反复高频采集，捕获间歇性跳频信号
用法：python pluto_iq_collector_targeted.py
"""
import adi
import numpy as np
import time
import os
from datetime import datetime

# ===================== 可配置参数 =====================
CONFIGS = {
    "sampling_rate": 60e6,   # 60 MHz
    "gain": 20,              # 增益 dB（已验证 20dB 工作正常）
    "buffer_size": 2048,     # rx buffer 大小
    "burst_count": 500,      # 大量 burst 累积，捕获跳频
    "num_rounds": 3,         # 采集轮数（每轮多burst）
}

# DJI 5.8GHz 跳频频点（±30MHz 范围）
DJI_FREQUENCIES = {
    "DJI_5700": 5700e6,
    "DJI_5750": 5750e6,
    "DJI_5800": 5800e6,   # 中心频点
    "DJI_5840": 5840e6,
    "DJI_5860": 5860e6,
}

# 针对性采集：只盯 5800 MHz（±30MHz 跳频范围）
CENTER_FREQ = 5800e6
SCAN_FREQS = [
    ("SCAN_5760", 5760e6),   # 5800 - 40
    ("SCAN_5775", 5775e6),   # 5800 - 25
    ("SCAN_5800", 5800e6),   # 5800 中心
    ("SCAN_5825", 5825e6),   # 5800 + 25
    ("SCAN_5850", 5850e6),   # 5800 + 50
]

# ===================== 设备发现 =====================
def find_pluto_uri():
    """自动发现连接的 Pluto 设备"""
    print("[*] 尝试自动发现 Pluto 设备...")
    
    # 方式1：网络连接
    for uri in ["ip:192.168.2.1", "ip:192.168.2.2", "ip:192.168.1.10"]:
        try:
            sdr = adi.Pluto(uri)
            print(f"  [✓] Pluto found: {uri}")
            return uri
        except Exception:
            pass
    
    # 方式2：自动搜索 USB
    try:
        sdr = adi.Pluto()
        print(f"  [✓] Pluto found via auto search (USB)")
        return None
    except Exception as e:
        print(f"  [-] Auto search failed: {e}")
    
    print("[✗] 未发现 Pluto 设备！")
    return None

# ===================== 采集函数 =====================
def collect_bursts(sdr, num_bursts=500, samples_per_burst=2048):
    """
    大量 Burst 累积采集
    每次 rx() 吐 2048 samples
    累积 500 次 → 约 17ms 等效窗口
    """
    all_samples = []
    print(f"  开始 {num_bursts} 次 burst 累积...")
    for i in range(num_bursts):
        try:
            chunk = sdr.rx()
            all_samples.append(chunk)
            if (i + 1) % 100 == 0:
                # 实时功率
                p = 20 * np.log10(np.sqrt(np.mean(np.abs(chunk)**2)) + 1e-12)
                print(f"    Burst {i+1}/{num_bursts} | 功率: {p:.2f} dBFS")
        except Exception as e:
            print(f"  [!] Burst {i} 采集失败: {e}")
            break
    if all_samples:
        return np.concatenate(all_samples)
    return np.array([])

# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("Pluto 针对性采集器 - DJI 5.8GHz 跳频捕获")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: 发现设备
    uri = find_pluto_uri()
    if uri is None:
        print("[✗] 无法发现 Pluto，退出")
        return
    
    # Step 2: 连接 Pluto
    print(f"\n[*] 连接 Pluto...")
    try:
        if uri:
            sdr = adi.Pluto(uri)
        else:
            sdr = adi.Pluto()
        print(f"  [✓] Pluto 连接成功")
    except Exception as e:
        print(f"[✗] 连接失败: {e}")
        return
    
    # Step 3: 配置 Pluto
    print(f"\n[*] 配置 Pluto:")
    print(f"  采样率: {CONFIGS['sampling_rate']/1e6:.0f} MHz")
    print(f"  增益: {CONFIGS['gain']} dB")
    print(f"  Buffer size: {CONFIGS['buffer_size']}")
    print(f"  Burst 次数: {CONFIGS['burst_count']} 次/轮")
    print(f"  采集轮数: {CONFIGS['num_rounds']} 轮")
    
    sdr.sample_rate = int(CONFIGS["sampling_rate"])
    sdr.gain_control_mode = "manual"
    sdr.rx_hardwaregain = CONFIGS["gain"]
    sdr.rx_buffer_size = CONFIGS["buffer_size"]
    
    # Step 4: 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"pluto_targeted_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[*] 输出目录: {output_dir}")
    
    # Step 5: 针对性多频点轮询采集
    print(f"\n[*] 开始针对性采集，{len(SCAN_FREQS)} 个频点 × {CONFIGS['num_rounds']} 轮")
    
    manifest = []
    
    for freq_name, freq in SCAN_FREQS:
        print(f"\n{'='*50}")
        print(f"频段: {freq_name} ({freq/1e6:.1f} MHz)")
        print(f"{'='*50}")
        
        sdr.center_freq = int(freq)
        print(f"  中心频率: {freq/1e6:.1f} MHz")
        time.sleep(0.5)
        
        all_rounds = []
        round_powers = []
        
        for round_idx in range(CONFIGS["num_rounds"]):
            print(f"\n  --- 第 {round_idx+1}/{CONFIGS['num_rounds']} 轮 ---")
            
            # Burst 累积
            samples = collect_bursts(
                sdr,
                num_bursts=CONFIGS["burst_count"],
                samples_per_burst=CONFIGS["buffer_size"]
            )
            
            if len(samples) == 0:
                print(f"    [!] 采集失败，跳过")
                continue
            
            # 实时功率
            power_db = 20 * np.log10(np.sqrt(np.mean(np.abs(samples)**2)) + 1e-12)
            print(f"    → 第{round_idx+1}轮: {len(samples)/1e6:.2f}M samples | 功率: {power_db:.2f} dBFS")
            round_powers.append(power_db)
            all_rounds.append(samples)
            
            time.sleep(0.2)
        
        if not all_rounds:
            continue
        
        # 合并所有轮次
        combined_iq = np.concatenate(all_rounds)
        avg_power = sum(round_powers) / len(round_powers)
        
        print(f"\n  [{freq_name}] 汇总:")
        print(f"    总计: {len(combined_iq)/1e6:.2f}M samples")
        print(f"    均值功率: {avg_power:.2f} dBFS")
        print(f"    各轮功率: {' / '.join(f'{p:.1f}' for p in round_powers)} dBFS")
        
        # 保存
        filename = f"{freq_name}_{len(combined_iq)/1e6:.1f}Msps.npz"
        filepath = os.path.join(output_dir, filename)
        
        np.savez_compressed(
            filepath,
            iq=combined_iq.astype(np.complex64),
            center_freq=freq,
            sampling_rate=CONFIGS["sampling_rate"],
            gain=CONFIGS["gain"],
            timestamp=time.time(),
            freq_name=freq_name,
            num_samples=len(combined_iq),
        )
        print(f"  [✓] 已保存: {filename} ({os.path.getsize(filepath)/1024/1024:.1f} MB)")
        
        manifest.append({
            "filename": filename,
            "freq_name": freq_name,
            "center_freq": freq,
            "num_samples": len(combined_iq),
            "power_dBFS": avg_power,
            "rounds": CONFIGS["num_rounds"],
        })
        
        time.sleep(1)
    
    # Step 6: 保存 manifest
    manifest_file = os.path.join(output_dir, "manifest.txt")
    with open(manifest_file, "w") as f:
        f.write("Pluto 针对性采集清单\n")
        f.write("采集时间: %s\n" % timestamp)
        f.write("采样率: %.0f MHz\n" % (CONFIGS["sampling_rate"]/1e6))
        f.write("增益: %d dB\n" % CONFIGS["gain"])
        f.write("Burst次数: %d 次/轮\n" % CONFIGS["burst_count"])
        f.write("采集轮数: %d 轮\n\n" % CONFIGS["num_rounds"])
        f.write("文件列表:\n")
        for item in manifest:
            f.write("  %s | %s | %.1f MHz | %.2f dBFS | %.1fM samples\n" % (
                item["filename"], item["freq_name"],
                item["center_freq"]/1e6, item["power_dBFS"],
                item["num_samples"]/1e6
            ))
    
    print("\n" + "="*60)
    print("采集完成！")
    print("输出目录: %s" % output_dir)
    print("manifest: %s" % manifest_file)
    print("="*60)
    print("\n崔老板，将整个文件夹发给我验证！")

if __name__ == "__main__":
    main()

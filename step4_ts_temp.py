import argparse
import atexit
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator


DEFAULT_LOG_FILE = Path(__file__).resolve().with_name("step4_runtime.log")
DEFAULT_LOCK_FILE = Path(__file__).resolve().with_name("step4_ts_mapping.lock")
DEFAULT_PROGRESS_INTERVAL = 30
_LOCK_HANDLE = None


def get_log_file():
    return Path(os.environ.get("STEP4_LOG_FILE", str(DEFAULT_LOG_FILE)))


def get_lock_file():
    return Path(os.environ.get("STEP4_LOCK_FILE", str(DEFAULT_LOCK_FILE)))


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_file = get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def acquire_single_instance_lock():
    global _LOCK_HANDLE
    lock_file = get_lock_file()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_HANDLE = lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"检测到已有 Step 4 进程正在运行，锁文件: {lock_file}")
    _LOCK_HANDLE.write(str(os.getpid()))
    _LOCK_HANDLE.flush()


def release_single_instance_lock():
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    _LOCK_HANDLE.close()
    _LOCK_HANDLE = None


def log_process_status(stage):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            status = f.read()
        interesting = []
        for key in ("State", "VmRSS", "VmHWM", "Threads"):
            for line in status.splitlines():
                if line.startswith(f"{key}:"):
                    interesting.append(line.strip())
                    break
        log(f"{stage} | PID={os.getpid()} | " + " | ".join(interesting))
    except Exception as exc:
        log(f"{stage} | 读取 /proc/self/status 失败: {exc}")


def log_gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            log(f"GPU 状态: {result.stdout.strip()}")
        else:
            stderr = result.stderr.strip() if result.stderr else "无输出"
            log(f"GPU 状态获取失败，返回码={result.returncode}，stderr={stderr}")
    except FileNotFoundError:
        log("未找到 nvidia-smi，跳过 GPU 状态记录。")
    except Exception as exc:
        log(f"获取 GPU 状态异常: {exc}")


def register_signal_handlers():
    def _handle_signal(signum, _frame):
        log(f"收到信号 {signum}，准备退出。")
        log_process_status("退出前进程状态")
        log_gpu_status()
        release_single_instance_lock()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def step_4_ai_base_mapping(ct_image_path, output_path, inference_device=None, use_lock=True):
    if use_lock:
        acquire_single_instance_lock()
        atexit.register(release_single_instance_lock)
    register_signal_handlers()

    log("开始执行 Step 4: AI 铺底计算...")
    log(f"输入 CT 路径: {ct_image_path}")
    log(f"输出路径: {output_path}")

    os.environ["HTTP_PROXY"] = "http://127.0.0.1:17890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:17890"
    os.environ["nnUNet_n_proc_DA"] = os.environ.get("nnUNet_n_proc_DA", "1")
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    inference_device = inference_device or os.environ.get("STEP4_DEVICE", "cpu")

    log(
        "运行环境: "
        f"HTTP_PROXY={os.environ.get('HTTP_PROXY')} | "
        f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY')} | "
        f"nnUNet_n_proc_DA={os.environ.get('nnUNet_n_proc_DA')} | "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} | "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} | "
        f"device={inference_device}"
    )
    log_process_status("启动前进程状态")
    log_gpu_status()

    log("正在调用 TotalSegmentator 进行全身分割 (这可能需要一些时间)...")
    temp_ts_path = str(Path(output_path).parent / "temp_ts_output.nii.gz")

    try:
        log(f"即将进入 TotalSegmentator 主调用，device={inference_device}...")
        ts_nib_image = totalsegmentator(
            ct_image_path,
            ml=True,
            fast=True, # 🌟 开启 fast 模式，提速百倍且不爆显存
            device=inference_device,
            verbose=True,
        )
        log("TotalSegmentator 主调用已返回。")
        log_process_status("分割返回后进程状态")
        log_gpu_status()

        import nibabel as nib

        log(f"正在写入临时分割结果: {temp_ts_path}")
        nib.save(ts_nib_image, temp_ts_path)

        ts_sitk_image = sitk.ReadImage(temp_ts_path)
        ts_array = sitk.GetArrayFromImage(ts_sitk_image)
        log(f"SimpleITK 读取完成，ts_array shape={ts_array.shape} dtype={ts_array.dtype}")

    except Exception as exc:
        log(f"TotalSegmentator 运行失败: {exc}")
        log(traceback.format_exc())
        log_process_status("异常退出前进程状态")
        log_gpu_status()
        return False
    finally:
        if os.path.exists(temp_ts_path):
            os.remove(temp_ts_path)
            log("finally: 已清理遗留的临时分割结果文件。")

    log("TotalSegmentator 分割完成，开始执行放射生物学物理降维映射...")

    class_b_serial_ids = [18, 19, 29]
    class_c_parallel_ids = [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 20, 21, 22, 24, 56, 57]
    class_d_bone_ids = list(range(30, 56)) + list(range(58, 118))

    # 目标编码:
    #   0 = 背景(体外)
    #   5 = 普通组织(体内, 非 1/2/3/4)
    # 先用 CT 粗分体内区域，再用器官类别覆写为 2/3/4。
    ct_image = sitk.ReadImage(ct_image_path)
    ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    if ct_array.shape != ts_array.shape:
        log(
            f"警告: CT 与 TS 体素形状不一致，ct={ct_array.shape}, ts={ts_array.shape}。"
            "将退化为基于 TS 标签构建普通组织区域。"
        )
        body_mask = ts_array > 0
    else:
        # -950 HU 基本可分开体内与体外空气，保留肺部等低密度组织为体内区域。
        body_mask = ct_array > -950.0

    base_map_canvas = np.zeros_like(ts_array, dtype=np.uint8)
    base_map_canvas[body_mask] = 5

    log("涂层 1: 铺设高密度防弹衣 (骨骼 -> 4)")
    base_map_canvas[np.isin(ts_array, class_d_bone_ids)] = 4

    log("涂层 2: 铺设容积约束网络 (并联内脏 -> 3)")
    base_map_canvas[np.isin(ts_array, class_c_parallel_ids)] = 3

    log("涂层 3: 铺设极值断崖禁区 (串联神经/通道 -> 2)")
    base_map_canvas[np.isin(ts_array, class_b_serial_ids)] = 2

    log("构建最终物理先验底座...")
    result_image = sitk.GetImageFromArray(base_map_canvas)

    result_image.CopyInformation(ct_image)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    sitk.WriteImage(result_image, output_path)

    log("=== Step 4 完成验证 ===")
    log(f"输出维度: {result_image.GetSize()}")
    log("像素值分布:")
    unique, counts = np.unique(base_map_canvas, return_counts=True)
    for value, count in zip(unique, counts):
        if value == 0:
            log(f"  [0] 背景(体外): {count} voxels")
        elif value == 2:
            log(f"  [2] 串联核心通道: {count} voxels")
        elif value == 3:
            log(f"  [3] 并联内脏容积: {count} voxels")
        elif value == 4:
            log(f"  [4] 骨骼防弹层: {count} voxels")
        elif value == 5:
            log(f"  [5] 普通组织(体内非关键结构): {count} voxels")

    log(f"AI Base Map 已成功保存至: {output_path}")
    return True


def parse_gpu_index(device):
    if device.startswith("gpu:"):
        return int(device.split(":", 1)[1])
    return 0


def parse_devices(device_spec):
    if not device_spec:
        return ["gpu:0"]
    devices = [part.strip() for part in device_spec.split(",") if part.strip()]
    return devices or ["gpu:0"]


def build_device_limits(devices, max_workers):
    if not devices:
        return {}
    base = max_workers // len(devices)
    remainder = max_workers % len(devices)
    limits = {}
    for index, device in enumerate(devices):
        limits[device] = base + (1 if index < remainder else 0)
    return limits


def discover_sample_dirs(input_root):
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"输入根目录不存在: {root}")

    sample_dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "ct.nii.gz").exists():
            sample_dirs.append(path)
    return sample_dirs


def build_output_path(sample_dir, output_subdir):
    sample_id = sample_dir.name
    return sample_dir / output_subdir / f"ai_base_map__{sample_id}.nii.gz"


def launch_sample_process(sample_dir, output_subdir, device, overwrite):
    sample_id = sample_dir.name
    ct_path = sample_dir / "ct.nii.gz"
    output_path = build_output_path(sample_dir, output_subdir)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return {
            "sample_id": sample_id,
            "status": "skipped",
            "output_path": str(output_path),
            "device": device,
        }

    env = os.environ.copy()
    env["STEP4_LOG_FILE"] = str(output_dir / "step4_runtime.log")
    env["STEP4_LOCK_FILE"] = str(output_dir / "step4_ts_mapping.lock")
    gpu_index = parse_gpu_index(device)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["STEP4_DEVICE"] = "gpu:0"

    console_log_path = output_dir / "step4_console.log"
    console_log_handle = console_log_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--ct-path",
        str(ct_path),
        "--output-path",
        str(output_path),
        "--device",
        "gpu:0",
        "--no-lock",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        stdout=console_log_handle,
        stderr=subprocess.STDOUT,
    )
    return {
        "sample_id": sample_id,
        "status": "running",
        "output_path": str(output_path),
        "console_log_path": str(console_log_path),
        "console_log_handle": console_log_handle,
        "process": process,
        "device": device,
        "start_time": time.time(),
    }


def read_recent_lines(path, max_lines=80):
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return list(deque(f, maxlen=max_lines))
    except OSError:
        return []


def infer_task_stage(task):
    lines = read_recent_lines(task["console_log_path"])
    if not lines:
        return "等待日志输出"

    stage = "启动中"
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if "Moving results arrays to CPU" in line:
            stage = "显存不足，回退 CPU"
        elif "Predicting part" in line:
            match = re.search(r"Predicting part (\d+) of (\d+)", line)
            if match:
                stage = f"TotalSegmentator 第 {match.group(1)}/{match.group(2)} 段"
        elif "Resampling..." in line:
            stage = "重采样"
        elif "TotalSegmentator 主调用已返回" in line:
            stage = "分割完成，准备映射"
        elif "涂层 1:" in line:
            stage = "解剖映射中"
        elif "构建最终物理先验底座" in line:
            stage = "写出结果中"
        elif "AI Base Map 已成功保存至" in line:
            stage = "已完成"
    return stage


def log_batch_progress(
    total_samples,
    completed,
    failed,
    skipped,
    running,
    pending_count,
    start_time,
    force=False,
):
    elapsed = time.time() - start_time
    finished_count = completed + failed
    processed_count = completed + failed + skipped

    if finished_count > 0:
        throughput = finished_count / max(elapsed, 1e-6)
        remaining_for_eta = total_samples - skipped - finished_count
        eta_text = format_duration(remaining_for_eta / throughput)
    else:
        eta_text = "待首个样本完成后估算"

    line = (
        f"批进度: done={processed_count}/{total_samples} | success={completed} | "
        f"failed={failed} | skipped={skipped} | running={len(running)} | "
        f"pending={pending_count} | elapsed={format_duration(elapsed)} | ETA={eta_text}"
    )
    log(line)

    if not running:
        return

    stage_counter = Counter()
    task_summaries = []
    for task in running:
        stage = infer_task_stage(task)
        task["stage"] = stage
        stage_counter[stage] += 1
        task_summaries.append(
            (
                time.time() - task["start_time"],
                task["sample_id"],
                task["device"],
                stage,
            )
        )

    stage_summary = " | ".join(
        f"{stage} x{count}" for stage, count in stage_counter.most_common()
    )
    if stage_summary:
        log(f"运行阶段汇总: {stage_summary}")

    task_summaries.sort(reverse=True)
    top_tasks = task_summaries[: min(4, len(task_summaries))]
    details = " ; ".join(
        f"{sample_id}@{device} 已跑 {format_duration(runtime)} [{stage}]"
        for runtime, sample_id, device, stage in top_tasks
    )
    if details:
        log(f"运行中样本: {details}")


def run_batch(
    input_root,
    output_subdir,
    device,
    max_workers=None,
    overwrite=False,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    sample_dirs = discover_sample_dirs(input_root)
    if not sample_dirs:
        log(f"未在 {input_root} 下找到包含 ct.nii.gz 的样本目录。")
        return

    if max_workers is None:
        max_workers = 2 

    devices = parse_devices(device)
    max_workers = max(1, max_workers)
    if len(devices) > max_workers:
        devices = devices[:max_workers]
    device_limits = build_device_limits(devices, max_workers)

    total_samples = len(sample_dirs)
    log(
        f"批处理开始: root={input_root} | samples={total_samples} | "
        f"devices={','.join(devices)} | max_workers={max_workers} | "
        f"per_device={device_limits} | output_subdir={output_subdir}"
    )

    pending = list(sample_dirs)
    running = []
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    last_progress_time = 0.0
    next_device_index = 0

    while pending or running:
        while pending and len(running) < max_workers:
            running_per_device = Counter(task["device"] for task in running)
            available_devices = [
                dev for dev in devices if running_per_device[dev] < device_limits[dev]
            ]
            if not available_devices:
                break

            assigned_device = None
            for offset in range(len(devices)):
                candidate = devices[(next_device_index + offset) % len(devices)]
                if candidate in available_devices:
                    assigned_device = candidate
                    next_device_index = (devices.index(candidate) + 1) % len(devices)
                    break
            if assigned_device is None:
                break

            sample_dir = pending.pop(0)
            task = launch_sample_process(
                sample_dir,
                output_subdir,
                assigned_device,
                overwrite,
            )
            if task["status"] == "skipped":
                skipped += 1
                log(f"跳过样本 {task['sample_id']}，结果已存在: {task['output_path']}")
                continue

            running.append(task)
            log(
                f"启动样本 {task['sample_id']} on {assigned_device} | "
                f"running={len(running)}/{max_workers} | output={task['output_path']}"
            )

        for task in running[:]:
            return_code = task["process"].poll()
            if return_code is None:
                continue

            task["console_log_handle"].close()
            running.remove(task)

            if return_code == 0 and Path(task["output_path"]).exists():
                completed += 1
                duration = time.time() - task["start_time"]
                log(
                    f"样本 {task['sample_id']} 处理完成: {task['output_path']} | "
                    f"耗时={format_duration(duration)}"
                )
            else:
                failed += 1
                duration = time.time() - task["start_time"]
                log(
                    f"样本 {task['sample_id']} 处理失败，返回码={return_code}，"
                    f"耗时={format_duration(duration)}，请查看 {task['console_log_path']}"
                )

        now = time.time()
        if now - last_progress_time >= progress_interval:
            log_batch_progress(
                total_samples=total_samples,
                completed=completed,
                failed=failed,
                skipped=skipped,
                running=running,
                pending_count=len(pending),
                start_time=start_time,
            )
            last_progress_time = now

        if pending or running:
            time.sleep(5)

    log_batch_progress(
        total_samples=total_samples,
        completed=completed,
        failed=failed,
        skipped=skipped,
        running=running,
        pending_count=len(pending),
        start_time=start_time,
        force=True,
    )
    log(
        f"批处理结束: 成功={completed} | 失败={failed} | 跳过={skipped} | "
        f"总计={total_samples} | 总耗时={format_duration(time.time() - start_time)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4 AI base map batch runner")
    parser.add_argument("--single", action="store_true", help="运行单个样本")
    parser.add_argument("--ct-path", help="单样本 CT 路径")
    parser.add_argument("--output-path", help="单样本输出路径")
    parser.add_argument("--device", help="推理设备，例如 cpu / gpu:0")
    parser.add_argument("--no-lock", action="store_true", help="关闭单实例锁")
    parser.add_argument("--batch-root", help="批处理输入根目录")
    parser.add_argument("--output-subdir", default="step4", help="样本内输出子目录名")
    parser.add_argument("--max-workers", type=int, help="并发样本数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="批处理进度日志刷新间隔（秒）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch_root:
        run_batch(
            input_root=args.batch_root,
            output_subdir=args.output_subdir,
            device=args.device or os.environ.get("STEP4_DEVICE", "gpu:0,gpu:1"),
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            progress_interval=max(5, args.progress_interval),
        )
    elif args.single or args.ct_path or args.output_path:
        if not args.ct_path or not args.output_path:
            raise SystemExit("--single 模式下必须同时提供 --ct-path 和 --output-path")
        ok = step_4_ai_base_mapping(
            args.ct_path,
            args.output_path,
            inference_device=args.device,
            use_lock=not args.no_lock,
        )
        if not ok:
            raise SystemExit(1)
    else:
        ct_path = "/data1/home/zhaojianxiang/Dosedata/test_sample/_488207G/ct.nii.gz"
        output_dir = "/data1/home/zhaojianxiang/Dosedata/step4/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "ai_base_map__488207G.nii.gz")
        step_4_ai_base_mapping(ct_path, output_file)
"""
Performance benchmark CI for Wan-AI/Wan2.2-T2V-A14B-Diffusers text-to-video diffusion model.

Each test case starts a DiffusionServer with its own parallel configuration,
runs diffusion_benchmark_serving.py against it, and asserts that performance
metrics stay within the configured baselines.

Test configurations (defined in test_wan22.json):
  baseline              — no sequence/cfg parallelism
  cfg2_ulysses2         — CFG-parallel=2 + Ulysses SP=2 (default, 4 GPUs)
  cfg2_ulysses2_cache_dit — default + CacheDiT acceleration
  cfg2_ulysses2_fp8     — default + FP8 quantization
  cfg2_ulysses2_tp2     — default + Tensor Parallel=2 (requires 8 GPUs)
  cfg2_ulysses2_hsdp    — default + HSDP weight sharding

Video shape: height=480, width=640, fps=16, num-frames=80 (random dataset, t2v task).

GPU requirement: 4× NVIDIA H100 80 GB (8× for the tp2 configuration)
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import pytest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

# Results directory: override via WAN22_BENCHMARK_DIR env var.
# Defaults to tests/perf/results/ inside the repo.
_DEFAULT_RESULT_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_RESULT_DIR = Path(os.environ.get("WAN22_BENCHMARK_DIR", str(_DEFAULT_RESULT_DIR)))

BENCHMARK_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"
)

CONFIG_FILE_PATH = str(Path(__file__).parent.parent / "tests" / "test_wan22.json")


def load_configs(config_path: str) -> list[dict[str, Any]]:
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)
        return configs
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)

_server_lock = threading.Lock()


class DiffusionServer:
    """Start a vLLM-Omni Wan2.2 diffusion model server as a subprocess.

    The server is launched with the Wan2.2-specific flags
    (--ulysses-degree, --ring, --boundary-ratio, --flow-shift, --cache-backend, etc.)
    directly on the CLI, without requiring a stage-configs YAML file.

    Minimum hardware: 4× NVIDIA H100 80 GB.
    """

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = port if port is not None else _get_open_port()
        self.test_name: str = ""

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching DiffusionServer: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )

        max_wait = 1200  # 20 minutes for model loading
        start = time.time()
        while time.time() - start < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    if s.connect_ex((self.host, self.port)) == 0:
                        print(f"DiffusionServer ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"DiffusionServer did not start within {max_wait}s")

    def _kill_process_tree(self, pid: int) -> None:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            all_pids = [pid] + [c.pid for c in children]

            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            gone, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

            time.sleep(1)
            still_alive = [p for p in all_pids if psutil.pid_exists(p)]
            if still_alive:
                print(f"Warning: processes still alive after shutdown: {still_alive}")
                for p in still_alive:
                    try:
                        subprocess.run(["kill", "-9", str(p)], timeout=2)
                    except Exception:
                        pass
        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self._kill_process_tree(self.proc.pid)


def _get_open_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _build_serve_args(serve_args_dict: dict[str, Any]) -> list[str]:
    """Convert a serve_args dict from test_wan22.json into a flat CLI argument list."""
    args: list[str] = []
    for key, value in serve_args_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            args.extend([flag, str(value)])
    return args


def _unique_server_params(configs: list[dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """Return one (test_name, model, serve_args_list) tuple per unique test."""
    seen: set[str] = set()
    result: list[tuple[str, str, list[str]]] = []
    for cfg in configs:
        test_name = cfg["test_name"]
        if test_name not in seen:
            seen.add(test_name)
            model = cfg["server_params"]["model"]
            serve_args = _build_serve_args(cfg["server_params"].get("serve_args", {}))
            result.append((test_name, model, serve_args))
    return result


def _test_param_mapping(configs: list[dict[str, Any]]) -> dict[str, list[dict]]:
    mapping: dict[str, list[dict]] = {}
    for cfg in configs:
        name = cfg["test_name"]
        mapping.setdefault(name, [])
        mapping[name].extend(cfg["benchmark_params"])
    return mapping


server_params = _unique_server_params(BENCHMARK_CONFIGS)
test_param_map = _test_param_mapping(BENCHMARK_CONFIGS)

benchmark_indices: list[tuple[str, int]] = []
for cfg in BENCHMARK_CONFIGS:
    name = cfg["test_name"]
    for idx in range(len(test_param_map[name])):
        entry = (name, idx)
        if entry not in benchmark_indices:
            benchmark_indices.append(entry)


@pytest.fixture(scope="module")
def diffusion_server(request):
    """Start one DiffusionServer per unique test configuration (module scope)."""
    with _server_lock:
        test_name, model, serve_args = request.param

        print(f"\nStarting DiffusionServer for test: {test_name}, model: {model}")
        with DiffusionServer(model, serve_args) as server:
            server.test_name = test_name
            print("DiffusionServer started successfully")
            yield server
            print("DiffusionServer stopping…")

    print("DiffusionServer stopped")


@pytest.fixture(params=benchmark_indices)
def benchmark_params(request, diffusion_server):
    """Yield the benchmark params dict for the current (test_name, index) pair."""
    test_name, param_index = request.param

    if test_name != diffusion_server.test_name:
        pytest.skip(f"Skipping {test_name} – server is configured for {diffusion_server.test_name}")

    params_list = test_param_map.get(test_name, [])
    if not params_list:
        raise ValueError(f"No benchmark params for test: {test_name}")
    if param_index >= len(params_list):
        raise ValueError(f"Param index {param_index} out of range for test: {test_name}")

    current = param_index + 1
    total = len(params_list)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": params_list[param_index]}


def run_benchmark(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    test_name: str,
) -> dict[str, Any]:
    """Run diffusion_benchmark_serving.py as a subprocess and return parsed metrics."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    BENCHMARK_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = BENCHMARK_RESULT_DIR / f"wan22_t2v_perf_{test_name}_{timestamp}.json"

    exclude_keys = {"baseline", "dataset", "task"}

    cmd = [
        sys.executable,
        BENCHMARK_SCRIPT,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--backend",
        "vllm-omni",
        "--dataset",
        params.get("dataset", "random"),
        "--task",
        params.get("task", "t2v"),
        "--output-file",
        str(result_file),
    ]

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, dict):
            cmd.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(value)])

    print(f"\nRunning benchmark: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(Path(__file__).parent.parent.parent.parent),
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    for line in iter(process.stderr.readline, ""):
        print(line, end="")
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Benchmark script exited with code {process.returncode}")

    if not result_file.exists():
        raise FileNotFoundError(f"Benchmark result file not found: {result_file}")

    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


def assert_result(result: dict[str, Any], params: dict[str, Any]) -> None:
    """Assert that benchmark metrics satisfy the configured baselines."""
    num_prompts = params.get("num-prompts", 10)
    completed = result.get("completed_requests", result.get("completed", 0))
    assert completed == num_prompts, f"Expected {num_prompts} completed requests, got {completed}"

    baseline = params.get("baseline", {})
    for metric, threshold in baseline.items():
        current = result.get(metric)
        assert current is not None, f"Metric '{metric}' not found in result: {list(result.keys())}"
        if "throughput" in metric:
            assert current >= threshold, f"{metric}: {current:.4f} < baseline {threshold}"
        else:
            assert current <= threshold, f"{metric}: {current:.4f} > baseline {threshold}"


@pytest.mark.parametrize("diffusion_server", server_params, indirect=True)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_wan22_t2v_performance_benchmark(diffusion_server, benchmark_params):
    """Run the Wan2.2 T2V performance benchmark and assert against baselines.

    One server is started per unique parallel configuration (module scope).
    For each server, all benchmark parameter sets defined in test_wan22.json are
    executed sequentially, and results are asserted against the baselines.

    Video shape: height=480, width=640, fps=16, num-frames=80 (random dataset).

    Tracked metrics:
        - throughput_qps (higher is better)
        - latency_p50, latency_p99 (lower is better)
    """
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]

    result = run_benchmark(
        host=diffusion_server.host,
        port=diffusion_server.port,
        model=diffusion_server.model,
        params=params,
        test_name=test_name,
    )

    print(f"\n{'=' * 60}")
    print(f"Results for {test_name}:")
    for key in (
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p50",
        "latency_p99",
    ):
        if key in result:
            print(f"  {key}: {result[key]:.4f}")
    saved = sorted(BENCHMARK_RESULT_DIR.glob(f"wan22_t2v_perf_{test_name}_*.json"))
    if saved:
        print(f"\n  Result files saved to: {BENCHMARK_RESULT_DIR}")
        for f in saved:
            print(f"    {f.name}")
    print(f"{'=' * 60}")

    assert_result(result, params)

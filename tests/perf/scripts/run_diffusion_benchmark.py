"""
Performance benchmark CI runner for diffusion models.

Supports two server backends:
  - vllm-omni (default): starts DiffusionServer via vllm_omni.entrypoints.cli.main,
    benchmarks with diffusion_benchmark_serving.py --backend vllm-omni
  - sglang: starts SglangServer via `sglang serve`,
    benchmarks with diffusion_benchmark_serving.py --backend sglang

A config JSON file is REQUIRED via --config-file:
  pytest run_diffusion_benchmark.py --config-file tests/perf/tests/test_qwen_image_vllm_omni.json
  pytest run_diffusion_benchmark.py --config-file tests/perf/tests/test_qwen_image_sglang.json

JSON config entries are distinguished by a "server_type" field ("vllm-omni" or "sglang").
sglang entries support two additional fields under server_params:
  - "env": dict of extra environment variables (e.g. SGLANG_CACHE_DIT_ENABLED)
  - "cache_dit_config": dict written to a temp YAML and passed as
    --cache-dit-config to sglang serve (requires cache-dit >= 1.2.0)

Results for every run are saved as individual JSON files under BENCHMARK_RESULT_DIR
(override via the DIFFUSION_BENCHMARK_DIR environment variable).
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import pytest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_RESULT_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_RESULT_DIR = Path(os.environ.get("DIFFUSION_BENCHMARK_DIR", str(_DEFAULT_RESULT_DIR)))

BENCHMARK_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"
)


def _get_config_file_from_argv() -> str | None:
    """Read --config-file from sys.argv at import time so pytest parametrize can use it.

    pytest_addoption (below) registers the same flag so pytest does not reject it.
    Supports both ``--config-file path`` and ``--config-file=path`` forms.
    Returns None if the flag is not present; callers must handle the missing case.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--config-file="):
            return arg.split("=", 1)[1]
    return None


CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    raise ValueError(
        "--config-file is required. Pass the path to a benchmark config JSON, e.g.:\n"
        "  pytest run_diffusion_benchmark.py "
        "--config-file tests/perf/tests/test_qwen_image_vllm_omni.json"
    )

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _resolve_refs(configs: list[dict[str, Any]], config_dir: Path) -> list[dict[str, Any]]:
    """Resolve {"$ref": "filename.json"} in benchmark_params fields."""
    for cfg in configs:
        bp = cfg.get("benchmark_params")
        if isinstance(bp, dict) and "$ref" in bp:
            ref_path = config_dir / bp["$ref"]
            try:
                with open(ref_path, encoding="utf-8") as f:
                    cfg["benchmark_params"] = json.load(f)
            except FileNotFoundError:
                raise ValueError(f"benchmark_params $ref not found: {ref_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing error in {ref_path}: {e}")
    return configs


def load_configs(config_path: str) -> list[dict[str, Any]]:
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)
        return _resolve_refs(configs, abs_path.parent)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)


# Register --config-file with pytest so it does not reject the argument.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--config-file",
        action="store",
        default=None,
        help=(
            "Path to the benchmark config JSON file (required). "
            "Example: --config-file tests/perf/tests/test_qwen_image_vllm_omni.json"
        ),
    )


_server_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: int = 1200) -> None:
    """Block until the given host:port accepts connections or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) == 0:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server did not start on {host}:{port} within {timeout}s")


def _kill_process_tree(pid: int) -> None:
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


# ---------------------------------------------------------------------------
# Server classes
# ---------------------------------------------------------------------------


class DiffusionServer:
    """Start a vLLM-Omni diffusion model server as a subprocess.

    Launched via vllm_omni.entrypoints.cli.main with the diffusion-specific
    parallelism flags (--usp, --ring, --cfg-parallel-size, etc.) passed directly
    on the CLI.  Minimum hardware: 4× NVIDIA H100 80 GB.
    """

    server_type = "vllm-omni"

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.host = "127.0.0.1"
        self.port = port if port is not None else _get_open_port()
        self.proc: subprocess.Popen | None = None
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
        _wait_for_port(self.host, self.port)
        print(f"DiffusionServer ready on {self.host}:{self.port}")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, *_):
        if self.proc:
            _kill_process_tree(self.proc.pid)


class SglangServer:
    """Start a sglang serve process for diffusion benchmarking.

    Supports two Cache-DiT activation modes:
      1. Environment variable:  pass env={"SGLANG_CACHE_DIT_ENABLED": "true"}
      2. YAML config file:      pass cache_dit_config={...} (written to a temp
         file and forwarded as --cache-dit-config; requires cache-dit >= 1.2.0)
    """

    server_type = "sglang"

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
        env_overrides: dict[str, str] | None = None,
        cache_dit_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.host = "127.0.0.1"
        self.port = port if port is not None else _get_open_port()
        self.env_overrides = env_overrides or {}
        self.cache_dit_config = cache_dit_config
        self.proc: subprocess.Popen | None = None
        self._tmp_yaml: str | None = None
        self.test_name: str = ""

    @staticmethod
    def _write_cache_dit_yaml(config: dict[str, Any]) -> str:
        """Serialize config dict to a temp YAML file and return its path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            import yaml  # PyYAML

            yaml.dump(config, tmp, default_flow_style=False)
        except ImportError:
            # Fallback: write flat YAML manually (no nested structures expected)
            for k, v in config.items():
                if isinstance(v, bool):
                    tmp.write(f"{k}: {'true' if v else 'false'}\n")
                elif v is None:
                    tmp.write(f"{k}: null\n")
                else:
                    tmp.write(f"{k}: {v}\n")
        tmp.close()
        return tmp.name

    def _start_server(self) -> None:
        env = os.environ.copy()
        env.update(self.env_overrides)

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        if self.cache_dit_config is not None:
            self._tmp_yaml = self._write_cache_dit_yaml(self.cache_dit_config)
            cmd += ["--cache-dit-config", self._tmp_yaml]

        print(f"Launching SglangServer: {' '.join(cmd)}")
        if self.env_overrides:
            print(f"  Extra env: {self.env_overrides}")

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
        _wait_for_port(self.host, self.port)
        print(f"SglangServer ready on {self.host}:{self.port}")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, *_):
        if self.proc:
            _kill_process_tree(self.proc.pid)
        if self._tmp_yaml:
            try:
                Path(self._tmp_yaml).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _build_serve_args(serve_args_dict: dict[str, Any]) -> list[str]:
    """Convert a serve_args dict from test.json into a flat CLI argument list."""
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


def _unique_server_params(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one server-config dict per unique test_name."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for cfg in configs:
        test_name = cfg["test_name"]
        if test_name in seen:
            continue
        seen.add(test_name)
        server_type = cfg.get("server_type", "vllm-omni")
        result.append(
            {
                "test_name": test_name,
                "server_type": server_type,
                "model": cfg["server_params"]["model"],
                "serve_args": _build_serve_args(cfg["server_params"].get("serve_args", {})),
                "env_overrides": cfg["server_params"].get("env", {}),
                "cache_dit_config": cfg["server_params"].get("cache_dit_config"),
                "benchmark_backend": server_type,  # "vllm-omni" or "sglang"
            }
        )
    return result


def _test_param_mapping(configs: list[dict[str, Any]]) -> dict[str, list[dict]]:
    mapping: dict[str, list[dict]] = {}
    for cfg in configs:
        name = cfg["test_name"]
        mapping.setdefault(name, [])
        mapping[name].extend(cfg["benchmark_params"])
    return mapping


def _make_server(server_cfg: dict[str, Any]) -> DiffusionServer | SglangServer:
    """Factory: return the appropriate server instance for the given config."""
    model = server_cfg["model"]
    serve_args = server_cfg["serve_args"]
    if server_cfg["server_type"] == "sglang":
        return SglangServer(
            model=model,
            serve_args=serve_args,
            env_overrides=server_cfg.get("env_overrides", {}),
            cache_dit_config=server_cfg.get("cache_dit_config"),
        )
    return DiffusionServer(model=model, serve_args=serve_args)


# ---------------------------------------------------------------------------
# Parametrize data
# ---------------------------------------------------------------------------

server_params = _unique_server_params(BENCHMARK_CONFIGS)
test_param_map = _test_param_mapping(BENCHMARK_CONFIGS)

benchmark_indices: list[int] = list(range(max(len(v) for v in test_param_map.values())))

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diffusion_server(request):
    """Start one server (vllm-omni or sglang) per unique test configuration."""
    with _server_lock:
        server_cfg: dict[str, Any] = request.param
        test_name = server_cfg["test_name"]
        server_type = server_cfg["server_type"]

        print(f"\nStarting {server_type} server for test: {test_name}")
        with _make_server(server_cfg) as server:
            server.test_name = test_name
            print(f"{server_type} server started successfully")
            yield server
            print(f"{server_type} server stopping…")

    print(f"{server_type} server stopped")


@pytest.fixture
def benchmark_params(request, diffusion_server):
    """Yield the benchmark params dict for the current (server, index) pair."""
    param_index: int = request.param
    test_name = diffusion_server.test_name

    params_list = test_param_map.get(test_name, [])
    if not params_list:
        raise ValueError(f"No benchmark params for test: {test_name}")
    if param_index >= len(params_list):
        pytest.skip(f"Param index {param_index} out of range for {test_name} (has {len(params_list)} params)")

    current = param_index + 1
    total = len(params_list)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": params_list[param_index]}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    test_name: str,
    backend: str = "vllm-omni",
) -> dict[str, Any]:
    """Run diffusion_benchmark_serving.py as a subprocess and return parsed metrics.

    The result JSON is saved under BENCHMARK_RESULT_DIR with the backend name
    embedded in the filename so vllm-omni and sglang results stay distinct.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    BENCHMARK_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    param_name = params.get("name", "")
    name_suffix = f"_{param_name}" if param_name else ""
    result_file = BENCHMARK_RESULT_DIR / f"diffusion_perf_{backend}_{test_name}{name_suffix}_{timestamp}.json"

    exclude_keys = {"baseline", "dataset", "task", "name"}

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
        backend,
        "--dataset",
        params.get("dataset", "random"),
        "--task",
        params.get("task", "t2i"),
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
        elif isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(value)])

    print(f"\nRunning benchmark (backend={backend}): {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(Path(__file__).parent.parent.parent.parent),
    )

    def _drain(stream) -> None:
        for line in iter(stream.readline, ""):
            print(line, end="")

    stdout_thread = threading.Thread(target=_drain, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Benchmark script exited with code {process.returncode}")

    if not result_file.exists():
        raise FileNotFoundError(f"Benchmark result file not found: {result_file}")

    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_result(result: dict[str, Any], params: dict[str, Any]) -> None:
    """Assert that benchmark metrics satisfy the configured baselines."""
    num_prompts = params.get("num-prompts", 10)
    completed = result.get("completed_requests", result.get("completed", 0))
    assert completed == num_prompts, f"Expected {num_prompts} completed requests, got {completed}"

    for metric, threshold in params.get("baseline", {}).items():
        current = result.get(metric)
        assert current is not None, f"Metric '{metric}' not found in result: {list(result.keys())}"
        if "throughput" in metric:
            assert current >= threshold, f"{metric}: {current:.4f} < baseline {threshold}"
        else:
            assert current <= threshold, f"{metric}: {current:.4f} > baseline {threshold}"


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "diffusion_server",
    server_params,
    ids=[p["test_name"] for p in server_params],
    indirect=True,
)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_diffusion_performance_benchmark(diffusion_server, benchmark_params):
    """Run the diffusion performance benchmark and assert against baselines.

    One server is started per unique parallel configuration (module scope).
    For each server, all benchmark parameter sets defined in the config JSON
    are executed sequentially; results are asserted against the baselines.

    Tracked metrics:
        - throughput_qps          (higher is better)
        - latency_p50, latency_p99 (lower is better)
    """
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    backend = diffusion_server.server_type  # "vllm-omni" or "sglang"

    result = run_benchmark(
        host=diffusion_server.host,
        port=diffusion_server.port,
        model=diffusion_server.model,
        params=params,
        test_name=test_name,
        backend=backend,
    )

    print(f"\n{'=' * 60}")
    print(f"Results for {test_name} (server={diffusion_server.server_type}, backend={backend}):")
    for key in (
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p50",
        "latency_p99",
        "peak_memory_mb_max",
        "peak_memory_mb_mean",
        "peak_memory_mb_median",
    ):
        if key in result:
            print(f"  {key}: {result[key]:.4f}")

    saved = sorted(BENCHMARK_RESULT_DIR.glob(f"diffusion_perf_{backend}_{test_name}_*.json"))
    if saved:
        print(f"\n  Result files: {BENCHMARK_RESULT_DIR}")
        for f in saved:
            print(f"    {f.name}")
    print("=" * 60)

    assert_result(result, params)

"""
Benchmark de threads para Ollama em CPU.

Objetivo:
  Encontrar o melhor valor de options.num_thread para os modelos qwen3:8b
  gemma2:9b, llama3.2:3b e granite3.3:8b no mesmo prompt, medindo
  latencia, TTFT, tokens/s, erros e swap.

Saidas:
  benchmark_thread_results/results_raw.csv
  benchmark_thread_results/summary.csv
  benchmark_thread_results/ranking.csv
  benchmark_thread_results/recommendation.md
  benchmark_thread_results/run_config.json
  benchmark_thread_results/figures/*.png

Variaveis de ambiente uteis:
  OLLAMA_BASE_URL=http://localhost:11434
  THREAD_MODELS=gemma4:e2b,gemma2:9b,qwen3:8b,llama3.1:8b,qwen2.5:7b,granite3.3:8b,qwen3.5:9b,qwen3.5:4b
  THREAD_LEVELS=6,8,10,12,14,16,18,19,20,22,24
  THREAD_REPEATS=5
  THREAD_WARMUPS=1
  THREAD_TIMEOUT=900
  THREAD_NUM_PREDICT=192
  THREAD_KEEP_ALIVE=10m
  THREAD_VALIDATE_RESPONSE=1
  THREAD_EXPECTED_RESPONSE={"itens":[...]}
  THREAD_ERROR_RERUNS=1
  THREAD_INVALID_RESPONSE_RERUNS=0
  THREAD_SWAP_DELTA_THRESHOLD_MB=256
  THREAD_IOWAIT_THRESHOLD=20
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import importlib
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


def load_env_file(filename: str = ".env") -> None:
    """
    Carrega variaveis locais do .env antes de montar a configuracao.
    Assim endpoints e outros dados privados nao precisam aparecer no codigo.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
    ]
    loaded = set()
    for env_path in candidates:
        if env_path in loaded or not os.path.exists(env_path):
            continue
        loaded.add(env_path)
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def mask_sensitive(value: str | None) -> str:
    """Mascara endpoints em telas e arquivos de recomendacao."""
    if not value:
        return ""
    value = str(value)
    if "localhost" in value or "127.0.0.1" in value:
        return value.rstrip("/")
    if "://" in value:
        scheme = value.split("://", 1)[0]
        return f"{scheme}://***"
    if len(value) <= 6:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


# .env e intencionalmente local: ele entra no .gitignore e guarda dados sensiveis.
load_env_file()


def _ensure_package(package: str, import_name: str | None = None):
    """Instala automaticamente dependencias faltantes para rodar o benchmark."""
    try:
        return importlib.import_module(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return importlib.import_module(import_name or package)


requests = _ensure_package("requests")
psutil = _ensure_package("psutil")
pd = _ensure_package("pandas")
matplotlib = _ensure_package("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel


console = Console()

# ============================================================================
# CONFIG
# ============================================================================
# Este bloco define o experimento de threads. O endpoint real fica no .env;
# os demais valores podem ser sobrescritos por variaveis THREAD_*.
# A lista padrao repete os modelos avaliados no benchmark complementar.

# O endpoint real deve ficar em .env. Localhost evita vazar servidor privado no Git.
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODELS = [
    m.strip()
    for m in os.environ.get(
        "THREAD_MODELS",
        (
            "gemma4:e2b,gemma2:9b,qwen3:8b,llama3.1:8b,"
            "qwen2.5:7b,granite3.3:8b,qwen3.5:9b,qwen3.5:4b"
        ),
    ).split(",")
    if m.strip()
]
THREAD_LEVELS = [
    int(x.strip())
    for x in os.environ.get("THREAD_LEVELS", "6,8,10,12,14,16,18,19,20,22,24").split(",")
    if x.strip()
]
REPEATS = int(os.environ.get("THREAD_REPEATS", "5"))
WARMUPS = int(os.environ.get("THREAD_WARMUPS", "1"))
TIMEOUT = int(os.environ.get("THREAD_TIMEOUT", "900"))
KEEP_ALIVE = os.environ.get("THREAD_KEEP_ALIVE", "10m")
NUM_PREDICT = int(os.environ.get("THREAD_NUM_PREDICT", "192"))
TEMPERATURE = float(os.environ.get("THREAD_TEMPERATURE", "0"))
SEED = int(os.environ.get("THREAD_SEED", "42"))
OUT_DIR = os.environ.get("THREAD_OUT_DIR", "benchmark_thread_results")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# O prompt e propositalmente unico e igual para todos os modelos/threads.
# A saida esperada e fixa e curta para reduzir vies de estilo/tamanho:
# o objetivo aqui e medir inferencia por num_thread, nao criatividade.
PROMPT = os.environ.get(
    "THREAD_PROMPT",
    (
        "Tarefa de copia controlada para benchmark de desempenho. "
        "Nao explique, nao resuma, nao reescreva e nao adicione nada. "
        "Sua resposta deve ser exatamente o JSON abaixo, em uma unica linha, "
        "sem markdown, sem texto antes e sem texto depois:\n"
        "{\"itens\":[\"Triagem automatizada de solicitacoes administrativas.\","
        "\"Resumo local de documentos internos sensiveis.\","
        "\"Consulta RAG a normas e manuais institucionais.\","
        "\"Geracao de respostas padronizadas ao cidadao.\","
        "\"Apoio a fluxos administrativos com registro auditavel.\","
        "\"Processamento on-premise para reduzir exposicao de dados.\"]}"
    ),
)
EXPECTED_RESPONSE = os.environ.get("THREAD_EXPECTED_RESPONSE")
if EXPECTED_RESPONSE is None and "\n" in PROMPT:
    EXPECTED_RESPONSE = PROMPT.split("\n", 1)[1].strip()
EXPECTED_RESPONSE = EXPECTED_RESPONSE or ""
VALIDATE_RESPONSE = os.environ.get("THREAD_VALIDATE_RESPONSE", "1") != "0"
MIN_REPEATS_FOR_P95 = int(os.environ.get("THREAD_MIN_REPEATS_FOR_P95", "5"))

CPU_THRESHOLD = float(os.environ.get("THREAD_CPU_THRESHOLD", "65"))
LOAD_THRESHOLD = float(os.environ.get("THREAD_LOAD_THRESHOLD", "0.5"))
MEM_MIN_AVAILABLE_MB = float(os.environ.get("THREAD_MEM_MIN_AVAILABLE_MB", "500"))
SWAP_STABLE_MAX_MB = float(os.environ.get("THREAD_SWAP_STABLE_MAX_MB", "128"))
SWAP_DELTA_THRESHOLD_MB = float(os.environ.get("THREAD_SWAP_DELTA_THRESHOLD_MB", "256"))
SWAP_HARD_LIMIT_MB = float(os.environ.get("THREAD_SWAP_HARD_LIMIT_MB", "2048"))
SWAP_CLEAR_MIN_MB = float(os.environ.get("THREAD_SWAP_CLEAR_MIN_MB", "1"))
IOWAIT_THRESHOLD = float(os.environ.get("THREAD_IOWAIT_THRESHOLD", "20"))
STABILITY_WINDOW = int(os.environ.get("THREAD_STABILITY_WINDOW", "3"))
STABILITY_INTERVAL = float(os.environ.get("THREAD_STABILITY_INTERVAL", "2"))
MAX_WAIT_STABILITY = float(os.environ.get("THREAD_MAX_WAIT_STABILITY", "900"))
WAIT_BEFORE_EACH_CALL = os.environ.get("THREAD_WAIT_BEFORE_EACH_CALL", "1") != "0"
CLEAR_SWAP_BEFORE_START = os.environ.get("THREAD_CLEAR_SWAP_BEFORE_START", "1") != "0"
CLEAR_SWAP_BEFORE_EACH_CALL = os.environ.get("THREAD_CLEAR_SWAP_BEFORE_EACH_CALL", "1") != "0"
CLEAR_SWAP_AFTER_MODEL = os.environ.get("THREAD_CLEAR_SWAP_AFTER_MODEL", "1") != "0"
SWAP_CONTAMINATION_RERUNS = int(os.environ.get("THREAD_SWAP_CONTAMINATION_RERUNS", "1"))
ERROR_RERUNS = int(os.environ.get("THREAD_ERROR_RERUNS", "1"))
INVALID_RESPONSE_RERUNS = int(os.environ.get("THREAD_INVALID_RESPONSE_RERUNS", "0"))
SWAP_RECOVERY_SLEEP = float(os.environ.get("THREAD_SWAP_RECOVERY_SLEEP", "5"))
UNLOAD_BEFORE_MODEL = os.environ.get("THREAD_UNLOAD_BEFORE_MODEL", "1") != "0"
UNLOAD_AFTER_MODEL = os.environ.get("THREAD_UNLOAD_AFTER_MODEL", "1") != "0"
MODEL_UNLOAD_WAIT = float(os.environ.get("THREAD_MODEL_UNLOAD_WAIT", "3"))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SWAP_BASELINE_MB = psutil.swap_memory().used / (1024 * 1024)


# ============================================================================
# HELPERS
# ============================================================================
# Funcoes auxiliares para estabilidade do sistema, chamada ao Ollama, calculo
# de score e geracao dos arquivos de saida.

def get_swap_used_mb() -> float:
    return psutil.swap_memory().used / (1024 * 1024)


def get_swap_delta_mb() -> float:
    return max(0.0, get_swap_used_mb() - SWAP_BASELINE_MB)


def is_swap_contaminated(swap_used_mb: float | None = None, swap_delta_mb: float | None = None) -> bool:
    used = get_swap_used_mb() if swap_used_mb is None else swap_used_mb
    delta = get_swap_delta_mb() if swap_delta_mb is None else swap_delta_mb
    return (
        used > SWAP_STABLE_MAX_MB
        or used >= SWAP_HARD_LIMIT_MB
        or delta >= SWAP_DELTA_THRESHOLD_MB
    )


def reset_swap_baseline(reason: str = ""):
    global SWAP_BASELINE_MB
    SWAP_BASELINE_MB = get_swap_used_mb()
    suffix = f" ({reason})" if reason else ""
    console.print(f"  [dim]baseline swap={SWAP_BASELINE_MB:.0f}MB{suffix}[/dim]")


def short_error_detail(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def is_recoverable_error(error: str | None) -> bool:
    if not error:
        return False
    recoverable = {
        "CONNECT_TIMEOUT",
        "READ_TIMEOUT",
        "TIMEOUT",
        "CONNECTION_ERROR",
        "REQUEST_ERROR",
        "STREAM_JSON_DECODE",
        "STREAM_INCOMPLETE",
        "OLLAMA_STREAM_ERROR",
        "EMPTY_RESPONSE",
    }
    if error in recoverable:
        return True
    if error.startswith("HTTP_"):
        try:
            status = int(error.split("_", 1)[1])
            return status in {408, 409, 425, 429} or status >= 500
        except Exception:
            return True
    return False


def get_load_average_1m() -> float:
    try:
        if hasattr(os, "getloadavg"):
            return os.getloadavg()[0]
    except Exception:
        pass
    try:
        return psutil.getloadavg()[0]
    except Exception:
        return 0.0


def is_base_url_local() -> bool:
    host = (urlparse(BASE_URL).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def collect_local_environment() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_mb": round(vm.total / (1024 * 1024)),
        "telemetry_scope": "local_process",
        "telemetry_matches_inference_host": is_base_url_local(),
    }


def collect_declared_server_environment() -> dict[str, str]:
    return {
        "cpu": os.environ.get("THREAD_SERVER_CPU", ""),
        "physical_cores": os.environ.get("THREAD_SERVER_PHYSICAL_CORES", ""),
        "logical_threads": os.environ.get("THREAD_SERVER_LOGICAL_THREADS", ""),
        "memory": os.environ.get("THREAD_SERVER_MEMORY", ""),
        "os": os.environ.get("THREAD_SERVER_OS", ""),
        "ollama_version": os.environ.get("THREAD_SERVER_OLLAMA_VERSION", ""),
        "model_quantization": os.environ.get("THREAD_MODEL_QUANTIZATION", ""),
        "notes": os.environ.get("THREAD_SERVER_NOTES", ""),
    }


def collect_ollama_metadata() -> dict[str, Any]:
    try:
        response = requests.get(f"{BASE_URL}/api/version", timeout=min(TIMEOUT, 10))
        version = ""
        if response.status_code == 200:
            try:
                version = response.json().get("version", "")
            except Exception as exc:
                version = ""
                json_error = type(exc).__name__
            else:
                json_error = ""
        else:
            json_error = ""
        return {
            "reachable": response.status_code == 200,
            "status_code": response.status_code,
            "version": version,
            "json_error": json_error,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "status_code": None,
            "version": "",
            "error": type(exc).__name__,
        }


def unload_model(model: str, reason: str):
    try:
        requests.post(
            f"{BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=min(TIMEOUT, 60),
        )
        console.print(f"  [dim]unload {model} ({reason})[/dim]")
    except Exception as e:
        console.print(f"  [dim yellow]unload {model} nao confirmado ({type(e).__name__})[/dim yellow]")
    time.sleep(MODEL_UNLOAD_WAIT)


def clear_swap_if_possible(force: bool = False) -> bool:
    """
    Tenta limpar swap via swapoff/swapon.
    Em Linux precisa rodar como root ou ter sudo sem senha para swapoff/swapon.
    """
    if os.name == "nt":
        return False

    before = get_swap_used_mb()
    if not force and before < SWAP_CLEAR_MIN_MB:
        return True

    command_sets: list[tuple[list[str], list[str]]] = []
    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            command_sets.append((["swapoff", "-a"], ["swapon", "-a"]))
    except Exception:
        pass
    command_sets.append((["sudo", "-n", "swapoff", "-a"], ["sudo", "-n", "swapon", "-a"]))

    for off_cmd, on_cmd in command_sets:
        try:
            console.print(f"  [yellow]limpando swap automaticamente ({before:.0f}MB)...[/yellow]")
            subprocess.run(off_cmd, check=True, timeout=180, capture_output=True, text=True)
            subprocess.run(on_cmd, check=True, timeout=180, capture_output=True, text=True)
            after = get_swap_used_mb()
            console.print(f"  [green]swap limpo com swapoff/swapon ({before:.0f}MB -> {after:.0f}MB)[/green]")
            reset_swap_baseline("apos limpar swap")
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue

    console.print(
        "  [yellow]nao foi possivel limpar swap automaticamente "
        "(execute como root ou permita sudo -n swapoff/swapon)[/yellow]"
    )
    return False


def prepare_for_call(stage: str) -> bool:
    if CLEAR_SWAP_BEFORE_EACH_CALL and get_swap_used_mb() >= SWAP_CLEAR_MIN_MB:
        if clear_swap_if_possible(force=True):
            time.sleep(SWAP_RECOVERY_SLEEP)

    if WAIT_BEFORE_EACH_CALL:
        return wait_system_stable(stage)
    return True


def wait_system_stable(stage: str) -> bool:
    # Antes de cada chamada relevante, o script espera CPU, memoria e swap
    # voltarem a uma faixa aceitavel para reduzir comparacoes injustas.
    console.print(f"\n[bold cyan]Aguardando estabilidade: {stage}[/bold cyan]")
    stable_count = 0
    start = time.time()

    while time.time() - start <= MAX_WAIT_STABILITY:
        cpu = psutil.cpu_percent(interval=1)
        load = get_load_average_1m()
        vm = psutil.virtual_memory()
        swap_used = get_swap_used_mb()
        swap_delta = get_swap_delta_mb()
        mem_available = vm.available / (1024 * 1024)
        cpu_times = psutil.cpu_times_percent(interval=0.1)
        iowait = getattr(cpu_times, "iowait", 0.0)

        ok = (
            cpu < CPU_THRESHOLD
            and load < LOAD_THRESHOLD
            and mem_available > MEM_MIN_AVAILABLE_MB
            and swap_used < SWAP_HARD_LIMIT_MB
            and swap_used <= SWAP_STABLE_MAX_MB
            and swap_delta < SWAP_DELTA_THRESHOLD_MB
            and iowait < IOWAIT_THRESHOLD
        )

        if ok:
            stable_count += 1
            console.print(
                f"  [dim green]OK {stable_count}/{STABILITY_WINDOW} "
                f"cpu={cpu:.1f}% load={load:.2f} mem={mem_available:.0f}MB "
                f"swap={swap_used:.0f}MB delta={swap_delta:.0f}MB iowait={iowait:.1f}%[/dim green]"
            )
        else:
            stable_count = 0
            reasons = []
            if cpu >= CPU_THRESHOLD:
                reasons.append(f"cpu>={CPU_THRESHOLD:g}%")
            if load >= LOAD_THRESHOLD:
                reasons.append(f"load>={LOAD_THRESHOLD:g}")
            if mem_available <= MEM_MIN_AVAILABLE_MB:
                reasons.append(f"mem<={MEM_MIN_AVAILABLE_MB:g}MB")
            if swap_used >= SWAP_HARD_LIMIT_MB:
                reasons.append(f"swap_hard>={SWAP_HARD_LIMIT_MB:g}MB")
            if swap_used > SWAP_STABLE_MAX_MB:
                reasons.append(f"swap>{SWAP_STABLE_MAX_MB:g}MB")
            if swap_delta >= SWAP_DELTA_THRESHOLD_MB:
                reasons.append(f"swap_delta>={SWAP_DELTA_THRESHOLD_MB:g}MB")
            if iowait >= IOWAIT_THRESHOLD:
                reasons.append(f"iowait>={IOWAIT_THRESHOLD:g}%")
            console.print(
                f"  [yellow]instavel cpu={cpu:.1f}% load={load:.2f} "
                f"mem={mem_available:.0f}MB swap={swap_used:.0f}MB delta={swap_delta:.0f}MB "
                f"iowait={iowait:.1f}% "
                f"({' ; '.join(reasons)})[/yellow]"
            )
            if is_swap_contaminated(swap_used, swap_delta) and CLEAR_SWAP_BEFORE_EACH_CALL:
                if clear_swap_if_possible(force=True):
                    time.sleep(SWAP_RECOVERY_SLEEP)

        if stable_count >= STABILITY_WINDOW:
            console.print("[bold green]Sistema estabilizado[/bold green]\n")
            return True

        time.sleep(STABILITY_INTERVAL)

    console.print("[bold red]Timeout aguardando estabilidade[/bold red]")
    return False


def p95(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    idx = min(len(clean) - 1, math.ceil(0.95 * len(clean)) - 1)
    return clean[idx]


def norm_high(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def norm_low(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(series), index=series.index)
    return (mx - series) / (mx - mn)


def ollama_generate(model: str, num_thread: int, prompt: str, warmup: bool = False) -> dict[str, Any]:
    # A opcao num_thread e o parametro estudado; o prompt fica fixo para que
    # diferencas venham principalmente da configuracao de CPU.
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_thread": num_thread,
            "seed": SEED,
        },
    }

    start = time.perf_counter()
    ttft = None
    text = ""
    final: dict[str, Any] = {}
    error = ""
    error_detail = ""
    status_code = None
    done_received = False
    done_reason = ""
    stream_chunks = 0

    try:
        r = requests.post(
            f"{BASE_URL}/api/generate",
            json=payload,
            timeout=TIMEOUT,
            stream=True,
        )
        status_code = r.status_code
        if r.status_code != 200:
            error = f"HTTP_{r.status_code}"
            try:
                error_detail = short_error_detail(r.text)
            except Exception:
                error_detail = ""
        else:
            for raw in r.iter_lines():
                if not raw:
                    continue
                stream_chunks += 1
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError as exc:
                    error = "STREAM_JSON_DECODE"
                    error_detail = short_error_detail(exc)
                    break
                if chunk.get("error"):
                    error = "OLLAMA_STREAM_ERROR"
                    error_detail = short_error_detail(chunk.get("error"))
                    break
                token = chunk.get("response", "")
                if token and ttft is None:
                    ttft = time.perf_counter() - start
                text += token
                if chunk.get("done"):
                    done_received = True
                    done_reason = str(chunk.get("done_reason") or "")
                    final = chunk
                    break
            if not error and not done_received:
                error = "STREAM_INCOMPLETE"
                error_detail = "stream terminou antes de done=true"
    except requests.exceptions.ConnectTimeout as exc:
        error = "CONNECT_TIMEOUT"
        error_detail = short_error_detail(exc)
    except requests.exceptions.ReadTimeout as exc:
        error = "READ_TIMEOUT"
        error_detail = short_error_detail(exc)
    except requests.exceptions.Timeout as exc:
        error = "TIMEOUT"
        error_detail = short_error_detail(exc)
    except requests.exceptions.ConnectionError as exc:
        error = "CONNECTION_ERROR"
        error_detail = short_error_detail(exc)
    except requests.exceptions.RequestException as exc:
        error = "REQUEST_ERROR"
        error_detail = short_error_detail(exc)
    except Exception as e:
        error = type(e).__name__
        error_detail = short_error_detail(e)

    dur = time.perf_counter() - start
    eval_count = int(final.get("eval_count") or 0)
    prompt_eval_count = int(final.get("prompt_eval_count") or 0)
    eval_duration_ns = int(final.get("eval_duration") or 0)
    prompt_eval_duration_ns = int(final.get("prompt_eval_duration") or 0)
    total_duration_ns = int(final.get("total_duration") or 0)
    load_duration_ns = int(final.get("load_duration") or 0)
    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    prompt_tps = (
        prompt_eval_count / (prompt_eval_duration_ns / 1e9)
        if prompt_eval_duration_ns > 0
        else 0.0
    )
    swap_used = get_swap_used_mb()
    swap_delta = get_swap_delta_mb()
    response_stripped = text.strip()
    expected_stripped = EXPECTED_RESPONSE.strip()
    response_nonempty = bool(response_stripped)
    response_exact_match = bool(expected_stripped) and response_stripped == expected_stripped
    response_markdown_fenced = "```" in text
    response_json_valid = False
    response_json_error = ""
    if response_stripped:
        try:
            json.loads(response_stripped)
            response_json_valid = True
        except Exception as exc:
            response_json_error = type(exc).__name__
    response_valid = (
        (response_exact_match if expected_stripped else response_nonempty)
        if VALIDATE_RESPONSE
        else True
    )
    response_issue = ""
    if not response_nonempty:
        response_issue = "EMPTY_RESPONSE"
    elif response_markdown_fenced:
        response_issue = "MARKDOWN_FENCE"
    elif expected_stripped and not response_exact_match:
        response_issue = "EXPECTED_MISMATCH"
    elif response_json_error:
        response_issue = f"JSON_{response_json_error}"
    if not error and not response_nonempty:
        error = "EMPTY_RESPONSE"

    return {
        "model": model,
        "num_thread": num_thread,
        "warmup": warmup,
        "error": error,
        "error_detail": error_detail,
        "recoverable_error": is_recoverable_error(error),
        "status_code": status_code,
        "done_received": done_received,
        "done_reason": done_reason,
        "stream_chunks": stream_chunks,
        "dur": dur if not error else None,
        "ttft": (ttft or dur) if not error else None,
        "tps": tps,
        "prompt_tps": prompt_tps,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "eval_duration_s": eval_duration_ns / 1e9,
        "prompt_eval_duration_s": prompt_eval_duration_ns / 1e9,
        "total_duration_s": total_duration_ns / 1e9,
        "load_duration_s": load_duration_ns / 1e9,
        "swap_used_mb": swap_used,
        "swap_delta_mb": swap_delta,
        "swap_baseline_mb": SWAP_BASELINE_MB,
        "swap_contaminated": is_swap_contaminated(swap_used, swap_delta),
        "rerun_used": False,
        "error_rerun_used": False,
        "invalid_response_rerun_used": False,
        "attempt_count": 1,
        "first_error": error,
        "first_error_detail": error_detail,
        "response_chars": len(text),
        "response_words": len(text.split()),
        "response_nonempty": response_nonempty,
        "response_exact_match": response_exact_match,
        "response_json_valid": response_json_valid,
        "response_json_error": response_json_error,
        "response_markdown_fenced": response_markdown_fenced,
        "response_issue": response_issue,
        "response_valid": response_valid,
        "response": text,
    }


def run_generate_fair(model: str, num_thread: int, warmup: bool, stage: str) -> dict[str, Any] | None:
    """
    Prepara o sistema, executa a chamada e repete problemas recuperaveis.
    Swap, erro transiente e resposta invalida sao rastreados separadamente.
    """
    last_row = None
    attempt_count = 0
    swap_reruns = 0
    error_reruns = 0
    invalid_response_reruns = 0
    first_error = ""
    first_error_detail = ""

    while True:
        if not prepare_for_call(stage):
            return None

        attempt_count += 1
        row = ollama_generate(model, num_thread, PROMPT, warmup=warmup)
        last_row = row
        if row.get("error") and not first_error:
            first_error = str(row.get("error") or "")
            first_error_detail = str(row.get("error_detail") or "")

        should_rerun = False
        rerun_reason = ""
        console.print(
            f"  [dim]tentativa {attempt_count}: erro={row.get('error') or '-'} "
            f"resp_valida={row.get('response_valid')} swap={row.get('swap_used_mb', 0):.0f}MB "
            f"delta={row.get('swap_delta_mb', 0):.0f}MB[/dim]"
        )

        if row.get("swap_contaminated") and swap_reruns < SWAP_CONTAMINATION_RERUNS:
            swap_reruns += 1
            should_rerun = True
            rerun_reason = "swap"
        elif (
            row.get("error")
            and row.get("recoverable_error")
            and error_reruns < ERROR_RERUNS
        ):
            error_reruns += 1
            should_rerun = True
            rerun_reason = f"erro recuperavel ({row.get('error')})"
        elif (
            VALIDATE_RESPONSE
            and not row.get("error")
            and not row.get("response_valid")
            and invalid_response_reruns < INVALID_RESPONSE_RERUNS
        ):
            invalid_response_reruns += 1
            should_rerun = True
            rerun_reason = f"resposta invalida ({row.get('response_issue') or 'INVALID_RESPONSE'})"

        if not should_rerun:
            row["attempt_count"] = attempt_count
            row["rerun_used"] = attempt_count > 1
            row["swap_rerun_used"] = swap_reruns > 0
            row["error_rerun_used"] = error_reruns > 0
            row["invalid_response_rerun_used"] = invalid_response_reruns > 0
            row["first_error"] = first_error or row.get("first_error", "")
            row["first_error_detail"] = first_error_detail or row.get("first_error_detail", "")
            return row

        console.print(
            f"  [yellow]repetindo {model} thread={num_thread}: {rerun_reason}[/yellow]"
        )
        if rerun_reason == "swap":
            console.print(
                f"  [red]swap contaminou {model} thread={num_thread} "
                f"(swap={row.get('swap_used_mb', 0):.0f}MB delta={row.get('swap_delta_mb', 0):.0f}MB)[/red]"
            )
        if row.get("swap_contaminated") or row.get("error"):
            unload_model(model, f"antes de repetir chamada por {rerun_reason}")
        if row.get("swap_contaminated"):
            if clear_swap_if_possible(force=True):
                time.sleep(SWAP_RECOVERY_SLEEP)
            reset_swap_baseline("antes de repetir chamada")
        else:
            time.sleep(min(SWAP_RECOVERY_SLEEP, 2))

    if last_row is not None:
        last_row["attempt_count"] = attempt_count
        last_row["rerun_used"] = attempt_count > 1
        last_row["swap_rerun_used"] = swap_reruns > 0
        last_row["error_rerun_used"] = error_reruns > 0
        last_row["invalid_response_rerun_used"] = invalid_response_reruns > 0
        last_row["first_error"] = first_error or last_row.get("first_error", "")
        last_row["first_error_detail"] = first_error_detail or last_row.get("first_error_detail", "")
    return last_row


def make_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Remove warmups do ranking e calcula um score normalizado por modelo,
    # combinando velocidade, latencia, confiabilidade, ausencia de swap e
    # validade basica da resposta.
    measured = df[df["warmup"] == False].copy()  # noqa: E712
    for col, default in {
        "response_nonempty": True,
        "response_exact_match": False,
        "response_json_valid": False,
        "response_markdown_fenced": False,
        "response_valid": True,
        "recoverable_error": False,
        "done_received": True,
        "rerun_used": False,
        "swap_rerun_used": False,
        "error_rerun_used": False,
        "invalid_response_rerun_used": False,
        "attempt_count": 1,
    }.items():
        if col not in measured.columns:
            measured[col] = default

    if measured.empty:
        summary = pd.DataFrame(columns=["model", "num_thread"])
    else:
        summary = (
            measured.groupby(["model", "num_thread"])
            .agg(
                runs=("num_thread", "count"),
                errors=("error", lambda x: x.fillna("").astype(str).ne("").sum()),
                error_rate=("error", lambda x: x.fillna("").astype(str).ne("").mean()),
                recoverable_error_rate=("recoverable_error", "mean"),
                done_rate=("done_received", "mean"),
                median_latency=("dur", "median"),
                mean_latency=("dur", "mean"),
                p95_latency=("dur", lambda x: p95(list(x.dropna()))),
                median_ttft=("ttft", "median"),
                mean_ttft=("ttft", "mean"),
                mean_tps=("tps", "mean"),
                median_tps=("tps", "median"),
                mean_prompt_tps=("prompt_tps", "mean"),
                mean_eval_count=("eval_count", "mean"),
                mean_prompt_eval_count=("prompt_eval_count", "mean"),
                mean_response_chars=("response_chars", "mean"),
                mean_response_words=("response_words", "mean"),
                response_nonempty_rate=("response_nonempty", "mean"),
                response_exact_rate=("response_exact_match", "mean"),
                response_json_valid_rate=("response_json_valid", "mean"),
                response_valid_rate=("response_valid", "mean"),
                markdown_fence_rate=("response_markdown_fenced", "mean"),
                swap_peak_mb=("swap_used_mb", "max"),
                swap_delta_peak_mb=("swap_delta_mb", "max"),
                swap_contamination_rate=("swap_contaminated", "mean"),
                rerun_rate=("rerun_used", "mean"),
                swap_rerun_rate=("swap_rerun_used", "mean"),
                error_rerun_rate=("error_rerun_used", "mean"),
                invalid_response_rerun_rate=("invalid_response_rerun_used", "mean"),
                mean_attempt_count=("attempt_count", "mean"),
            )
            .reset_index()
        )

    expected_grid = pd.MultiIndex.from_product(
        [MODELS, THREAD_LEVELS], names=["model", "num_thread"]
    ).to_frame(index=False)
    summary = expected_grid.merge(summary, on=["model", "num_thread"], how="left")
    defaults = {
        "runs": 0,
        "errors": REPEATS,
        "error_rate": 1.0,
        "recoverable_error_rate": 0.0,
        "done_rate": 0.0,
        "median_latency": TIMEOUT,
        "mean_latency": TIMEOUT,
        "p95_latency": TIMEOUT,
        "median_ttft": TIMEOUT,
        "mean_ttft": TIMEOUT,
        "mean_tps": 0.0,
        "median_tps": 0.0,
        "mean_prompt_tps": 0.0,
        "mean_eval_count": 0.0,
        "mean_prompt_eval_count": 0.0,
        "mean_response_chars": 0.0,
        "mean_response_words": 0.0,
        "response_nonempty_rate": 0.0,
        "response_exact_rate": 0.0,
        "response_json_valid_rate": 0.0,
        "response_valid_rate": 0.0,
        "markdown_fence_rate": 0.0,
        "swap_peak_mb": 0.0,
        "swap_delta_peak_mb": 0.0,
        "swap_contamination_rate": 1.0,
        "rerun_rate": 1.0,
        "swap_rerun_rate": 0.0,
        "error_rerun_rate": 0.0,
        "invalid_response_rerun_rate": 0.0,
        "mean_attempt_count": 0.0,
    }
    for col, value in defaults.items():
        if col not in summary.columns:
            summary[col] = value
        summary[col] = summary[col].fillna(value)
    summary["expected_runs"] = REPEATS
    summary["run_coverage"] = (summary["runs"] / max(REPEATS, 1)).clip(upper=1.0)
    summary["complete_thread_run"] = summary["runs"] >= REPEATS

    scored_parts = []
    for _, part in summary.groupby("model", sort=False):
        part = part.copy()
        part["lat_norm"] = norm_low(part["median_latency"].fillna(part["median_latency"].max()))
        part["ttft_norm"] = norm_low(part["median_ttft"].fillna(part["median_ttft"].max()))
        part["tps_norm"] = norm_high(part["mean_tps"].fillna(0.0))
        part["reliability"] = 1.0 - part["error_rate"].fillna(1.0)
        part["swap_cleanliness"] = 1.0 - part["swap_contamination_rate"].fillna(1.0)
        part["output_validity"] = part["response_valid_rate"].fillna(0.0)
        part["retry_cleanliness"] = 1.0 - (
            0.50 * part["error_rerun_rate"].fillna(1.0)
            + 0.30 * part["swap_rerun_rate"].fillna(1.0)
            + 0.20 * part["invalid_response_rerun_rate"].fillna(1.0)
        ).clip(lower=0.0, upper=1.0)
        part["thread_score_raw"] = (
            0.40 * part["tps_norm"]
            + 0.27 * part["lat_norm"]
            + 0.13 * part["ttft_norm"]
            + 0.07 * part["reliability"]
            + 0.04 * part["swap_cleanliness"]
            + 0.05 * part["output_validity"]
            + 0.02 * part["retry_cleanliness"]
            + 0.02 * part["run_coverage"]
        )
        part["thread_score"] = part["thread_score_raw"] * part["run_coverage"]
        scored_parts.append(part)

    summary = pd.concat(scored_parts, ignore_index=True)
    ranking = (
        summary.sort_values(
            ["model", "complete_thread_run", "thread_score", "mean_tps", "median_latency"],
            ascending=[True, False, False, False, True],
        )
        .groupby("model")
        .head(1)
        .sort_values(["complete_thread_run", "thread_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary, ranking


def build_run_config() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url_masked": mask_sensitive(BASE_URL),
        "models": MODELS,
        "thread_levels": THREAD_LEVELS,
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "timeout_s": TIMEOUT,
        "keep_alive": KEEP_ALIVE,
        "num_predict": NUM_PREDICT,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "validate_response": VALIDATE_RESPONSE,
        "expected_response_chars": len(EXPECTED_RESPONSE),
        "min_repeats_for_p95": MIN_REPEATS_FOR_P95,
        "cpu_threshold": CPU_THRESHOLD,
        "load_threshold": LOAD_THRESHOLD,
        "mem_min_available_mb": MEM_MIN_AVAILABLE_MB,
        "swap_stable_max_mb": SWAP_STABLE_MAX_MB,
        "swap_delta_threshold_mb": SWAP_DELTA_THRESHOLD_MB,
        "swap_hard_limit_mb": SWAP_HARD_LIMIT_MB,
        "swap_clear_min_mb": SWAP_CLEAR_MIN_MB,
        "iowait_threshold": IOWAIT_THRESHOLD,
        "stability_window": STABILITY_WINDOW,
        "stability_interval": STABILITY_INTERVAL,
        "max_wait_stability": MAX_WAIT_STABILITY,
        "clear_swap_before_start": CLEAR_SWAP_BEFORE_START,
        "clear_swap_before_each_call": CLEAR_SWAP_BEFORE_EACH_CALL,
        "clear_swap_after_model": CLEAR_SWAP_AFTER_MODEL,
        "swap_contamination_reruns": SWAP_CONTAMINATION_RERUNS,
        "error_reruns": ERROR_RERUNS,
        "invalid_response_reruns": INVALID_RESPONSE_RERUNS,
        "swap_recovery_sleep": SWAP_RECOVERY_SLEEP,
        "unload_before_model": UNLOAD_BEFORE_MODEL,
        "unload_after_model": UNLOAD_AFTER_MODEL,
        "local_environment": collect_local_environment(),
        "declared_server_environment": collect_declared_server_environment(),
        "ollama_metadata": collect_ollama_metadata(),
        "telemetry_note": (
            "CPU/memoria/swap sao lidos no processo local. Se OLLAMA_BASE_URL nao for local, "
            "esses dados podem nao representar o host de inferencia."
        ),
        "prompt": PROMPT,
    }


def write_config(config: dict[str, Any]) -> str:
    path = os.path.join(OUT_DIR, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def save_figures(summary: pd.DataFrame):
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#4a5568"]
    for model, part in summary.groupby("model"):
        safe = model.replace(":", "_").replace("/", "_")
        part = part.sort_values("num_thread")

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(part["num_thread"], part["median_latency"], marker="o", label="Latencia mediana (s)", color="#2b6cb0")
        ax2.plot(part["num_thread"], part["mean_tps"], marker="s", label="TPS medio", color="#c05621")
        ax1.set_xlabel("num_thread")
        ax1.set_ylabel("Latencia mediana (s)", color="#2b6cb0")
        ax2.set_ylabel("Tokens/s medio", color="#c05621")
        ax1.set_title(f"Threads x desempenho - {model}")
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{safe}_latency_tps.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(part["num_thread"], part["thread_score"], marker="o", color="#2f855a")
        ax.set_xlabel("num_thread")
        ax.set_ylabel("thread_score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Score de desempenho por threads - {model}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{safe}_thread_score.png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx, (model, part) in enumerate(summary.groupby("model")):
        part = part.sort_values("num_thread")
        color = colors[idx % len(colors)]
        axes[0].plot(part["num_thread"], part["mean_tps"], marker="o", label=model, color=color)
        axes[1].plot(part["num_thread"], part["median_latency"], marker="s", label=model, color=color)

    axes[0].set_title("Comparativo dos modelos por numero de threads")
    axes[0].set_ylabel("Tokens/s medio")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("num_thread")
    axes[1].set_ylabel("Latencia mediana (s)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "combined_models_threads.png"), dpi=180)
    plt.close(fig)


def write_detailed_report(df: pd.DataFrame, summary: pd.DataFrame, ranking: pd.DataFrame, config: dict[str, Any]) -> tuple[str, str]:
    measured = df[df["warmup"] == False].copy()  # noqa: E712
    for col, default in {
        "response_nonempty": True,
        "response_exact_match": False,
        "response_json_valid": False,
        "response_markdown_fenced": False,
        "response_valid": True,
    }.items():
        if col not in measured.columns:
            measured[col] = default
    total_expected = len(MODELS) * len(THREAD_LEVELS) * REPEATS
    total_measured = len(measured)
    total_warmups = int((df["warmup"] == True).sum())  # noqa: E712
    error_rate = measured["error"].fillna("").astype(str).ne("").mean() if total_measured else 1.0
    swap_rate = measured["swap_contaminated"].mean() if total_measured else 1.0
    rerun_rate = measured["rerun_used"].mean() if total_measured else 0.0
    coverage_rate = total_measured / total_expected if total_expected else 0.0
    response_valid_rate = measured["response_valid"].mean() if total_measured else 0.0
    response_exact_rate = measured["response_exact_match"].mean() if total_measured else 0.0
    response_empty_rate = (1.0 - measured["response_nonempty"].mean()) if total_measured else 1.0

    def fmt(value: Any, digits: int = 3) -> str:
        if pd.isna(value):
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    md_lines = [
        "# Relatorio detalhado do benchmark de threads",
        "",
        f"Executado em `{config['generated_at']}` contra `{config['base_url_masked']}`.",
        "",
        "## Configuracao",
        "",
        f"- Modelos: `{', '.join(MODELS)}`",
        f"- Threads testadas: `{', '.join(map(str, THREAD_LEVELS))}`",
        f"- Repeticoes medidas por thread: `{REPEATS}`",
        f"- Warmups por thread: `{WARMUPS}`",
        f"- Timeout: `{TIMEOUT}s`; num_predict: `{NUM_PREDICT}`; keep_alive: `{KEEP_ALIVE}`; temperatura: `{TEMPERATURE}`",
        f"- Validacao de resposta: `{VALIDATE_RESPONSE}`; resposta esperada: `{len(EXPECTED_RESPONSE)}` caracteres",
        f"- Limpeza de swap: inicio={CLEAR_SWAP_BEFORE_START}, antes de cada chamada={CLEAR_SWAP_BEFORE_EACH_CALL}, apos modelo={CLEAR_SWAP_AFTER_MODEL}, reruns={SWAP_CONTAMINATION_RERUNS}",
        "",
        "## Cobertura e confiabilidade",
        "",
        f"- Chamadas medidas esperadas: `{total_expected}`",
        f"- Chamadas medidas registradas: `{total_measured}`",
        f"- Cobertura geral do plano: `{coverage_rate:.1%}`",
        f"- Warmups registrados: `{total_warmups}`",
        f"- Taxa geral de erro nas chamadas medidas: `{error_rate:.1%}`",
        f"- Taxa geral de swap contaminado nas chamadas medidas: `{swap_rate:.1%}`",
        f"- Taxa de chamadas refeitas por swap: `{rerun_rate:.1%}`",
        f"- Taxa de respostas validas: `{response_valid_rate:.1%}`",
        f"- Taxa de respostas exatamente iguais ao esperado: `{response_exact_rate:.1%}`",
        f"- Taxa de respostas vazias: `{response_empty_rate:.1%}`",
        "",
        "## Melhor configuracao por modelo",
        "",
        "| Modelo | num_thread | Score | Cob. | TPS medio | Lat.med | P95 lat | TTFT.med | Erro | Resp.valida | Swap | Rerun |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking.itertuples():
        md_lines.append(
            f"| {row.model} | {row.num_thread} | {row.thread_score:.3f} | "
            f"{row.run_coverage:.0%} | {row.mean_tps:.2f} | {fmt(row.median_latency, 1)}s | {fmt(row.p95_latency, 1)}s | "
            f"{fmt(row.median_ttft, 1)}s | {row.error_rate:.0%} | "
            f"{row.response_valid_rate:.0%} | {row.swap_contamination_rate:.0%} | {row.rerun_rate:.0%} |"
        )
    if REPEATS < MIN_REPEATS_FOR_P95:
        md_lines.extend([
            "",
            f"> Aviso: `THREAD_REPEATS={REPEATS}` esta abaixo de `THREAD_MIN_REPEATS_FOR_P95={MIN_REPEATS_FOR_P95}`. "
            "O P95 deve ser tratado como indicio exploratorio, nao como estimativa robusta de cauda.",
        ])

    md_lines.extend(["", "## Resultado completo por modelo e thread", ""])
    for model, part in summary.groupby("model"):
        md_lines.append(f"### {model}")
        md_lines.append("")
        md_lines.append("| num_thread | Score | Cob. | TPS | Lat.med | Lat.media | P95 | TTFT.med | Erro | Resp.valida | Resp.exata | JSON | Markdown | Swap pico | Delta swap | Swap contam. | Rerun | Tent. |")
        md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in part.sort_values("num_thread").itertuples():
            md_lines.append(
                f"| {row.num_thread} | {row.thread_score:.3f} | {row.run_coverage:.0%} | {row.mean_tps:.2f} | "
                f"{fmt(row.median_latency, 1)}s | {fmt(row.mean_latency, 1)}s | {fmt(row.p95_latency, 1)}s | "
                f"{fmt(row.median_ttft, 1)}s | {row.error_rate:.0%} | {row.response_valid_rate:.0%} | "
                f"{row.response_exact_rate:.0%} | {row.response_json_valid_rate:.0%} | {row.markdown_fence_rate:.0%} | "
                f"{row.swap_peak_mb:.0f}MB | {row.swap_delta_peak_mb:.0f}MB | {row.swap_contamination_rate:.0%} | "
                f"{row.rerun_rate:.0%} | {row.mean_attempt_count:.1f} |"
            )
        md_lines.append("")

    md_lines.extend([
        "## Observacoes metodologicas",
        "",
        "O ranking ignora chamadas de warmup e calcula o `thread_score` dentro de cada modelo. "
        "Isso evita comparar diretamente modelos com tamanhos e velocidades muito diferentes ao escolher o melhor `num_thread` de cada um.",
        "",
        "O `thread_score_raw` combina `0.40*tps_norm + 0.27*lat_norm + 0.13*ttft_norm + 0.07*reliability + "
        "0.04*swap_cleanliness + 0.05*output_validity + 0.02*retry_cleanliness + 0.02*run_coverage`; "
        "o `thread_score` final multiplica esse valor pela cobertura da combinacao modelo/thread. "
        "Logo, configuracoes com mais tokens/s, menor latencia, menor TTFT, sem erros, sem swap, sem reruns e com resposta valida recebem maior pontuacao.",
        "",
        "As metricas de CPU, memoria e swap sao coletadas pelo processo local. "
        "Quando o endpoint Ollama nao e local, esses indicadores devem ser tratados como telemetria do cliente, nao do servidor de inferencia.",
    ])

    md_path = os.path.join(OUT_DIR, "detailed_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    ranking_html = ranking.to_html(index=False, float_format=lambda x: f"{x:.3f}", classes="table")
    summary_html = summary.sort_values(["model", "num_thread"]).to_html(index=False, float_format=lambda x: f"{x:.3f}", classes="table")
    error_rows = measured[measured["error"].fillna("").astype(str).ne("")]
    errors_html = (
        error_rows.groupby(["model", "num_thread", "error"])
        .size()
        .reset_index(name="count")
        .to_html(index=False, classes="table")
        if not error_rows.empty
        else "<p>Nenhum erro registrado nas chamadas medidas.</p>"
    )
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark de threads Ollama</title>
<style>
body{{font-family:Arial,sans-serif;margin:2rem;color:#172033;background:#f8fafc}}
h1,h2{{color:#102a43}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1rem 0}}
.kpi{{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:1rem}}
.val{{font-size:1.5rem;font-weight:bold;color:#0b7285}}
.lbl{{font-size:.8rem;color:#52606d}}
.table{{border-collapse:collapse;width:100%;background:white;margin:1rem 0;font-size:.85rem}}
.table th,.table td{{border:1px solid #d9e2ec;padding:.35rem .5rem;text-align:right}}
.table th:first-child,.table td:first-child{{text-align:left}}
.table th{{background:#e3f2fd;color:#102a43}}
code{{background:#eef2f7;padding:.1rem .25rem;border-radius:4px}}
</style>
</head>
<body>
<h1>Benchmark de threads Ollama</h1>
<p>Executado em <code>{config['generated_at']}</code> contra <code>{config['base_url_masked']}</code>.</p>
<div class="grid">
<div class="kpi"><div class="val">{total_measured}/{total_expected}</div><div class="lbl">chamadas medidas</div></div>
<div class="kpi"><div class="val">{error_rate:.1%}</div><div class="lbl">erro geral</div></div>
<div class="kpi"><div class="val">{swap_rate:.1%}</div><div class="lbl">swap contaminado</div></div>
<div class="kpi"><div class="val">{rerun_rate:.1%}</div><div class="lbl">reruns por swap</div></div>
</div>
<h2>Configuração</h2>
<pre>{json.dumps(config, ensure_ascii=False, indent=2)}</pre>
<h2>Melhor configuração por modelo</h2>
{ranking_html}
<h2>Resultado completo por modelo e thread</h2>
{summary_html}
<h2>Erros por modelo/thread</h2>
{errors_html}
</body>
</html>"""
    html_path = os.path.join(OUT_DIR, "detailed_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return md_path, html_path


def write_recommendation(summary: pd.DataFrame, ranking: pd.DataFrame):
    lines = [
        "# Recomendacao de num_thread para Ollama em CPU",
        "",
        f"Executado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} contra `{mask_sensitive(BASE_URL)}`.",
        "",
        "O `thread_score` mede desempenho por modelo, normalizado dentro de cada modelo:",
        "",
        "`thread_score_raw = 0.40*tps_norm + 0.27*lat_norm + 0.13*ttft_norm + 0.07*reliability + 0.04*swap_cleanliness + 0.05*output_validity + 0.02*retry_cleanliness + 0.02*run_coverage`",
        "",
        "`thread_score = thread_score_raw * run_coverage`",
        "",
        "Assim, ele favorece configuracoes com maior geracao de tokens/s, menor latencia, menor tempo ate o primeiro token, sem erros, sem swap, sem reruns e com resposta valida.",
        "",
        "## Melhor configuracao por modelo",
        "",
        "| Modelo | num_thread | Score | Cob. | TPS medio | Lat.med | TTFT.med | Erro | Resp.valida | Swap | Rerun |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in ranking.itertuples():
        lines.append(
            f"| {row.model} | {row.num_thread} | {row.thread_score:.3f} | "
            f"{row.run_coverage:.0%} | {row.mean_tps:.2f} | {row.median_latency:.1f}s | "
            f"{row.median_ttft:.1f}s | {row.error_rate:.0%} | "
            f"{row.response_valid_rate:.0%} | {row.swap_contamination_rate:.0%} | {row.rerun_rate:.0%} |"
        )

    lines.extend(["", "## Ranking completo por modelo", ""])
    for model, part in summary.groupby("model"):
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| num_thread | Score | Cob. | TPS | Lat.med | TTFT.med | Erro | Resp.valida | Swap pico | Rerun |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in part.sort_values("thread_score", ascending=False).itertuples():
            lines.append(
                f"| {row.num_thread} | {row.thread_score:.3f} | {row.run_coverage:.0%} | {row.mean_tps:.2f} | "
                f"{row.median_latency:.1f}s | {row.median_ttft:.1f}s | "
                f"{row.error_rate:.0%} | {row.response_valid_rate:.0%} | {row.swap_peak_mb:.0f}MB | {row.rerun_rate:.0%} |"
            )
        lines.append("")

    with open(os.path.join(OUT_DIR, "recommendation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_final_table(ranking: pd.DataFrame):
    table = Table(title="Melhor num_thread por modelo")
    table.add_column("Modelo")
    table.add_column("Threads", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Cob.", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("Lat.med", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Erro", justify="right")
    table.add_column("Resp", justify="right")
    table.add_column("Swap", justify="right")
    table.add_column("Rerun", justify="right")
    for row in ranking.itertuples():
        table.add_row(
            row.model,
            str(row.num_thread),
            f"{row.thread_score:.3f}",
            f"{row.run_coverage:.0%}",
            f"{row.mean_tps:.2f}",
            f"{row.median_latency:.1f}s",
            f"{row.median_ttft:.1f}s",
            f"{row.error_rate:.0%}",
            f"{row.response_valid_rate:.0%}",
            f"{row.swap_contamination_rate:.0%}",
            f"{row.rerun_rate:.0%}",
        )
    console.print(table)


# ============================================================================
# RUN
# ============================================================================
# Execucao do plano: para cada modelo e cada num_thread, roda warmups e
# repeticoes medidas, salva dados brutos e gera recomendacao.

console.print(
    Panel(
        f"[bold cyan]Benchmark de threads para Ollama em CPU[/bold cyan]\n"
        f"URL: [yellow]{mask_sensitive(BASE_URL)}[/yellow]\n"
        f"Modelos: [green]{', '.join(MODELS)}[/green]\n"
        f"Threads: [green]{', '.join(map(str, THREAD_LEVELS))}[/green]\n"
        f"Repeats: {REPEATS}  Warmups/thread: {WARMUPS}  Timeout: {TIMEOUT}s\n"
        f"num_predict={NUM_PREDICT} keep_alive={KEEP_ALIVE} temp={TEMPERATURE}\n"
        f"validate_response={VALIDATE_RESPONSE} expected_chars={len(EXPECTED_RESPONSE)} "
        f"min_repeats_p95={MIN_REPEATS_FOR_P95}\n"
        f"clear_swap=start:{CLEAR_SWAP_BEFORE_START} each_call:{CLEAR_SWAP_BEFORE_EACH_CALL} "
        f"after_model:{CLEAR_SWAP_AFTER_MODEL} min={SWAP_CLEAR_MIN_MB:g}MB "
        f"swap_delta={SWAP_DELTA_THRESHOLD_MB:g}MB iowait={IOWAIT_THRESHOLD:g}%\n"
        f"reruns: swap={SWAP_CONTAMINATION_RERUNS}x erro={ERROR_RERUNS}x "
        f"resp_invalida={INVALID_RESPONSE_RERUNS}x\n"
        f"Prompt unico: {PROMPT[:120]}...",
        title="benchmark_threads.py",
        border_style="blue",
    )
)

records: list[dict[str, Any]] = []

if CLEAR_SWAP_BEFORE_START:
    had_swap_to_clear = get_swap_used_mb() >= SWAP_CLEAR_MIN_MB
    if clear_swap_if_possible(force=had_swap_to_clear) and had_swap_to_clear:
        time.sleep(SWAP_RECOVERY_SLEEP)
    reset_swap_baseline("inicio do benchmark")

for model in MODELS:
    console.rule(f"[bold cyan]{model}[/bold cyan]")
    if UNLOAD_BEFORE_MODEL:
        unload_model(model, "antes do modelo")
        if CLEAR_SWAP_BEFORE_EACH_CALL:
            had_swap_to_clear = get_swap_used_mb() >= SWAP_CLEAR_MIN_MB
            if clear_swap_if_possible(force=had_swap_to_clear) and had_swap_to_clear:
                time.sleep(SWAP_RECOVERY_SLEEP)
            reset_swap_baseline("antes do modelo")

    if not wait_system_stable(f"antes de iniciar {model}"):
        console.print(f"[bold red]Modelo {model} ignorado: sistema instavel[/bold red]")
        continue

    for num_thread in THREAD_LEVELS:
        console.print(f"\n[bold]Modelo={model} num_thread={num_thread}[/bold]")

        for warmup_idx in range(WARMUPS):
            row = run_generate_fair(
                model,
                num_thread,
                warmup=True,
                stage=f"{model} thread={num_thread} warmup={warmup_idx + 1}",
            )
            if row is None:
                console.print(f"  [red]warmup {warmup_idx + 1}/{WARMUPS}: sistema instavel[/red]")
                continue
            row.update({"rep": None, "warmup_idx": warmup_idx})
            records.append(row)
            console.print(
                f"  [dim]warmup {warmup_idx + 1}/{WARMUPS}: "
                f"erro={row['error'] or '-'} dur={row['dur'] or 0:.1f}s "
                f"tps={row['tps']:.2f} swap={row['swap_used_mb']:.0f}MB[/dim]"
            )

        for rep in range(REPEATS):
            row = run_generate_fair(
                model,
                num_thread,
                warmup=False,
                stage=f"{model} thread={num_thread} rep={rep + 1}",
            )
            if row is None:
                row = {
                    "model": model,
                    "num_thread": num_thread,
                    "warmup": False,
                    "error": "UNSTABLE_SYSTEM",
                    "error_detail": "sistema nao atingiu estabilidade antes da chamada",
                    "recoverable_error": False,
                    "status_code": None,
                    "done_received": False,
                    "done_reason": "",
                    "stream_chunks": 0,
                    "dur": None,
                    "ttft": None,
                    "tps": 0.0,
                    "prompt_tps": 0.0,
                    "eval_count": 0,
                    "prompt_eval_count": 0,
                    "eval_duration_s": 0.0,
                    "prompt_eval_duration_s": 0.0,
                    "total_duration_s": 0.0,
                    "load_duration_s": 0.0,
                    "swap_used_mb": get_swap_used_mb(),
                    "swap_delta_mb": get_swap_delta_mb(),
                    "swap_baseline_mb": SWAP_BASELINE_MB,
                    "swap_contaminated": is_swap_contaminated(),
                    "rerun_used": False,
                    "swap_rerun_used": False,
                    "error_rerun_used": False,
                    "invalid_response_rerun_used": False,
                    "attempt_count": 0,
                    "first_error": "UNSTABLE_SYSTEM",
                    "first_error_detail": "sistema nao atingiu estabilidade antes da chamada",
                    "response_chars": 0,
                    "response_words": 0,
                    "response_nonempty": False,
                    "response_exact_match": False,
                    "response_json_valid": False,
                    "response_json_error": "",
                    "response_markdown_fenced": False,
                    "response_issue": "NO_CALL",
                    "response_valid": False,
                    "response": "",
                }
            row.update({"rep": rep, "warmup_idx": None})
            records.append(row)
            console.print(
                f"  rep {rep + 1}/{REPEATS}: erro={row['error'] or '-'} "
                f"dur={row['dur'] or 0:.1f}s ttft={row['ttft'] or 0:.1f}s "
                f"tps={row['tps']:.2f} tokens={row['eval_count']} swap={row['swap_used_mb']:.0f}MB"
            )

    if UNLOAD_AFTER_MODEL:
        unload_model(model, "apos modelo")
    if CLEAR_SWAP_AFTER_MODEL:
        had_swap_to_clear = get_swap_used_mb() >= SWAP_CLEAR_MIN_MB
        if clear_swap_if_possible(force=had_swap_to_clear) and had_swap_to_clear:
            time.sleep(SWAP_RECOVERY_SLEEP)
        reset_swap_baseline("apos modelo")


df = pd.DataFrame(records)
raw_path = os.path.join(OUT_DIR, "results_raw.csv")
df.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)

if df.empty:
    console.print("[bold red]Nenhum registro gerado.[/bold red]")
    sys.exit(1)

summary, ranking = make_summary(df)
config = build_run_config()
config_path = write_config(config)
summary_path = os.path.join(OUT_DIR, "summary.csv")
ranking_path = os.path.join(OUT_DIR, "ranking.csv")
summary.to_csv(summary_path, index=False)
ranking.to_csv(ranking_path, index=False)
save_figures(summary)
write_recommendation(summary, ranking)
detailed_md_path, detailed_html_path = write_detailed_report(df, summary, ranking, config)
print_final_table(ranking)

console.print(
    Panel(
        "\n".join(
            [
                f"Arquivos salvos em {OUT_DIR}",
                f"- {raw_path}",
                f"- {summary_path}",
                f"- {ranking_path}",
                f"- {config_path}",
                f"- {os.path.join(OUT_DIR, 'recommendation.md')}",
                f"- {detailed_md_path}",
                f"- {detailed_html_path}",
                f"- {os.path.join(FIG_DIR, 'combined_models_threads.png')}",
                f"- {FIG_DIR}/",
            ]
        ),
        title="Concluido",
        border_style="green",
    )
)

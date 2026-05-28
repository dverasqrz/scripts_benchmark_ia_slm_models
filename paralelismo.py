"""
Benchmark LLM — versao conceitual correta
==========================================
Correção principal: metricas separadas
  - semantic_acc : modelo sabia a resposta real? (estado/capital corretos)
  - strict_acc   : modelo obedeceu o formato? (JSON perfeito)
  - codigo_acc   : modelo cumpriu a restricao numerica artificial?
  - parallel_score : modelo escala sob concorrencia sem perder qualidade?

Score normalizado:
  score = 0.30*semantic + 0.15*strict + 0.25*parallel + 0.15*tps_norm + 0.10*lat_norm + 0.05*codigo

Historico de correções:
  v1 -> v2: RPS correto, erros HTTP registrados, warmup, TTFT, retry
  v2 -> v3: dual accuracy, score equilibrado, classificacao de erros por camada
  v3 -> v4: prompt sem exemplo fixo, cold start com keep_alive=0,
            diversidade de codigo, score min-max e samples por categoria
"""

import time, threading, statistics, json, signal, sys, os, re, subprocess, importlib, shutil, gc
from datetime import datetime


def load_env_file(filename: str = ".env") -> None:
    """
    Carrega variaveis de ambiente locais sem depender de python-dotenv.
    Valores ja definidos no sistema operacional têm prioridade sobre o .env.
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
    """Oculta endpoints, webhooks e identificadores antes de imprimir relatórios."""
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


# Carregue primeiro o .env: o endpoint real do Ollama, webhooks e IDs ficam fora do Git.
load_env_file()


def _ensure_package(package: str, import_name: str | None = None):
    """Instala dependencias ausentes para permitir executar o benchmark em maquina limpa."""
    try:
        return importlib.import_module(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return importlib.import_module(import_name or package)


requests = _ensure_package("requests")
psutil = _ensure_package("psutil")
pd = _ensure_package("pandas")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
matplotlib = _ensure_package("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
np = _ensure_package("numpy")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.rule import Rule
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.rule import Rule

console = Console()

# ============================================================================
#  CONFIG
# ============================================================================
# Esta area concentra tudo que muda entre ambientes: endpoint, modelos,
# concorrencia, timeouts, estabilidade e pesos do score. Prefira alterar via .env
# quando a informacao for sensivel ou variar entre maquinas.
# Endpoint padrao seguro para Git. O endpoint real deve ficar em .env como OLLAMA_BASE_URL.
BASE_URL       = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
ACTIVE_BASE_URL = BASE_URL
OLLAMA_DIRECT_URLS = [
    u.strip().rstrip("/")
    for u in os.environ.get("OLLAMA_DIRECT_URL", "").split(",")
    if u.strip()
]
MODELS         = [
    m.strip()
    for m in os.environ.get(
        "BENCH_MODELS",
        (
            "gemma4:e2b,qwen3.5:9b,gemma2:9b,qwen3:8b,"
            "qwen2.5:7b,granite3.3:8b,llama3.1:8b,qwen3.5:4b"
        ),
    ).split(",")
    if m.strip()
]
REQUEST_LEVELS = [1, 3, 5]
REPEATS        = {1: 2, 3: 2, 5: 1}
EXPECTED_RECORDS_PER_MODEL = sum(n * REPEATS.get(n, 1) for n in REQUEST_LEVELS)
EXPECTED_LEVELS_PER_MODEL = len(REQUEST_LEVELS)
COOLDOWN       = 5
TIMEOUT        = 300
WARMUP_REPS    = 2
MAX_RETRIES    = 1
OUT_DIR        = "benchmark_results"
SAMPLES_DIR    = os.path.join(OUT_DIR, "samples")
SAMPLES_PER_CATEGORY = 3

# Cold start em Ollama: keep_alive=0 pede descarregamento do modelo apos a chamada.
# Em servidor remoto/compartilhado isso melhora o isolamento, mas nao substitui
# reiniciar o container quando for preciso provar cold start absoluto.
COLD_START     = True
KEEP_ALIVE_WARM = "5m"
KEEP_ALIVE_COLD = 0
COLD_UNLOAD_WAIT = 1.0
UNLOAD_AFTER_MODEL = True
UNLOAD_BEFORE_MODEL = True
MODEL_UNLOAD_WAIT = 3.0

# ===============================
# ESTABILIDADE DO SISTEMA
# ===============================
CPU_THRESHOLD = float(os.environ.get("BENCH_CPU_THRESHOLD", "50"))          # %
LOAD_THRESHOLD = float(os.environ.get("BENCH_LOAD_THRESHOLD", "1.5"))        # load average 1 min
SWAP_STABLE_MAX_MB = float(os.environ.get("BENCH_SWAP_STABLE_MAX_MB", "128"))  # swap absoluto aceitavel
SWAP_DELTA_THRESHOLD_MB = float(os.environ.get("BENCH_SWAP_DELTA_THRESHOLD_MB", "256"))  # crescimento aceitavel vs baseline
SWAP_HARD_LIMIT_MB = float(os.environ.get("BENCH_SWAP_HARD_LIMIT_MB", "2048"))      # limite absoluto de seguranca
SWAP_CLEAR_MIN_MB = float(os.environ.get("BENCH_SWAP_CLEAR_MIN_MB", "1"))
MEM_MIN_AVAILABLE_MB = float(os.environ.get("BENCH_MEM_MIN_AVAILABLE_MB", "500"))  # memoria minima disponivel
IOWAIT_THRESHOLD = float(os.environ.get("BENCH_IOWAIT_THRESHOLD", "20"))       # %

STABILITY_WINDOW = int(os.environ.get("BENCH_STABILITY_WINDOW", "5"))        # leituras consecutivas boas
STABILITY_INTERVAL = float(os.environ.get("BENCH_STABILITY_INTERVAL", "2"))      # segundos entre leituras
MAX_WAIT_STABILITY = float(os.environ.get("BENCH_MAX_WAIT_STABILITY", "600"))    # timeout total (10 min)

# Recuperacao do Ollama/container.
# O script tenta descobrir o container pelo nome/imagem contendo "ollama".
# Se OLLAMA_CONTAINER_ID vier no ambiente, ele e usado como primeira opcao.
ENABLE_OLLAMA_CONTAINER_RECOVERY = True
OLLAMA_CONTAINER_HINT = os.environ.get("OLLAMA_CONTAINER_ID", "").strip()
OLLAMA_CONTAINER_MATCH = os.environ.get("OLLAMA_CONTAINER_MATCH", "ollama").strip().lower()
OLLAMA_SERVICE_HINT = os.environ.get("OLLAMA_SERVICE_NAME", "").strip()
EASYPANEL_DEPLOY_WEBHOOK = os.environ.get("EASYPANEL_DEPLOY_WEBHOOK", "").strip()
EASYPANEL_RESTART_MODE = os.environ.get("EASYPANEL_RESTART_MODE", "swarm").strip().lower()
OLLAMA_HEALTH_TIMEOUT = 420
OLLAMA_RECOVERY_REQUEST_RETRIES = 2
OLLAMA_RECOVERY_COOLDOWN = 30
OLLAMA_PUBLIC_READY_CONSECUTIVE = 2
OLLAMA_INTERNAL_HEALTHCHECK = False
OLLAMA_POST_RESTART_GRACE = 15
VERIFY_OLLAMA_BEFORE_ROUND = True
OLLAMA_READY_BEFORE_ROUND_TIMEOUT = 120
OLLAMA_SERVICE_UPDATE_TIMEOUT = 300

# Mitigacao de swap. Limpar swap exige permissao de root/sudo sem senha em Linux.
# Quando nao for possivel limpar, o script descarrega modelos/reinicia o container
# e aguarda voltar ao baseline antes de aceitar nova rodada.
ENABLE_SWAP_MITIGATION = os.environ.get("BENCH_ENABLE_SWAP_MITIGATION", "1") != "0"
SWAP_MITIGATION_COOLDOWN = float(os.environ.get("BENCH_SWAP_MITIGATION_COOLDOWN", "0"))
SWAP_CONTAMINATION_RERUNS = int(os.environ.get("BENCH_SWAP_CONTAMINATION_RERUNS", "1"))
SWAP_RECOVERY_SLEEP = float(os.environ.get("BENCH_SWAP_RECOVERY_SLEEP", "5"))
CLEAR_SWAP_BEFORE_START = os.environ.get("BENCH_CLEAR_SWAP_BEFORE_START", "1") != "0"
CLEAR_SWAP_BEFORE_EACH_ROUND = (
    os.environ.get("BENCH_CLEAR_SWAP_BEFORE_EACH_ROUND", os.environ.get("BENCH_CLEAR_SWAP_BEFORE_EACH_CALL", "1")) != "0"
)
CLEAR_SWAP_AFTER_MODEL = os.environ.get("BENCH_CLEAR_SWAP_AFTER_MODEL", "1") != "0"
WAIT_BEFORE_EACH_ROUND = os.environ.get("BENCH_WAIT_BEFORE_EACH_ROUND", "1") != "0"
RESTART_CONTAINER_ON_SWAP = os.environ.get("BENCH_RESTART_CONTAINER_ON_SWAP", "1") != "0"

# Pesos do score — explicitamente justificados
# 0.30 semantico: estado/capital corretos, sem punir codigo artificial.
# 0.15 estrutural: JSON estrito importa quando o fluxo automatizado quebra.
# 0.25 paralelismo: mede se o modelo escala no Ollama sem degradação grave.
# 0.15 tps:         velocidade de geração.
# 0.10 latencia:    tempo de resposta, normalizado por min-max.
# 0.05 codigo:      restricao numerica artificial, relevante mas separada.
W_SEM = 0.30
W_STR = 0.15
W_PAR = 0.25
W_TPS = 0.15
W_LAT = 0.10
W_CTL = 0.05

# Valores esperados — usados na validação semântica
EXPECTED = {
    "estado":  ["São Paulo", "Sao Paulo", "sao paulo", "SP"],
    "capital": ["São Paulo", "Sao Paulo", "sao paulo"],
    "ibge_range": (1000, 9999),
}
FORBIDDEN_IBGE_CODES = {1234}

PROMPT_TEMPLATE = """Responda EXCLUSIVAMENTE em JSON valido.

Para o estado brasileiro "Sao Paulo", retorne exatamente um objeto com estas chaves:
- estado
- capital
- codigo_ibge

Regras:
- JSON valido, sem markdown e sem texto fora do JSON.
- estado deve ser "Sao Paulo".
- capital deve ser "Sao Paulo".
- codigo_ibge deve ser um numero inteiro entre 1000 e 9999.
- codigo_ibge deve ser escolhido nesta execucao; nao use 1234 e nao copie o codigo_de_controle.

codigo_de_controle: {control_code}
"""

stop_all = False
SWAP_BASELINE_MB = 0.0
OLLAMA_CONTAINER_ID = OLLAMA_CONTAINER_HINT or ""
LAST_OLLAMA_RECOVERY_TS = 0.0
LAST_SWAP_MITIGATION_TS = 0.0
ollama_recovery_lock = threading.Lock()
swap_mitigation_lock = threading.Lock()
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

def signal_handler(sig, frame):
    global stop_all
    stop_all = True
    console.print("\n[bold red]Interrompido[/bold red]")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_swap_used_mb() -> float:
    return psutil.swap_memory().used / (1024 * 1024)


SWAP_BASELINE_MB = get_swap_used_mb()


def get_swap_delta_mb() -> float:
    return max(0.0, get_swap_used_mb() - SWAP_BASELINE_MB)


def is_swap_contaminated(swap_used_mb: float | None = None, swap_delta_mb: float | None = None) -> bool:
    used = get_swap_used_mb() if swap_used_mb is None else swap_used_mb
    delta = max(0.0, used - SWAP_BASELINE_MB) if swap_delta_mb is None else swap_delta_mb
    return (
        used > SWAP_STABLE_MAX_MB or
        used >= SWAP_HARD_LIMIT_MB or
        delta >= SWAP_DELTA_THRESHOLD_MB
    )


def get_load_average_1m() -> float:
    try:
        if hasattr(os, "getloadavg"):
            return os.getloadavg()[0]
    except (OSError, AttributeError):
        pass

    try:
        return psutil.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0


def reset_swap_baseline(reason: str = ""):
    global SWAP_BASELINE_MB
    SWAP_BASELINE_MB = get_swap_used_mb()
    suffix = f" ({reason})" if reason else ""
    console.print(f"[dim]baseline de swap atualizado: {SWAP_BASELINE_MB:.0f}MB{suffix}[/dim]")


def _run_cmd(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        return proc.returncode, output
    except FileNotFoundError:
        return 127, f"comando nao encontrado: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout executando: {' '.join(args)}"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        clean = item.strip().rstrip("/")
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def ollama_api_url(path: str, base_url: str | None = None) -> str:
    base = (base_url or ACTIVE_BASE_URL).rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def set_active_base_url(base_url: str, reason: str):
    global ACTIVE_BASE_URL
    base_url = base_url.rstrip("/")
    if ACTIVE_BASE_URL != base_url:
        console.print(f"  [green]usando endpoint Ollama {mask_sensitive(base_url)} ({reason})[/green]")
    ACTIVE_BASE_URL = base_url


def endpoint_label(url: str) -> str:
    if url == BASE_URL:
        return "publico"
    if "127.0.0.1" in url or "localhost" in url:
        return "local"
    if re.search(r"http://10\.|http://172\.|http://192\.168\.", url):
        return "docker"
    return "direto"


def find_ollama_container_id(include_stopped: bool = True) -> str | None:
    """
    Localiza o container atual do Ollama pelo ID conhecido, nome ou imagem.
    Retorna o ID curto atual, porque alguns ambientes recriam o container.
    """
    if not ENABLE_OLLAMA_CONTAINER_RECOVERY or not shutil.which("docker"):
        return None

    format_arg = "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    scopes = [["docker", "ps", "--format", format_arg]]
    if include_stopped:
        scopes.append(["docker", "ps", "-a", "--format", format_arg])

    candidates = []
    for cmd in scopes:
        rc, out = _run_cmd(cmd, timeout=30)
        if rc != 0:
            continue
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            cid, name, image, status = parts[:4]
            ports = parts[4] if len(parts) > 4 else ""
            haystack = f"{cid} {name} {image} {ports}".lower()
            if OLLAMA_CONTAINER_ID and (cid.startswith(OLLAMA_CONTAINER_ID) or OLLAMA_CONTAINER_ID.startswith(cid)):
                return cid
            if OLLAMA_CONTAINER_MATCH and OLLAMA_CONTAINER_MATCH in haystack:
                running_rank = 0 if status.lower().startswith("up") else 1
                port_rank = 0 if "11434" in ports else 1
                image_rank = 0 if OLLAMA_CONTAINER_MATCH in image.lower() else 1
                name_rank = 0 if OLLAMA_CONTAINER_MATCH in name.lower() else 1
                candidates.append((running_rank, port_rank, image_rank, name_rank, cid, name, image, status, ports))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][4]


def docker_inspect(container_id: str) -> dict | None:
    if not container_id or not shutil.which("docker"):
        return None
    rc, out = _run_cmd(["docker", "inspect", container_id], timeout=30)
    if rc != 0:
        return None
    try:
        payload = json.loads(out)
        return payload[0] if payload else None
    except Exception:
        return None


def find_ollama_service_name(container_id: str | None = None) -> str | None:
    if OLLAMA_SERVICE_HINT:
        return OLLAMA_SERVICE_HINT
    if not shutil.which("docker"):
        return None

    if container_id:
        info = docker_inspect(container_id)
        labels = (((info or {}).get("Config") or {}).get("Labels") or {})
        service = labels.get("com.docker.swarm.service.name")
        if service:
            return service

    format_arg = "{{.Name}}\t{{.Image}}\t{{.Ports}}"
    rc, out = _run_cmd(["docker", "service", "ls", "--format", format_arg], timeout=30)
    if rc != 0:
        return None

    candidates = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, image = parts[:2]
        ports = parts[2] if len(parts) > 2 else ""
        haystack = f"{name} {image} {ports}".lower()
        if OLLAMA_CONTAINER_MATCH and OLLAMA_CONTAINER_MATCH in haystack:
            port_rank = 0 if "11434" in ports else 1
            image_rank = 0 if OLLAMA_CONTAINER_MATCH in image.lower() else 1
            name_rank = 0 if OLLAMA_CONTAINER_MATCH in name.lower() else 1
            candidates.append((port_rank, image_rank, name_rank, name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def service_container_ids(service_name: str) -> list[str]:
    if not service_name or not shutil.which("docker"):
        return []
    label = f"label=com.docker.swarm.service.name={service_name}"
    rc, out = _run_cmd(["docker", "ps", "-q", "--filter", label], timeout=30)
    if rc != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def wait_service_recreated(service_name: str, previous_ids: list[str]) -> str | None:
    previous = set(previous_ids)
    start = time.time()
    while time.time() - start < OLLAMA_SERVICE_UPDATE_TIMEOUT:
        if stop_all:
            return None
        current = service_container_ids(service_name)
        fresh = [cid for cid in current if cid not in previous]
        if fresh:
            return fresh[0]
        if current and not previous:
            return current[0]
        time.sleep(3)
    return None


def trigger_easypanel_webhook() -> bool:
    if not EASYPANEL_DEPLOY_WEBHOOK:
        return False
    try:
        r = requests.post(EASYPANEL_DEPLOY_WEBHOOK, timeout=30)
        if r.status_code in {200, 201, 202, 204}:
            console.print("  [yellow]redeploy solicitado via webhook do Easypanel[/yellow]")
            return True
        console.print(f"  [yellow]webhook Easypanel respondeu HTTP {r.status_code}[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]webhook Easypanel falhou: {type(e).__name__}[/yellow]")
    return False


def force_update_ollama_service(container_id: str | None, reason: str) -> tuple[bool, str | None]:
    global OLLAMA_CONTAINER_ID
    service = find_ollama_service_name(container_id)
    if not service:
        return False, None

    previous = service_container_ids(service)
    console.print(f"  [yellow]recriando servico Swarm {service} ({reason})...[/yellow]")
    rc, out = _run_cmd(
        ["docker", "service", "update", "--force", "--detach=false", service],
        timeout=OLLAMA_SERVICE_UPDATE_TIMEOUT,
    )
    if rc != 0:
        console.print(f"  [red]falha no service update: {_short(out)}[/red]")
        return False, None

    new_cid = wait_service_recreated(service, previous) or find_ollama_container_id(include_stopped=False)
    if new_cid:
        OLLAMA_CONTAINER_ID = new_cid
        console.print(f"  [green]servico recriado; container atual {new_cid}[/green]")
    else:
        console.print("  [yellow]servico atualizado, mas novo container ainda nao foi localizado[/yellow]")
    return True, new_cid


def docker_ollama_urls(container_id: str | None) -> list[str]:
    if not container_id or not shutil.which("docker"):
        return []

    urls = []

    rc_port, out_port = _run_cmd(["docker", "port", container_id, "11434"], timeout=20)
    if rc_port == 0:
        for line in out_port.splitlines():
            endpoint = line.strip()
            if not endpoint:
                continue
            if endpoint.startswith("0.0.0.0:") or endpoint.startswith(":::"):
                port = endpoint.rsplit(":", 1)[-1]
                urls.append(f"http://127.0.0.1:{port}")
            elif endpoint.startswith("[::]:"):
                port = endpoint.rsplit(":", 1)[-1]
                urls.append(f"http://127.0.0.1:{port}")
            else:
                urls.append(f"http://{endpoint}")

    rc_ip, out_ip = _run_cmd(
        ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", container_id],
        timeout=20,
    )
    if rc_ip == 0:
        for ip in out_ip.split():
            if ip and ip != "<no":
                urls.append(f"http://{ip}:11434")

    return _dedupe(urls)


def ollama_candidate_urls(container_id: str | None = None) -> list[str]:
    candidates = []
    candidates.extend(OLLAMA_DIRECT_URLS)
    candidates.append(ACTIVE_BASE_URL)
    candidates.extend(["http://127.0.0.1:11434", "http://localhost:11434"])
    candidates.extend(docker_ollama_urls(container_id))
    candidates.append(BASE_URL)
    return _dedupe(candidates)


def public_ollama_status(base_url: str | None = None) -> tuple[bool, str]:
    url = base_url or ACTIVE_BASE_URL
    try:
        r = requests.get(ollama_api_url("/api/tags", url), timeout=10)
        if r.status_code == 200:
            return True, "HTTP 200"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "connection"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, type(e).__name__


def container_ollama_status(container_id: str | None) -> tuple[bool | None, str]:
    if not container_id or not OLLAMA_INTERNAL_HEALTHCHECK or not shutil.which("docker"):
        return None, "interno indisponivel"

    probe = (
        "wget -qO- http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || "
        "curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || "
        "python3 - <<'PY'\n"
        "import urllib.request\n"
        "urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5).read()\n"
        "PY"
    )
    rc, out = _run_cmd(["docker", "exec", container_id, "sh", "-lc", probe], timeout=25)
    if rc == 0:
        return True, "HTTP interno OK"

    lower = out.lower()
    if "no such container" in lower or "is not running" in lower:
        return False, out or "container nao esta rodando"

    if "executable file not found" in lower or "not found" in lower or "no such file" in lower:
        return None, out or "ferramentas internas de HTTP nao encontradas"

    return False, out or f"docker exec rc={rc}"


def wait_ollama_ready(timeout_s: int = OLLAMA_HEALTH_TIMEOUT, container_id: str | None = None) -> bool:
    start = time.time()
    ok_counts = {}
    last_log = 0.0
    while time.time() - start < timeout_s:
        if stop_all:
            return False

        candidates = ollama_candidate_urls(container_id)
        statuses = []
        labels_seen = set()
        ready_url = None
        for url in candidates:
            ok, status = public_ollama_status(url)
            label = endpoint_label(url)
            if label not in labels_seen:
                statuses.append(f"{label}={status}")
                labels_seen.add(label)
            if ok:
                ok_counts[url] = ok_counts.get(url, 0) + 1
                if ok_counts[url] >= OLLAMA_PUBLIC_READY_CONSECUTIVE:
                    ready_url = url
                    break
            else:
                ok_counts[url] = 0

        internal_ok, internal_status = container_ollama_status(container_id)

        if ready_url:
            reason = "endpoint direto" if ready_url != BASE_URL else "endpoint publico"
            set_active_base_url(ready_url, reason)
            return True

        now = time.time()
        if now - last_log >= 20:
            elapsed = now - start
            status_line = "; ".join(statuses[:3])
            if internal_ok is True:
                console.print(
                    f"  [yellow]Ollama interno OK; aguardando endpoint externo ({status_line}; {elapsed:.0f}/{timeout_s}s)[/yellow]"
                )
            else:
                console.print(
                    f"  [dim yellow]aguardando Ollama ({status_line}; {elapsed:.0f}/{timeout_s}s)[/dim yellow]"
                )
            last_log = now

        time.sleep(5)
    return False


def restart_ollama_container(reason: str) -> bool:
    global OLLAMA_CONTAINER_ID
    if not ENABLE_OLLAMA_CONTAINER_RECOVERY:
        return False
    if not shutil.which("docker"):
        console.print("  [yellow]docker nao encontrado; nao foi possivel reiniciar o container do Ollama[/yellow]")
        return trigger_easypanel_webhook()

    cid = find_ollama_container_id(include_stopped=True)
    if not cid:
        console.print(
            "  [yellow]container do Ollama nao localizado "
            f"(match='{OLLAMA_CONTAINER_MATCH}'); recuperacao por Docker ignorada[/yellow]"
        )
        return trigger_easypanel_webhook()

    new_cid = None
    restarted = False
    if EASYPANEL_RESTART_MODE in {"webhook", "auto"} and EASYPANEL_DEPLOY_WEBHOOK:
        service = find_ollama_service_name(cid)
        previous = service_container_ids(service) if service else []
        if trigger_easypanel_webhook():
            restarted = True
        if service:
            new_cid = wait_service_recreated(service, previous)

    if not restarted and EASYPANEL_RESTART_MODE in {"swarm", "service", "auto"}:
        restarted, new_cid = force_update_ollama_service(cid, reason)

    if not restarted:
        console.print(f"  [yellow]fallback: docker restart {cid} ({reason})...[/yellow]")
        rc, out = _run_cmd(["docker", "restart", cid], timeout=120)
        if rc != 0:
            console.print(f"  [red]falha ao reiniciar container {cid}: {_short(out)}[/red]")
            return False
        restarted = True

    time.sleep(2)
    new_cid = new_cid or find_ollama_container_id(include_stopped=False) or cid
    OLLAMA_CONTAINER_ID = new_cid
    console.print(f"  [green]container Ollama atual: {new_cid}[/green]")

    if OLLAMA_POST_RESTART_GRACE > 0:
        console.print(f"  [dim]aguardando grace period pos-restart ({OLLAMA_POST_RESTART_GRACE}s)...[/dim]")
        time.sleep(OLLAMA_POST_RESTART_GRACE)

    if wait_ollama_ready(container_id=new_cid):
        console.print(f"  [green]Ollama pronto em {mask_sensitive(ACTIVE_BASE_URL)}[/green]")
        return True

    console.print("  [red]Ollama nao ficou pronto dentro do tempo apos restart[/red]")
    return False


def recover_ollama_server(reason: str) -> bool:
    global LAST_OLLAMA_RECOVERY_TS
    now = time.time()
    with ollama_recovery_lock:
        elapsed = now - LAST_OLLAMA_RECOVERY_TS
        if elapsed < OLLAMA_RECOVERY_COOLDOWN:
            wait_left = OLLAMA_RECOVERY_COOLDOWN - elapsed
            console.print(f"    [dim yellow]recuperacao recente; aguardando {wait_left:.0f}s antes de retestar[/dim yellow]")
            time.sleep(wait_left)
            cid = find_ollama_container_id(include_stopped=False) or OLLAMA_CONTAINER_ID
            return wait_ollama_ready(timeout_s=OLLAMA_HEALTH_TIMEOUT, container_id=cid)

        LAST_OLLAMA_RECOVERY_TS = time.time()
        ok = restart_ollama_container(reason)
        if ok:
            reset_swap_baseline("apos restart do Ollama")
        return ok


def clear_swap_if_possible(force: bool = False) -> bool:
    """
    Tenta esvaziar swap em Linux. Requer root ou sudo -n configurado.
    Retorna True quando nao havia swap relevante ou quando swapoff/swapon funcionou.
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
        console.print(f"  [yellow]limpando swap automaticamente ({before:.0f}MB)...[/yellow]")
        rc_off, out_off = _run_cmd(off_cmd, timeout=180)
        if rc_off != 0:
            continue
        rc_on, out_on = _run_cmd(on_cmd, timeout=180)
        if rc_on == 0:
            after = get_swap_used_mb()
            console.print(f"  [green]swap limpo com swapoff/swapon ({before:.0f}MB -> {after:.0f}MB)[/green]")
            reset_swap_baseline("apos limpar swap")
            return True
        console.print(f"  [yellow]swapoff funcionou, mas swapon falhou: {out_on or out_off}[/yellow]")
        return False

    console.print(
        "  [yellow]nao foi possivel limpar swap automaticamente "
        "(execute como root ou permita sudo -n swapoff/swapon)[/yellow]"
    )
    return False


def mitigate_swap_pressure(reason: str) -> bool:
    global LAST_SWAP_MITIGATION_TS
    if not ENABLE_SWAP_MITIGATION:
        return False

    with swap_mitigation_lock:
        now = time.time()
        elapsed = now - LAST_SWAP_MITIGATION_TS
        if elapsed < SWAP_MITIGATION_COOLDOWN:
            console.print(f"  [dim yellow]mitigacao de swap recente; aguardando {SWAP_MITIGATION_COOLDOWN - elapsed:.0f}s[/dim yellow]")
            time.sleep(SWAP_MITIGATION_COOLDOWN - elapsed)

        LAST_SWAP_MITIGATION_TS = time.time()
        console.print(f"  [yellow]mitigando pressao de swap ({reason})...[/yellow]")

        gc.collect()
        try:
            unload_loaded_benchmark_models("mitigacao de swap")
        except Exception as e:
            console.print(f"  [dim yellow]unload durante mitigacao falhou ({type(e).__name__})[/dim yellow]")

        cleared = clear_swap_if_possible(force=True)
        if cleared:
            time.sleep(SWAP_RECOVERY_SLEEP)
            return True

        restarted = False
        if RESTART_CONTAINER_ON_SWAP:
            restarted = recover_ollama_server("swap detectado")

        if restarted:
            time.sleep(SWAP_RECOVERY_SLEEP)
            reset_swap_baseline("apos mitigacao de swap")
            return True

        console.print(
            "  [yellow]nao foi possivel limpar swap diretamente; "
            "aguardando estabilizacao antes de continuar[/yellow]"
        )
        return False


def prepare_for_round(stage: str) -> bool:
    if CLEAR_SWAP_BEFORE_EACH_ROUND and get_swap_used_mb() >= SWAP_CLEAR_MIN_MB:
        if clear_swap_if_possible(force=True):
            time.sleep(SWAP_RECOVERY_SLEEP)

    if WAIT_BEFORE_EACH_ROUND:
        return wait_system_stable(stage)
    return True


def wait_system_stable(stage: str = "") -> bool:
    label = f": {stage}" if stage else ""
    console.print(f"\n[bold cyan]Aguardando estabilidade REAL do sistema{label}...[/bold cyan]")

    stable_count = 0
    start_time = time.time()

    while True:
        if stop_all:
            return False

        cpu = psutil.cpu_percent(interval=1)
        load = get_load_average_1m()

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        mem_available = vm.available / (1024 * 1024)
        swap_used = swap.used / (1024 * 1024)
        swap_delta = max(0.0, swap_used - SWAP_BASELINE_MB)

        cpu_times = psutil.cpu_times_percent(interval=0.1)
        iowait = getattr(cpu_times, "iowait", 0.0)

        console.print(
            f"[dim]CPU={cpu:.1f}% | load={load:.2f} | "
            f"mem_free={mem_available:.0f}MB | swap={swap_used:.0f}MB "
            f"(delta={swap_delta:.0f}MB, base={SWAP_BASELINE_MB:.0f}MB) | "
            f"iowait={iowait:.1f}%[/dim]"
        )

        swap_abs_ok = swap_used <= SWAP_STABLE_MAX_MB
        swap_ok = (
            swap_abs_ok and
            swap_used < SWAP_HARD_LIMIT_MB and
            swap_delta < SWAP_DELTA_THRESHOLD_MB
        )

        conditions_ok = (
            cpu < CPU_THRESHOLD and
            load < LOAD_THRESHOLD and
            mem_available > MEM_MIN_AVAILABLE_MB and
            swap_ok and
            iowait < IOWAIT_THRESHOLD
        )

        if conditions_ok:
            stable_count += 1
            console.print(f"[green]OK estavel ({stable_count}/{STABILITY_WINDOW})[/green]")
        else:
            stable_count = 0
            reasons = []
            if cpu >= CPU_THRESHOLD:
                reasons.append(f"CPU {cpu:.1f}% >= {CPU_THRESHOLD}%")
            if load >= LOAD_THRESHOLD:
                reasons.append(f"load {load:.2f} >= {LOAD_THRESHOLD}")
            if mem_available <= MEM_MIN_AVAILABLE_MB:
                reasons.append(f"mem_free {mem_available:.0f}MB <= {MEM_MIN_AVAILABLE_MB}MB")
            if not swap_abs_ok:
                reasons.append(f"swap {swap_used:.0f}MB > {SWAP_STABLE_MAX_MB:g}MB")
            if swap_delta >= SWAP_DELTA_THRESHOLD_MB:
                reasons.append(f"swap_delta {swap_delta:.0f}MB >= {SWAP_DELTA_THRESHOLD_MB:g}MB")
            if swap_used >= SWAP_HARD_LIMIT_MB:
                reasons.append(f"swap_total {swap_used:.0f}MB >= {SWAP_HARD_LIMIT_MB:g}MB")
            if iowait >= IOWAIT_THRESHOLD:
                reasons.append(f"iowait {iowait:.1f}% >= {IOWAIT_THRESHOLD}%")
            console.print(
                "[yellow]Sistema ainda instavel... resetando contador[/yellow] "
                f"[dim]({'; '.join(reasons)})[/dim]"
            )
            if not swap_ok:
                mitigate_swap_pressure("swap acima do limite antes da rodada")

        if stable_count >= STABILITY_WINDOW:
            if VERIFY_OLLAMA_BEFORE_ROUND:
                cid = find_ollama_container_id(include_stopped=False) or OLLAMA_CONTAINER_ID
                if not wait_ollama_ready(timeout_s=OLLAMA_READY_BEFORE_ROUND_TIMEOUT, container_id=cid):
                    stable_count = 0
                    console.print(
                        "[yellow]Sistema operacional esta estavel, mas Ollama/Traefik ainda nao; "
                        "continuando espera[/yellow]"
                    )
                    continue
            console.print("[bold green]Sistema estabilizado[/bold green]\n")
            return True

        if time.time() - start_time > MAX_WAIT_STABILITY:
            console.print("[bold red]Timeout aguardando estabilidade — experimento nao sera iniciado em estado contaminado[/bold red]")
            return False

        time.sleep(STABILITY_INTERVAL)


def build_prompt(model: str, n: int, kind: str, idx: int) -> tuple[str, int]:
    """
    Gera um prompt sem exemplo de codigo_ibge.
    O codigo_de_controle varia por chamada para reduzir memorizacao/eco.
    """
    seed = f"{model}|{n}|{kind}|{idx}|{time.time_ns()}"
    control_code = 1000 + (abs(hash(seed)) % 9000)
    if control_code in FORBIDDEN_IBGE_CODES:
        control_code = 9999
    return PROMPT_TEMPLATE.format(control_code=control_code), control_code

# ============================================================================
#  VALIDACAO DUPLA
#  Camada 1 — Semântica: o modelo SABE a resposta?
#  Camada 2 — Estrutural: o modelo obedeceu o FORMATO?
# ============================================================================

def _normalize(s: str) -> str:
    """Remove acentos, lowercase, strip."""
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()

def _extract_json_best_effort(text: str) -> dict | None:
    """
    Tenta extrair JSON de qualquer lugar do texto.
    Usado APENAS na validação semântica — não na estrutural.
    """
    # 1. texto puro
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 2. bloco ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. primeiro objeto JSON no texto
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 4. chaves por regex sem JSON formal
    estado  = re.search(r'"estado"\s*:\s*"([^"]+)"', text)
    capital = re.search(r'"capital"\s*:\s*"([^"]+)"', text)
    ibge    = re.search(r'"codigo_ibge"\s*:\s*(\d+)', text)
    if estado or capital or ibge:
        return {
            "estado":      estado.group(1)  if estado  else None,
            "capital":     capital.group(1) if capital else None,
            "codigo_ibge": int(ibge.group(1)) if ibge else None,
        }
    return None


def validate_semantic(text: str) -> tuple:
    """
    Pergunta: o modelo SABE que Sao Paulo tem capital Sao Paulo?
    Ignora formato e separa a restricao numerica artificial.
    Retorna (ok: bool, detalhes: dict)
    """
    if not text:
        return False, {"reason": "EMPTY"}

    data = _extract_json_best_effort(text)

    if data is None:
        # fallback: busca livre no texto
        estado_ok  = any(_normalize(v) in _normalize(text) for v in EXPECTED["estado"])
        capital_ok = any(_normalize(v) in _normalize(text) for v in EXPECTED["capital"])
        ibge_match = re.search(r'\b([1-9]\d{3})\b', text)
        ibge_value = int(ibge_match.group(1)) if ibge_match else None
        ibge_ok    = ibge_value is not None and ibge_value not in FORBIDDEN_IBGE_CODES
        consistency_ok = estado_ok and capital_ok
        ok = estado_ok and capital_ok and consistency_ok
        return ok, {
            "reason":        "TEXT_FALLBACK",
            "estado_ok":     estado_ok,
            "capital_ok":    capital_ok,
            "consistency_ok": consistency_ok,
            "semantic_ok":   ok,
            "ibge_ok":       ibge_ok,
            "codigo_ok":     ibge_ok,
            "ibge_value":    ibge_value,
        }

    estado_val = data.get("estado")
    capital_val = data.get("capital")
    estado_ok  = estado_val is not None and any(
        _normalize(v) == _normalize(str(estado_val)) for v in EXPECTED["estado"])
    capital_ok = capital_val is not None and any(
        _normalize(v) == _normalize(str(capital_val)) for v in EXPECTED["capital"])
    consistency_ok = (
        estado_val is not None and capital_val is not None and
        _normalize(str(estado_val)) == _normalize(str(capital_val))
    )

    ibge_raw = data.get("codigo_ibge")
    ibge_val = None
    try:
        if isinstance(ibge_raw, bool):
            raise ValueError("bool is not int")
        if isinstance(ibge_raw, int):
            ibge_val = ibge_raw
        elif isinstance(ibge_raw, str) and ibge_raw.strip().isdigit():
            ibge_val = int(ibge_raw.strip())
    except Exception:
        ibge_val = None

    ibge_in_range = (
        ibge_val is not None and
        EXPECTED["ibge_range"][0] <= ibge_val <= EXPECTED["ibge_range"][1]
    )
    ibge_not_forbidden = ibge_val is not None and ibge_val not in FORBIDDEN_IBGE_CODES
    ibge_ok = ibge_in_range and ibge_not_forbidden

    ok = estado_ok and capital_ok and consistency_ok
    return ok, {
        "reason":     "OK" if ok else "WRONG_CONTENT",
        "estado_ok":  estado_ok,
        "capital_ok": capital_ok,
        "consistency_ok": consistency_ok,
        "semantic_ok": ok,
        "ibge_ok":    ibge_ok,
        "codigo_ok":  ibge_ok,
        "ibge_in_range": ibge_in_range,
        "ibge_not_forbidden": ibge_not_forbidden,
        "ibge_value": ibge_val,
        "parsed":     data,
    }


def validate_strict(text: str) -> tuple:
    """
    Pergunta: o modelo obedeceu o formato JSON perfeito?
    Sem extração. Critério rígido.
    Retorna (ok: bool, reason: str)
    """
    if not text:
        return False, "EMPTY"

    stripped = text.strip()

    # detectar tipo de falha estrutural com precisão
    if stripped != text:
        return False, "STRUCT:whitespace_externo"

    has_prefix = not text.startswith("{")
    has_suffix = not text.endswith("}")
    if has_prefix or has_suffix:
        if "```" in text:
            return False, "STRUCT:markdown_code_block"
        if text.startswith("{") and not text.endswith("}"):
            return False, "STRUCT:json_incompleto"
        return False, "STRUCT:texto_fora_do_json"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"STRUCT:json_invalido({e.msg})"

    expected_keys = {"estado", "capital", "codigo_ibge"}
    if set(data.keys()) != expected_keys:
        extra   = set(data.keys()) - expected_keys
        missing = expected_keys - set(data.keys())
        return False, f"STRUCT:schema(extra={extra or '{}'}, missing={missing or '{}'})"

    if not isinstance(data.get("codigo_ibge"), int):
        return False, "STRUCT:tipo_ibge_errado"

    return True, "OK"


def classify_error(text: str, sem_ok: bool, str_ok: bool, http_err: str | None, sem_details: dict | None = None) -> str:
    """
    Classifica o erro em uma categoria interpretável.
    Essencial para diagnóstico real.
    """
    if http_err:
        if "TIMEOUT" in http_err:
            return "INFRA:timeout"
        return f"INFRA:http_error"

    if not text:
        return "INFRA:empty_response"

    if sem_ok and not str_ok:
        if "markdown" in (validate_strict(text)[1] or ""):
            return "FORMAT:markdown_wrapper"
        if "whitespace" in (validate_strict(text)[1] or ""):
            return "FORMAT:whitespace"
        if "texto_fora" in (validate_strict(text)[1] or ""):
            return "FORMAT:texto_antes_ou_depois"
        return "FORMAT:outros"

    if sem_ok and str_ok:
        sem_details = sem_details or {}
        if sem_details.get("ibge_value") in FORBIDDEN_IBGE_CODES:
            return "CONTROL:codigo_fixo_proibido"
        if sem_details.get("ibge_ok") is False:
            return "CONTROL:codigo_ibge_invalido"
        return "OK"

    if not sem_ok:
        sem_details = sem_details or {}
        if sem_details.get("consistency_ok") is False and sem_details.get("estado_ok") and sem_details.get("capital_ok"):
            return "CONTENT:inconsistente"
        if sem_details.get("ibge_value") in FORBIDDEN_IBGE_CODES:
            return "CONTROL:codigo_fixo_proibido"
        if sem_details.get("ibge_ok") is False and (sem_details.get("estado_ok") and sem_details.get("capital_ok")):
            return "CONTROL:codigo_ibge_invalido"
        if "São Paulo" in text or "Sao Paulo" in text or "SP" in text:
            return "CONTENT:parcialmente_correto"
        return "CONTENT:resposta_errada"

    return "UNKNOWN"

# ============================================================================
#  HTTP REQUEST — stream para TTFT
# ============================================================================
def _is_recoverable_ollama_error(err: str | None) -> bool:
    if not err:
        return False
    if err.startswith("TIMEOUT") or "connection" in err or "stream" in err:
        return True
    match = re.search(r"status=(\d+)", err)
    if match:
        return int(match.group(1)) >= 500
    return err.startswith("HTTP_ERROR")


def _do_request(model: str, prompt: str, keep_alive=KEEP_ALIVE_WARM) -> tuple:
    # Usa streaming para medir TTFT: o primeiro token indica quando o modelo começou a responder.
    start = time.perf_counter()
    ttft  = None
    try:
        payload = {"model": model, "prompt": prompt, "stream": True}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        r = requests.post(
            ollama_api_url("/api/generate"),
            json=payload,
            timeout=TIMEOUT, stream=True,
        )
    except requests.exceptions.Timeout:
        return None, "TIMEOUT"
    except requests.exceptions.ConnectionError:
        return None, "HTTP_ERROR:connection"
    except Exception as e:
        return None, f"HTTP_ERROR:{type(e).__name__}"

    if r.status_code != 200:
        return None, f"HTTP_ERROR:status={r.status_code}"

    full, final, got = "", {}, False
    try:
        for raw in r.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            tok = chunk.get("response", "")
            if tok and not got:
                ttft = time.perf_counter() - start
                got  = True
            full += tok
            if chunk.get("done"):
                final = chunk
                break
    except requests.exceptions.Timeout:
        return None, "TIMEOUT:streaming"
    except Exception as e:
        return None, f"HTTP_ERROR:stream:{type(e).__name__}"

    dur = time.perf_counter() - start
    tps = (final["eval_count"] / (final["eval_duration"] / 1e9)
           if final.get("eval_duration", 0) > 0 else 0.0)
    prompt_tps = (final.get("prompt_eval_count", 0) / (final["prompt_eval_duration"] / 1e9)
                  if final.get("prompt_eval_duration", 0) > 0 else 0.0)
    return {
        "dur": dur, "ttft": ttft or dur,
        "response": full, "tps": tps, "prompt_tps": prompt_tps,
        "eval_count": final.get("eval_count", 0),
    }, None


def http_request(model: str, prompt: str, keep_alive=KEEP_ALIVE_WARM) -> tuple:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        data, err = _do_request(model, prompt, keep_alive=keep_alive)
        if data is not None:
            return data, None
        last_err = err
        if attempt < MAX_RETRIES:
            console.print(f"    [dim yellow]retry {attempt+1} ({err})[/dim yellow]")
            time.sleep(1)

    recoverable = _is_recoverable_ollama_error(last_err)
    action = "tentando recuperar Ollama" if recoverable else "erro nao recuperavel por restart"
    console.print(
        f"    [yellow]falhou apos {MAX_RETRIES + 1} tentativa(s) para {model}: "
        f"{last_err}; {action}[/yellow]"
    )
    if recoverable and recover_ollama_server(f"falha persistente no modelo {model}: {last_err}"):
        for attempt in range(OLLAMA_RECOVERY_REQUEST_RETRIES):
            data, err = _do_request(model, prompt, keep_alive=keep_alive)
            if data is not None:
                if attempt:
                    console.print(f"    [green]requisição recuperada para {model}[/green]")
                return data, None
            last_err = err
            console.print(f"    [dim yellow]retry pos-restart {attempt+1} ({err})[/dim yellow]")
            time.sleep(2)

    return None, last_err


def warmup(model: str):
    console.print(f"  [dim]warmup {model} ({WARMUP_REPS}x)...[/dim]")
    for i in range(WARMUP_REPS):
        prompt, _ = build_prompt(model, 0, "warmup", i)
        http_request(model, prompt, keep_alive=KEEP_ALIVE_WARM)


def unload_model(model: str):
    console.print(f"  [dim]cold start: solicitando unload de {model} (keep_alive=0)...[/dim]")
    try:
        requests.post(
            ollama_api_url("/api/generate"),
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=min(TIMEOUT, 60),
        )
    except Exception as e:
        console.print(f"  [dim yellow]unload nao confirmado ({type(e).__name__})[/dim yellow]")
    time.sleep(COLD_UNLOAD_WAIT)


def get_loaded_models() -> list[str]:
    try:
        r = requests.get(ollama_api_url("/api/ps"), timeout=min(TIMEOUT, 30))
        if r.status_code != 200:
            return []
        payload = r.json()
        return [
            item.get("name") or item.get("model")
            for item in payload.get("models", [])
            if item.get("name") or item.get("model")
        ]
    except Exception:
        return []


def unload_loaded_benchmark_models(reason: str):
    """
    Pede ao Ollama para descarregar modelos do benchmark que ja estao carregados.
    Isso reduz contaminacao entre modelos quando OLLAMA_KEEP_ALIVE e alto.
    """
    loaded = [m for m in get_loaded_models() if m in MODELS]
    if not loaded:
        console.print(f"  [dim]nenhum modelo do benchmark carregado ({reason})[/dim]")
        return

    console.print(f"  [dim]liberando modelos carregados ({reason}): {', '.join(loaded)}[/dim]")
    for model in loaded:
        try:
            requests.post(
                ollama_api_url("/api/generate"),
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=min(TIMEOUT, 60),
            )
        except Exception as e:
            console.print(f"  [dim yellow]unload de {model} nao confirmado ({type(e).__name__})[/dim yellow]")
    time.sleep(MODEL_UNLOAD_WAIT)

# ============================================================================
#  RUNNER
# ============================================================================
def _run_concurrent_once(model: str, n: int, kind: str) -> tuple:
    # Cada worker representa uma chamada simultanea ao mesmo modelo.
    # A rodada mede se qualidade e latencia se sustentam quando n aumenta.
    records, lock = [], threading.Lock()
    wall_start    = time.perf_counter()

    def worker(idx: int):
        if stop_all:
            return
        prompt, control_code = build_prompt(model, n, kind, idx)
        keep_alive = KEEP_ALIVE_COLD if kind == "cold" else KEEP_ALIVE_WARM
        data, http_err = http_request(model, prompt, keep_alive=keep_alive)
        swap_used_mb = get_swap_used_mb()
        swap_delta_mb = max(0.0, swap_used_mb - SWAP_BASELINE_MB)
        swap_contaminated = is_swap_contaminated(swap_used_mb, swap_delta_mb)

        if swap_contaminated:
            with lock:
                console.print(
                    "[red]SWAP DETECTADO DURANTE EXECUCAO — RESULTADO PODE ESTAR CONTAMINADO "
                    f"(idx={idx}, swap={swap_used_mb:.0f}MB, delta={swap_delta_mb:.0f}MB)[/red]"
                )

        if data is None:
            with lock:
                records.append({
                    "dur": None, "ttft": None, "tps": 0.0, "prompt_tps": 0.0,
                    "eval_count": 0, "response": "",
                    "semantic_ok": False, "strict_ok": False,
                    "sem_details": {}, "strict_reason": http_err,
                    "error_class": classify_error("", False, False, http_err),
                    "kind": kind, "idx": idx, "control_code": control_code,
                    "swap_used_mb": swap_used_mb,
                    "swap_delta_mb": swap_delta_mb,
                    "swap_baseline_mb": SWAP_BASELINE_MB,
                    "swap_contaminated": swap_contaminated,
                    "rerun_used": False,
                })
            return

        sem_ok, sem_det = validate_semantic(data["response"])
        str_ok, str_rsn = validate_strict(data["response"])
        err_cls          = classify_error(data["response"], sem_ok, str_ok, None, sem_det)

        with lock:
            records.append({
                **data,
                "semantic_ok":    sem_ok,
                "strict_ok":      str_ok,
                "sem_details":    sem_det,
                "strict_reason":  str_rsn,
                "error_class":    err_cls,
                "kind":           kind,
                "idx":            idx,
                "control_code":   control_code,
                "swap_used_mb":   swap_used_mb,
                "swap_delta_mb":  swap_delta_mb,
                "swap_baseline_mb": SWAP_BASELINE_MB,
                "swap_contaminated": swap_contaminated,
                "rerun_used": False,
            })

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads: t.start()
    for t in threads:
        while t.is_alive():
            t.join(timeout=0.2)
            if stop_all:
                return records, 0.0

    return records, time.perf_counter() - wall_start


def run_concurrent(model: str, n: int, kind: str) -> tuple:
    """
    Executa uma rodada e, se swap aparecer, tenta limpar/mitigar e refaz uma vez.
    Assim a rodada salva tende a ser a rodada pos-recuperacao, nao a contaminada.
    """
    last_records, last_wall_time = [], 0.0
    for attempt in range(SWAP_CONTAMINATION_RERUNS + 1):
        records, wall_time = _run_concurrent_once(model, n, kind)
        last_records, last_wall_time = records, wall_time
        swap_dirty = any(r.get("swap_contaminated", False) for r in records)
        if not swap_dirty:
            for r in records:
                r["rerun_used"] = attempt > 0
            return records, wall_time

        if attempt >= SWAP_CONTAMINATION_RERUNS:
            console.print(
                "[red]swap persistiu apos mitigacao; mantendo rodada marcada como contaminada[/red]"
            )
            for r in records:
                r["rerun_used"] = attempt > 0
            return records, wall_time

        console.print(
            "[yellow]rodada contaminada por swap; tentando limpar/recuperar e repetir "
            f"{model} n={n} ({kind})[/yellow]"
        )
        mitigate_swap_pressure(f"swap durante {model} n={n} {kind}")
        if not wait_system_stable(f"antes de repetir {model} n={n} {kind}"):
            for r in last_records:
                r["rerun_used"] = attempt > 0
            return last_records, last_wall_time

    return last_records, last_wall_time

# ============================================================================
#  ESTATÍSTICAS
# ============================================================================
def pct(data, p):
    if not data: return 0.0
    s   = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo  = int(idx); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def compute_stats(records, wall_time):
    n   = len(records)
    ok  = [r for r in records if r["dur"] is not None]
    lats  = [r["dur"]  for r in ok] or [0.0]
    ttfts = [r["ttft"] for r in ok if r["ttft"]] or [0.0]
    tpss  = [r["tps"]  for r in ok] or [0.0]
    ibges = [
        r.get("sem_details", {}).get("ibge_value")
        for r in records
        if r.get("sem_details", {}).get("ibge_value") is not None
    ]
    valid_ibges = [
        r.get("sem_details", {}).get("ibge_value")
        for r in records
        if r.get("sem_details", {}).get("ibge_ok") and
        r.get("sem_details", {}).get("ibge_value") is not None
    ]
    swap_values = [float(r.get("swap_used_mb", 0.0) or 0.0) for r in records]
    swap_deltas = [float(r.get("swap_delta_mb", 0.0) or 0.0) for r in records]
    swap_contaminated = [bool(r.get("swap_contaminated", False)) for r in records]
    reruns = [bool(r.get("rerun_used", False)) for r in records]
    unique_ibges = len(set(valid_ibges))
    ibge_diversity = unique_ibges / len(valid_ibges) if valid_ibges else 0.0

    # contagem por categoria de erro
    err_counts = {}
    for r in records:
        k = r["error_class"]
        err_counts[k] = err_counts.get(k, 0) + 1

    return {
        "n_total":      n,
        "n_http_ok":    len(ok),
        "n_semantic":   sum(r["semantic_ok"] for r in records),
        "n_strict":     sum(r["strict_ok"]   for r in records),
        "n_estado":     sum(bool(r.get("sem_details", {}).get("estado_ok")) for r in records),
        "n_capital":    sum(bool(r.get("sem_details", {}).get("capital_ok")) for r in records),
        "n_codigo":     sum(bool(r.get("sem_details", {}).get("ibge_ok")) for r in records),
        "n_consistency": sum(bool(r.get("sem_details", {}).get("consistency_ok")) for r in records),
        "n_unique_ibge": unique_ibges,
        "semantic_acc": sum(r["semantic_ok"] for r in records) / n if n else 0.0,
        "strict_acc":   sum(r["strict_ok"]   for r in records) / n if n else 0.0,
        "estado_acc":   sum(bool(r.get("sem_details", {}).get("estado_ok")) for r in records) / n if n else 0.0,
        "capital_acc":  sum(bool(r.get("sem_details", {}).get("capital_ok")) for r in records) / n if n else 0.0,
        "codigo_acc":   sum(bool(r.get("sem_details", {}).get("ibge_ok")) for r in records) / n if n else 0.0,
        "consistency_acc": sum(bool(r.get("sem_details", {}).get("consistency_ok")) for r in records) / n if n else 0.0,
        "ibge_diversity": ibge_diversity,
        "forbidden_ibge_rate": sum(1 for v in ibges if v in FORBIDDEN_IBGE_CODES) / len(ibges) if ibges else 0.0,
        "error_rate":   1 - len(ok) / n if n else 1.0,
        "swap_contamination_rate": sum(swap_contaminated) / n if n else 0.0,
        "swap_peak_mb": max(swap_values) if swap_values else 0.0,
        "swap_delta_peak_mb": max(swap_deltas) if swap_deltas else 0.0,
        "rerun_rate": sum(reruns) / n if n else 0.0,
        "err_counts":   err_counts,
        "rps":          len(ok) / wall_time if wall_time else 0.0,
        "rps_attempted": n / wall_time if wall_time else 0.0,
        "wall_time":    wall_time,
        "lat_mean":     statistics.mean(lats),
        "lat_median":   statistics.median(lats),
        "lat_p95":      pct(lats, 95),
        "lat_std":      statistics.stdev(lats) if len(lats) > 1 else 0.0,
        "lat_min":      min(lats),
        "lat_max":      max(lats),
        "ttft_mean":    statistics.mean(ttfts),
        "ttft_p95":     pct(ttfts, 95),
        "tps_mean":     statistics.mean(tpss),
        "tps_max":      max(tpss),
        "tokens_total": sum(r["eval_count"] for r in records),
        "throughput":   sum(r["eval_count"] for r in records) / wall_time if wall_time else 0.0,
    }

# ============================================================================
#  RICH — tabela por rodada
# ============================================================================
def _c(v): return "green" if v >= 0.8 else ("yellow" if v >= 0.5 else "red")

def print_round_table(model, n, rep, kind, s, records):
    sc = _c(s["semantic_acc"]); rc = _c(1 - s["error_rate"])
    title = (f"[bold cyan]{model}[/bold cyan]  n=[yellow]{n}[/yellow]  "
             f"rep=[dim]{rep+1}[/dim]  "
             f"[{'dim green' if kind=='warm' else 'bold magenta'}]{kind}[/{'dim green' if kind=='warm' else 'bold magenta'}]")

    t = Table(title=title, box=box.ROUNDED, header_style="bold magenta", min_width=80)
    t.add_column("Metrica",        width=28, style="bold")
    t.add_column("Valor",          width=14, justify="right")
    t.add_column("Detalhe",        width=36, style="dim")

    # --- dual accuracy ---
    t.add_row(
        f"[{sc}]Semantic accuracy[/{sc}]",
        f"[{sc}]{s['semantic_acc']*100:.0f}%[/{sc}]",
        f"[bold]estado/capital reais[/bold]  {s['n_semantic']}/{s['n_total']} corretos"
    )
    estado_c = _c(s["estado_acc"])
    capital_c = _c(s["capital_acc"])
    codigo_c = _c(s["codigo_acc"])
    t.add_row(
        "  por campo",
        f"[{estado_c}]E {s['estado_acc']*100:.0f}%[/{estado_c}] "
        f"[{capital_c}]C {s['capital_acc']*100:.0f}%[/{capital_c}] "
        f"[{codigo_c}]Cod {s['codigo_acc']*100:.0f}%[/{codigo_c}]",
        f"estado={s['n_estado']}/{s['n_total']} capital={s['n_capital']}/{s['n_total']} codigo={s['n_codigo']}/{s['n_total']}"
    )
    sc2 = _c(s["strict_acc"])
    t.add_row(
        f"[{sc2}]Strict accuracy[/{sc2}]",
        f"[{sc2}]{s['strict_acc']*100:.0f}%[/{sc2}]",
        f"[bold]seguiu o formato?[/bold]  {s['n_strict']}/{s['n_total']} perfeitos"
    )
    # delta revela o gap "sabe mas nao formata"
    delta = s["semantic_acc"] - s["strict_acc"]
    delta_c = "yellow" if delta > 0.1 else "dim"
    t.add_row(
        f"[{delta_c}]  gap (sem - strict)[/{delta_c}]",
        f"[{delta_c}]{delta*100:+.0f}pp[/{delta_c}]",
        "[yellow]alto = modelo sabe, mas nao formata[/yellow]" if delta > 0.1 else "baixo"
    )
    div_c = _c(s["ibge_diversity"])
    t.add_row(
        f"[{div_c}]Diversidade codigo[/{div_c}]",
        f"[{div_c}]{s['ibge_diversity']*100:.0f}%[/{div_c}]",
        f"{s['n_unique_ibge']} codigos unicos validos; fixo proibido={s['forbidden_ibge_rate']*100:.0f}%"
    )

    t.add_row("", "", "")  # separador

    ec = "green" if s["error_rate"] == 0 else ("yellow" if s["error_rate"] < 0.3 else "red")
    t.add_row(
        f"[{ec}]Erros HTTP/infra[/{ec}]",
        f"[{ec}]{s['error_rate']*100:.0f}%[/{ec}]",
        f"{s['n_total'] - s['n_http_ok']} falhas  wall={s['wall_time']:.2f}s"
    )
    swc = "red" if s["swap_contamination_rate"] > 0 else "green"
    t.add_row(
        f"[{swc}]Swap durante execucao[/{swc}]",
        f"[{swc}]{s['swap_contamination_rate']*100:.0f}%[/{swc}]",
        f"pico={s['swap_peak_mb']:.0f}MB  delta_pico={s['swap_delta_peak_mb']:.0f}MB  "
        f"limites={SWAP_STABLE_MAX_MB:g}/{SWAP_DELTA_THRESHOLD_MB:g}MB  rerun={s['rerun_rate']*100:.0f}%"
    )
    t.add_row("RPS real (sucesso/wall)", f"{s['rps']:.3f}",      f"{s['n_http_ok']} sucessos / {s['wall_time']:.2f}s  tentado={s['rps_attempted']:.3f}")
    t.add_row("Latencia media",        f"{s['lat_mean']:.2f}s", f"p95={s['lat_p95']:.2f}s  med={s['lat_median']:.2f}s  sigma={s['lat_std']:.2f}s")
    t.add_row("TTFT medio/p95",        f"{s['ttft_mean']:.2f}s",f"p95={s['ttft_p95']:.2f}s")
    t.add_row("Tokens/s",              f"{s['tps_mean']:.1f}",  f"max={s['tps_max']:.1f}  throughput={s['throughput']:.1f} tok/s")
    console.print(t)

    # --- classificação de erros ---
    if s["err_counts"]:
        et = Table(box=box.SIMPLE, header_style="dim", min_width=80)
        et.add_column("Categoria de erro",  width=32)
        et.add_column("Qtd", width=5, justify="right")
        et.add_column("Interpretacao",      width=40, style="dim")
        labels = {
            "OK":                        ("green",  "resposta perfeita"),
            "FORMAT:markdown_wrapper":   ("yellow", "sabe, mas embrulhou em ```json"),
            "FORMAT:whitespace":         ("yellow", "sabe, mas tem espaco/newline extra"),
            "FORMAT:texto_antes_ou_depois":("yellow","sabe, mas tem texto fora do JSON"),
            "FORMAT:outros":             ("yellow", "sabe, mas formato incorreto"),
            "CONTENT:parcialmente_correto":("red",  "conteudo parcial (SP menciona, valor errado)"),
            "CONTROL:codigo_ibge_invalido":("red", "estado/capital ok, codigo invalido"),
            "CONTROL:codigo_fixo_proibido":("red", "repetiu codigo fixo proibido"),
            "CONTENT:inconsistente":      ("red",   "estado e capital incoerentes"),
            "CONTENT:resposta_errada":   ("red",   "resposta errada de fato"),
            "INFRA:timeout":             ("red",   "timeout — infra/rede"),
            "INFRA:http_error":          ("red",   "falha HTTP"),
            "INFRA:empty_response":      ("red",   "resposta vazia"),
        }
        for cls, cnt in sorted(s["err_counts"].items(), key=lambda x: -x[1]):
            c, interp = labels.get(cls, ("dim", cls))
            et.add_row(f"[{c}]{cls}[/{c}]", str(cnt), interp)
        console.print(et)

    # --- amostra ---
    samp = Table(box=box.SIMPLE, header_style="dim", min_width=80)
    samp.add_column("#",      width=3, style="dim")
    samp.add_column("TTFT",   width=7, justify="right")
    samp.add_column("Lat",    width=7, justify="right")
    samp.add_column("TPS",    width=6, justify="right")
    samp.add_column("Sem", width=5, justify="center")
    samp.add_column("Str", width=5, justify="center")
    samp.add_column("Resposta (primeiros 40 chars)", width=42)
    for r in sorted(records, key=lambda x: x["idx"])[:6]:
        resp = (r["response"] or "")[:40].replace("\n"," ")
        resp += "..." if len(r["response"] or "") > 40 else ""
        ttft = f"{r['ttft']:.2f}s" if r["ttft"] else "—"
        lat  = f"{r['dur']:.2f}s"  if r["dur"]  else "—"
        tps  = f"{r['tps']:.0f}"   if r["tps"]  else "—"
        s_em = "[green]S[/green]" if r["semantic_ok"] else "[red]F[/red]"
        s_tr = "[green]S[/green]" if r["strict_ok"]   else "[red]F[/red]"
        samp.add_row(str(r["idx"]), ttft, lat, tps, s_em, s_tr, resp)
    console.print(samp)

# ============================================================================
#  RICH — ranking final
# ============================================================================
def print_final_ranking(ranking):
    console.rule("[bold yellow]RANKING FINAL[/bold yellow]")
    t = Table(box=box.HEAVY_HEAD, header_style="bold white on dark_blue", min_width=124)
    t.add_column("Pos",         width=4,  justify="center")
    t.add_column("Modelo",      width=18)
    t.add_column("Score",       width=8,  justify="right")
    t.add_column("Cob.",        width=7,  justify="right")
    t.add_column("Sem.Real",    width=9,  justify="right")
    t.add_column("Str.Acc",     width=9,  justify="right")
    t.add_column("Cod.",        width=7,  justify="right")
    t.add_column("Par.",        width=7,  justify="right")
    t.add_column("Eff.",        width=7,  justify="right")
    t.add_column("LatGrow",     width=8,  justify="right")
    t.add_column("Gap",         width=7,  justify="right")
    t.add_column("TPS",         width=8,  justify="right")
    t.add_column("Lat.med",     width=9,  justify="right")
    t.add_column("TTFT",        width=9,  justify="right")
    t.add_column("RPS",         width=8,  justify="right")
    t.add_column("Erros",       width=8,  justify="right")
    t.add_column("Swap",        width=8,  justify="right")

    medals = ["1o","2o","3o"]
    for pos, r in enumerate(ranking.itertuples(), 1):
        medal = medals[pos-1] if pos <= 3 else str(pos)
        sc    = _c(r.semantic_acc)
        str_c = _c(r.strict_acc)
        cod_c = _c(r.codigo_acc)
        par_c = _c(r.parallel_score)
        eff_c = _c(r.parallel_efficiency)
        delta = r.semantic_acc - r.strict_acc
        gap_c = "yellow" if delta > 0.1 else "dim"
        ec    = "green" if r.error_rate < 0.05 else ("yellow" if r.error_rate < 0.3 else "red")
        swc   = "red" if r.swap_contamination_rate > 0 else "green"
        t.add_row(
            medal, f"[bold]{r.model}[/bold]",
            f"[bold cyan]{r.score:.3f}[/bold cyan]",
            f"{r.run_coverage*100:.0f}%",
            f"[{sc}]{r.semantic_acc*100:.0f}%[/{sc}]",
            f"[{str_c}]{r.strict_acc*100:.0f}%[/{str_c}]",
            f"[{cod_c}]{r.codigo_acc*100:.0f}%[/{cod_c}]",
            f"[{par_c}]{r.parallel_score*100:.0f}%[/{par_c}]",
            f"[{eff_c}]{r.parallel_efficiency*100:.0f}%[/{eff_c}]",
            f"{r.latency_growth:.1f}x",
            f"[{gap_c}]{delta*100:+.0f}pp[/{gap_c}]",
            f"{r.tps_mean:.1f}",
            f"{r.lat_median:.2f}s",
            f"{r.ttft_mean:.2f}s",
            f"{r.rps:.3f}",
            f"[{ec}]{r.error_rate*100:.0f}%[/{ec}]",
            f"[{swc}]{r.swap_contamination_rate*100:.0f}%[/{swc}]",
        )
    console.print(t)
    console.print(
        f"\n  [dim]Score = {W_SEM}*sem + {W_STR}*strict + {W_PAR}*parallel + {W_TPS}*tps + {W_LAT}*lat + {W_CTL}*codigo[/dim]"
        f"\n  [dim]Cob. = cobertura da execucao; modelo incompleto tem score ponderado pela cobertura[/dim]"
        f"\n  [dim]Par. = ganho de RPS + retencao de latencia + retencao de qualidade + erros sob concorrencia[/dim]\n"
    )

# ============================================================================
#  PLOTS
# ============================================================================
PALETTE = ["#4C9EE8","#E88C4C","#55C87A","#E85555","#A078CC"]

def make_plots(df_raw, agg, ranking):
    models = ranking["model"].tolist()
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    levels = sorted(agg["requests"].unique())
    labels = [m.split(":")[0] for m in models]

    fig = plt.figure(figsize=(22, 26), facecolor="#0d0d18")
    fig.suptitle("Benchmark LLM — Qualidade por Campo e Paralelismo",
                 fontsize=18, fontweight="bold", color="white", y=0.985)
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.48, wspace=0.36,
                            left=0.06, right=0.97, top=0.96, bottom=0.03)

    def S(ax, title, xl="", yl=""):
        ax.set_facecolor("#161628"); ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xl, color="#aaa", fontsize=9); ax.set_ylabel(yl, color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4a")
        ax.grid(True, color="#1e1e38", linewidth=0.7); return ax

    # 1. Semantic accuracy vs concorrência
    ax = S(fig.add_subplot(gs[0, 0]), "Semantic Accuracy (sabe a resposta?)", "Concorrencia", "%")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.semantic_acc * 100, marker="o", lw=2.5,
                color=colors[m], label=m.split(":")[0])
    ax.axhline(100, color="#55C87A", ls=":", lw=1, alpha=0.4)
    ax.set_ylim(-5, 108); ax.set_xticks(levels)
    ax.legend(fontsize=7, facecolor="#0d0d18", labelcolor="white")

    # 2. Strict accuracy vs concorrência
    ax = S(fig.add_subplot(gs[0, 1]), "Strict Accuracy (seguiu o formato?)", "Concorrencia", "%")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.strict_acc * 100, marker="s", lw=2, ls="--", color=colors[m])
    ax.axhline(100, color="#55C87A", ls=":", lw=1, alpha=0.4)
    ax.set_ylim(-5, 108); ax.set_xticks(levels)

    # 3. Codigo artificial
    ax = S(fig.add_subplot(gs[0, 2]), "Codigo Acc. (restricao artificial)", "Concorrencia", "%")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.codigo_acc * 100, marker="^", lw=2, color=colors[m], label=m.split(":")[0])
    ax.axhline(100, color="#55C87A", ls=":", lw=1, alpha=0.4)
    ax.set_ylim(-5, 108)
    ax.set_xticks(levels); ax.legend(fontsize=6, facecolor="#0d0d18", labelcolor="white")

    # 4. Latência média
    ax = S(fig.add_subplot(gs[1, 0]), "Latencia Media", "Concorrencia", "s")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.lat_mean, marker="o", lw=2, color=colors[m])
    ax.set_xticks(levels)

    # 5. TTFT
    ax = S(fig.add_subplot(gs[1, 1]), "TTFT (Time-to-First-Token)", "Concorrencia", "s")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.ttft_mean, marker="s", lw=2, ls="--", color=colors[m])
    ax.set_xticks(levels)

    # 6. Tokens/s
    ax = S(fig.add_subplot(gs[1, 2]), "Tokens/s (geracao)", "Concorrencia", "tok/s")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.tps_mean, marker="^", lw=2, color=colors[m])
    ax.set_xticks(levels)

    # 7. Boxplot latência
    ax = S(fig.add_subplot(gs[2, 0]), "Distribuicao de Latencia (boxplot)")
    data_b = [df_raw[(df_raw.model==m) & df_raw.dur.notna()]["dur"].tolist() or [0.0] for m in models]
    bp = ax.boxplot(data_b, patch_artist=True,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color="#888"), capprops=dict(color="#888"),
                    flierprops=dict(marker=".", color="#888", ms=4))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.8)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("s", color="#aaa", fontsize=9)

    # 8. RPS
    ax = S(fig.add_subplot(gs[2, 1]), "RPS Real (sucessos / wall-time)", "Concorrencia", "req/s")
    for m in models:
        sub = agg[agg.model == m]
        ax.plot(sub.requests, sub.rps, marker="o", lw=2, color=colors[m])
    ax.set_xticks(levels)

    # 9. Score final
    ax = S(fig.add_subplot(gs[2, 2]),
           f"Score Final\n({W_SEM}*sem + {W_STR}*str + {W_PAR}*par + {W_TPS}*tps + {W_LAT}*lat + {W_CTL}*cod)")
    rnk = ranking.sort_values("score")
    bars = ax.barh([m.split(":")[0] for m in rnk.model], rnk.score,
                   color=[colors[m] for m in rnk.model], alpha=0.85, height=0.55)
    for bar, val in zip(bars, rnk.score):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, rnk.score.max() * 1.22)

    # 10. Heatmap semantic accuracy
    ax = S(fig.add_subplot(gs[3, 0]), "Heatmap Semantic Accuracy (%)")
    pv = agg.pivot_table(index="model", columns="requests", values="semantic_acc") * 100
    pv.index = [m.split(":")[0] for m in pv.index]
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, color="#aaa", fontsize=8)
    ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index,   color="#aaa", fontsize=8)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="black" if v > 55 else "white", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaa")

    # 11. Heatmap strict accuracy
    ax = S(fig.add_subplot(gs[3, 1]), "Heatmap Strict Accuracy (%)")
    pv2 = agg.pivot_table(index="model", columns="requests", values="strict_acc") * 100
    pv2.index = [m.split(":")[0] for m in pv2.index]
    im2 = ax.imshow(pv2.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pv2.columns))); ax.set_xticklabels(pv2.columns, color="#aaa", fontsize=8)
    ax.set_yticks(range(len(pv2.index)));   ax.set_yticklabels(pv2.index,   color="#aaa", fontsize=8)
    for i in range(len(pv2.index)):
        for j in range(len(pv2.columns)):
            v = pv2.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="black" if v > 55 else "white", fontsize=9, fontweight="bold")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaa")

    # 12. Heatmap TTFT
    ax = S(fig.add_subplot(gs[3, 2]), "Heatmap TTFT (s)")
    pv3 = agg.pivot_table(index="model", columns="requests", values="ttft_mean")
    pv3.index = [m.split(":")[0] for m in pv3.index]
    im3 = ax.imshow(pv3.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pv3.columns))); ax.set_xticklabels(pv3.columns, color="#aaa", fontsize=8)
    ax.set_yticks(range(len(pv3.index)));   ax.set_yticklabels(pv3.index,   color="#aaa", fontsize=8)
    for i in range(len(pv3.index)):
        for j in range(len(pv3.columns)):
            v = pv3.values[i, j]
            ax.text(j, i, f"{v:.1f}s", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaa")

    # 13. Radar — perfil normalizado
    ax = fig.add_subplot(gs[4, :], polar=True)
    # vamos usar apenas metade da largura
    ax = fig.add_subplot(gs[4, 1], polar=True)
    ax.set_facecolor("#161628")
    ax.set_title("Perfil Normalizado", color="white", fontsize=10, fontweight="bold", pad=14)
    ax.tick_params(colors="#aaa", labelsize=7); ax.spines["polar"].set_color("#2a2a4a")

    cats   = ["Sem", "Strict", "Codigo", "Parallel", "TPS", "1/Lat", "RPS"]
    N      = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    ref = dict(
        sem  = agg.semantic_acc.max(),
        str_ = max(agg.strict_acc.max(), 0.01),
        cod  = max(agg.codigo_acc.max(), 0.01),
        par  = max(agg.parallel_score.max(), 0.01),
        tps  = agg.tps_mean.max(),
        lat  = agg.lat_mean.min(),
        rps  = agg.rps.max(),
    )
    for m in models:
        sub  = agg[agg.model == m]
        vals = [
            min(sub.semantic_acc.mean() / (ref["sem"]  + 1e-9), 1.0),
            min(sub.strict_acc.mean()   / (ref["str_"] + 1e-9), 1.0),
            min(sub.codigo_acc.mean() / (ref["cod"] + 1e-9), 1.0),
            min(sub.parallel_score.mean() / (ref["par"] + 1e-9), 1.0),
            min(sub.tps_mean.mean()     / (ref["tps"]  + 1e-9), 1.0),
            min(ref["lat"] / (sub.lat_mean.mean()   + 1e-9), 1.0),
            min(sub.rps.mean()          / (ref["rps"]  + 1e-9), 1.0),
        ]
        vals = [max(v, 0) for v in vals] + [vals[0]]
        ax.plot(angles, vals, lw=1.8, color=colors[m], label=m.split(":")[0])
        ax.fill(angles, vals, color=colors[m], alpha=0.10)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=8, color="#ccc")
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["","","",""], size=6)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.45, 1.15),
              facecolor="#0d0d18", labelcolor="white")

    # 14. Stacked bar de error classes
    ax = S(fig.add_subplot(gs[4, 0]), "Distribuicao de Erros por Modelo")
    err_cats  = ["OK","FORMAT:markdown_wrapper","FORMAT:whitespace",
                 "FORMAT:texto_antes_ou_depois","FORMAT:outros",
                 "CONTROL:codigo_ibge_invalido","CONTROL:codigo_fixo_proibido",
                 "CONTENT:inconsistente",
                 "CONTENT:parcialmente_correto","CONTENT:resposta_errada",
                 "INFRA:timeout","INFRA:http_error","INFRA:empty_response","UNKNOWN"]
    err_colors_map = {
        "OK": "#55C87A",
        "FORMAT:markdown_wrapper":      "#a0d060",
        "FORMAT:whitespace":            "#c8e060",
        "FORMAT:texto_antes_ou_depois": "#e8d040",
        "FORMAT:outros":                "#f0a030",
        "CONTROL:codigo_ibge_invalido": "#f08030",
        "CONTROL:codigo_fixo_proibido": "#d86030",
        "CONTENT:inconsistente":        "#d05050",
        "CONTENT:parcialmente_correto": "#e87030",
        "CONTENT:resposta_errada":      "#e85555",
        "INFRA:timeout":                "#aa2222",
        "INFRA:http_error":             "#882222",
        "INFRA:empty_response":         "#661111",
        "UNKNOWN":                      "#555555",
    }
    bottoms = np.zeros(len(models))
    for cat in err_cats:
        vals = []
        for m in models:
            sub   = df_raw[df_raw.model == m]
            n_tot = len(sub)
            cnt   = (sub["error_class"] == cat).sum()
            vals.append(cnt / n_tot * 100 if n_tot else 0)
        if sum(vals) > 0:
            ax.bar(range(len(models)), vals, bottom=bottoms,
                   color=err_colors_map.get(cat, "#555"), label=cat.split(":")[-1], alpha=0.85)
            bottoms += np.array(vals)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("%", color="#aaa", fontsize=9)
    ax.legend(fontsize=6, loc="upper right", facecolor="#0d0d18", labelcolor="white")

    # 15. Cold vs Warm latencia
    ax = S(fig.add_subplot(gs[4, 2]), "Cold vs Warm — Latencia Media", "", "s")
    w = 0.35; x = np.arange(len(models))
    cold_lats = [df_raw[(df_raw.model==m) & (df_raw.kind=="cold") & df_raw.dur.notna()]["dur"].mean() for m in models]
    warm_lats = [df_raw[(df_raw.model==m) & (df_raw.kind=="warm") & df_raw.dur.notna()]["dur"].mean() for m in models]
    bars_c = ax.bar(x-w/2, [v if not (isinstance(v,float) and np.isnan(v)) else 0 for v in cold_lats],
                    w, label="cold", color="#4C9EE8", alpha=0.85)
    bars_w = ax.bar(x+w/2, [v if not (isinstance(v,float) and np.isnan(v)) else 0 for v in warm_lats],
                    w, label="warm", color="#E88C4C", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.legend(fontsize=8, facecolor="#0d0d18", labelcolor="white")
    for bar in list(bars_c)+list(bars_w):
        h = bar.get_height()
        if h > 0.1:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f"{h:.1f}",
                    ha="center", va="bottom", color="white", fontsize=7)

    path = os.path.join(OUT_DIR, "benchmark_full.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d0d18")
    plt.close(fig)
    return path

# ============================================================================
#  HTML
# ============================================================================
def make_html(df_raw, agg, ranking):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rank_rows():
        out = ""
        for i, r in enumerate(ranking.itertuples(), 1):
            sc  = "good" if r.semantic_acc >= 0.8 else ("mid" if r.semantic_acc >= 0.5 else "bad")
            stc = "good" if r.strict_acc   >= 0.8 else ("mid" if r.strict_acc   >= 0.5 else "bad")
            codc = "good" if r.codigo_acc >= 0.8 else ("mid" if r.codigo_acc >= 0.5 else "bad")
            parc = "good" if r.parallel_score >= 0.8 else ("mid" if r.parallel_score >= 0.5 else "bad")
            effc = "good" if r.parallel_efficiency >= 0.8 else ("mid" if r.parallel_efficiency >= 0.5 else "bad")
            delta = r.semantic_acc - r.strict_acc
            gap_c = "mid" if delta > 0.1 else ""
            ec    = "good" if r.error_rate < 0.05 else ("mid" if r.error_rate < 0.3 else "bad")
            swc   = "bad" if r.swap_contamination_rate > 0 else "good"
            covc  = "good" if r.complete_run else "bad"
            out += (f"<tr><td class='c'>{i}</td><td><b>{r.model}</b></td>"
                    f"<td class='c score'>{r.score:.3f}</td>"
                    f"<td class='c {covc}'>{r.run_coverage*100:.0f}%</td>"
                    f"<td class='c {sc}'>{r.semantic_acc*100:.0f}%</td>"
                    f"<td class='c {stc}'>{r.strict_acc*100:.0f}%</td>"
                    f"<td class='c {codc}'>{r.codigo_acc*100:.0f}%</td>"
                    f"<td class='c {parc}'>{r.parallel_score*100:.0f}%</td>"
                    f"<td class='c {effc}'>{r.parallel_efficiency*100:.0f}%</td>"
                    f"<td class='c'>{r.latency_growth:.1f}x</td>"
                    f"<td class='c {gap_c}'>{delta*100:+.0f}pp</td>"
                    f"<td class='c'>{r.tps_mean:.1f}</td>"
                    f"<td class='c'>{r.lat_median:.2f}s</td>"
                    f"<td class='c'>{r.ttft_mean:.2f}s</td>"
                    f"<td class='c'>{r.rps:.3f}</td>"
                    f"<td class='c {ec}'>{r.error_rate*100:.0f}%</td>"
                    f"<td class='c {swc}'>{r.swap_contamination_rate*100:.0f}%</td></tr>")
        return out

    def detail_rows():
        out = ""
        for r in agg.sort_values(["model","requests"]).itertuples():
            sc  = "good" if r.semantic_acc >= 0.8 else ("mid" if r.semantic_acc >= 0.5 else "bad")
            stc = "good" if r.strict_acc   >= 0.8 else ("mid" if r.strict_acc   >= 0.5 else "bad")
            codc = "good" if r.codigo_acc >= 0.8 else ("mid" if r.codigo_acc >= 0.5 else "bad")
            parc = "good" if r.parallel_score >= 0.8 else ("mid" if r.parallel_score >= 0.5 else "bad")
            swc = "bad" if r.swap_contamination_rate > 0 else "good"
            out += (f"<tr><td>{r.model}</td><td class='c'>{r.requests}</td>"
                    f"<td class='c {sc}'>{r.semantic_acc*100:.0f}%</td>"
                    f"<td class='c {stc}'>{r.strict_acc*100:.0f}%</td>"
                    f"<td class='c {codc}'>{r.codigo_acc*100:.0f}%</td>"
                    f"<td class='c {parc}'>{r.parallel_score*100:.0f}%</td>"
                    f"<td class='c'>{r.parallel_efficiency*100:.0f}%</td>"
                    f"<td class='c'>{r.latency_growth:.1f}x</td>"
                    f"<td class='c'>{r.lat_mean:.2f}s</td>"
                    f"<td class='c'>{r.lat_p95:.2f}s</td>"
                    f"<td class='c'>{r.ttft_mean:.2f}s</td>"
                    f"<td class='c'>{r.tps_mean:.1f}</td>"
                    f"<td class='c'>{r.rps:.3f}</td>"
                    f"<td class='c'>{r.error_rate*100:.0f}%</td>"
                    f"<td class='c {swc}'>{r.swap_contamination_rate*100:.0f}%</td></tr>")
        return out

    def err_breakdown():
        cats = ["OK","FORMAT:markdown_wrapper","FORMAT:whitespace",
                "FORMAT:texto_antes_ou_depois","FORMAT:outros",
                "CONTROL:codigo_ibge_invalido","CONTROL:codigo_fixo_proibido",
                "CONTENT:inconsistente",
                "CONTENT:parcialmente_correto","CONTENT:resposta_errada",
                "INFRA:timeout","INFRA:http_error"]
        header = "<tr><th>Modelo</th>" + "".join(f"<th>{c.split(':')[-1]}</th>" for c in cats) + "</tr>"
        rows   = ""
        for m in ranking.model:
            sub   = df_raw[df_raw.model == m]
            n_tot = len(sub)
            row   = f"<tr><td>{m}</td>"
            for cat in cats:
                cnt = (sub["error_class"] == cat).sum()
                pct_v = cnt / n_tot * 100 if n_tot else 0
                cls = "good" if cat == "OK" and pct_v > 80 else (
                      "bad"  if cat != "OK" and pct_v > 20 else "")
                row += f"<td class='c {cls}'>{cnt} ({pct_v:.0f}%)</td>"
            rows += row + "</tr>"
        return header + rows

    total_sem  = df_raw["semantic_ok"].mean() * 100
    total_str  = df_raw["strict_ok"].mean()   * 100
    total_err  = (1 - df_raw["dur"].notna().mean()) * 100
    total_swap = df_raw["swap_contaminated"].mean() * 100
    total_ttft = df_raw["ttft"].dropna().mean()
    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark LLM — {now}</title>
<style>
:root{{--bg:#0d0d18;--card:#161628;--border:#252540;--text:#dde;--sub:#8899bb;
      --acc:#4C9EE8;--good:#55c87a;--mid:#e8a030;--bad:#e85555;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:2rem}}
h1{{font-size:1.8rem;color:var(--acc);margin-bottom:.3rem}}
.sub{{color:var(--sub);font-size:.85rem;margin-bottom:1.8rem}}
h2{{color:var(--text);font-size:1.1rem;margin:2rem 0 .8rem;border-left:3px solid var(--acc);padding-left:.7rem}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.4rem;margin-bottom:1.4rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.8rem;margin-bottom:1.5rem}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.9rem;text-align:center}}
.kpi .val{{font-size:1.5rem;font-weight:bold;color:var(--acc)}}
.kpi .lbl{{font-size:.72rem;color:var(--sub);margin-top:.2rem}}
.kpi .sub2{{font-size:.7rem;color:var(--sub);margin-top:.1rem}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:#0f0f22;color:var(--acc);padding:.5rem .6rem;text-align:left;font-weight:600;border-bottom:2px solid var(--border)}}
td{{padding:.4rem .6rem;border-bottom:1px solid var(--border)}}
tr:hover td{{background:#1e1e38}}
.c{{text-align:center}}.good{{color:var(--good);font-weight:bold}}
.mid{{color:var(--mid);font-weight:bold}}.bad{{color:var(--bad);font-weight:bold}}
.score{{color:#a0d8ff;font-weight:bold}}
img{{max-width:100%;border-radius:8px;margin-top:.8rem}}
.callout{{background:#1a1a08;border:1px solid #444420;border-radius:8px;padding:1rem;margin-bottom:1rem;font-size:.85rem;color:#ddc}}
.note{{color:var(--sub);font-size:.75rem;margin-top:.6rem}}
footer{{color:var(--sub);font-size:.75rem;margin-top:2.5rem;text-align:center}}
</style></head><body>
<h1>Benchmark LLM — Qualidade por Campo e Paralelismo</h1>
<p class="sub">Gerado em {now} · {mask_sensitive(ACTIVE_BASE_URL)}</p>

<div class="callout">
  <b>Como interpretar:</b> este benchmark separa duas perguntas distintas.<br>
  <b>Semantic accuracy</b>: o modelo <em>sabe</em> que a capital de SP e SP? (conteudo correto, formato ignorado)<br>
  <b>Strict accuracy</b>: o modelo <em>obedeceu</em> o formato JSON exato? (sem nenhum texto fora)<br>
  <b>Codigo</b>: o modelo cumpriu a restricao numerica artificial?<br>
  <b>Parallel score</b>: o modelo aumentou vazao sob concorrencia sem explodir latencia, erro ou queda de qualidade?<br>
  <b>Gap (sem - strict)</b>: valores altos indicam que o modelo sabe a resposta mas tem dificuldade com formatacao rigida.<br>
  <b>Swap</b>: qualquer valor acima de 0% indica rodada potencialmente contaminada por pressao de memoria.
</div>

<div class="grid">
  <div class="kpi"><div class="val">{len(MODELS)}</div><div class="lbl">Modelos</div></div>
  <div class="kpi"><div class="val">{len(df_raw)}</div><div class="lbl">Requisicoes</div></div>
  <div class="kpi"><div class="val">{total_sem:.0f}%</div><div class="lbl">Semantic Acc</div><div class="sub2">sabe a resposta</div></div>
  <div class="kpi"><div class="val">{total_str:.0f}%</div><div class="lbl">Strict Acc</div><div class="sub2">formato perfeito</div></div>
  <div class="kpi"><div class="val">{total_err:.0f}%</div><div class="lbl">Erros HTTP</div></div>
  <div class="kpi"><div class="val">{total_swap:.0f}%</div><div class="lbl">Swap</div><div class="sub2">rodadas contaminadas</div></div>
  <div class="kpi"><div class="val">{df_raw['tps'].mean():.1f}</div><div class="lbl">TPS medio</div></div>
  <div class="kpi"><div class="val">{total_ttft:.2f}s</div><div class="lbl">TTFT medio</div></div>
</div>

<h2>Ranking Final</h2>
<div class="card">
<table>
<tr><th>#</th><th>Modelo</th><th>Score</th><th>Cob.</th><th>Sem.Real</th><th>Str.Acc</th><th>Cod.</th>
    <th>Par.</th><th>Eff.</th><th>LatGrow</th><th>Gap</th><th>TPS</th><th>Lat.med</th><th>TTFT</th><th>RPS</th><th>Erros</th><th>Swap</th></tr>
{rank_rows()}
</table>
<p class="note">Score = {W_SEM}*sem + {W_STR}*strict + {W_PAR}*parallel + {W_TPS}*tps + {W_LAT}*lat + {W_CTL}*codigo
&nbsp;|&nbsp; Cob. = cobertura da execucao; score final e ponderado por cobertura.
&nbsp;|&nbsp; Par. combina eficiencia de RPS, latencia, qualidade e erro sob concorrencia.</p>
</div>

<h2>Breakdown de Erros por Categoria</h2>
<div class="card">
<table>{err_breakdown()}</table>
<p class="note">FORMAT = sabe a resposta, erro so de formatacao.
CONTROL = falha em restricao artificial. CONTENT = resposta errada de fato. INFRA = falha de rede/timeout.</p>
</div>

<h2>Metricas por Modelo x Concorrencia</h2>
<div class="card">
<table>
<tr><th>Modelo</th><th>Conc.</th><th>Sem.Real</th><th>Str.Acc</th><th>Cod.</th><th>Par.</th><th>Eff.</th><th>LatGrow</th>
    <th>Lat.med</th><th>Lat.p95</th><th>TTFT</th><th>TPS</th><th>RPS</th><th>Erros</th><th>Swap</th></tr>
{detail_rows()}
</table>
</div>

<h2>Graficos</h2>
<div class="card"><img src="benchmark_full.png" alt="Graficos"></div>
<footer>Benchmark LLM · {now}</footer>
</body></html>"""

    path = os.path.join(OUT_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def _sample_bucket(error_class: str) -> str:
    if error_class == "OK":
        return "ok"
    if error_class.startswith("FORMAT:"):
        return "format_error"
    if error_class.startswith("CONTROL:"):
        return "control_error"
    if error_class.startswith("CONTENT:"):
        return "content_error"
    if error_class.startswith("INFRA:"):
        return "infra_error"
    return "unknown_error"


def save_samples(df_raw: pd.DataFrame):
    """
    Salva outputs completos por modelo e exemplos por categoria de erro.
    Isso preserva evidencias qualitativas para auditoria/paper.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fields = [
        "model", "requests", "rep", "kind", "idx", "control_code",
        "dur", "ttft", "tps", "semantic_ok", "strict_ok",
        "estado_ok", "capital_ok", "consistency_ok", "codigo_ok", "ibge_ok", "ibge_value",
        "error_class", "strict_reason", "response",
    ]

    for model, sub in df_raw.groupby("model", sort=False):
        model_dir = os.path.join(SAMPLES_DIR, _safe_name(model))
        os.makedirs(model_dir, exist_ok=True)

        all_path = os.path.join(model_dir, "all_outputs.jsonl")
        with open(all_path, "w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                payload = {k: row.get(k) for k in fields}
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        counts = {}
        for _, row in sub.iterrows():
            bucket = _sample_bucket(str(row.get("error_class", "UNKNOWN")))
            counts[bucket] = counts.get(bucket, 0)
            if counts[bucket] >= SAMPLES_PER_CATEGORY:
                continue
            counts[bucket] += 1
            suffix = "json" if bucket == "ok" else "txt"
            sample_path = os.path.join(model_dir, f"{bucket}_{counts[bucket]:02d}.{suffix}")
            payload = {k: row.get(k) for k in fields}
            if suffix == "json":
                with open(sample_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            else:
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"model: {payload['model']}\n"
                        f"requests: {payload['requests']}\n"
                        f"rep: {payload['rep']}\n"
                        f"kind: {payload['kind']}\n"
                        f"idx: {payload['idx']}\n"
                        f"error_class: {payload['error_class']}\n"
                        f"strict_reason: {payload['strict_reason']}\n"
                        f"semantic_ok: {payload['semantic_ok']}\n"
                        f"strict_ok: {payload['strict_ok']}\n"
                        f"ibge_value: {payload['ibge_value']}\n\n"
                        f"{payload['response'] or ''}\n"
                    )


def write_recommendation(ranking: pd.DataFrame):
    eligible = ranking[ranking["complete_run"] == True]
    if eligible.empty:
        eligible = ranking
    best = eligible.iloc[0]
    best_strict = ranking.sort_values(
        ["complete_run", "strict_acc", "semantic_acc", "error_rate", "score"],
        ascending=[False, False, False, True, False],
    ).iloc[0]
    best_parallel = eligible.sort_values(
        ["parallel_score", "parallel_efficiency", "semantic_acc", "error_rate"],
        ascending=[False, False, False, True],
    ).iloc[0]
    best_speed = eligible.sort_values(
        ["tps_mean", "semantic_acc", "strict_acc"],
        ascending=[False, False, False],
    ).iloc[0]
    best_latency = eligible.sort_values(
        ["lat_median", "semantic_acc", "strict_acc"],
        ascending=[True, False, False],
    ).iloc[0]

    recommendation = {
        "best_overall": best.to_dict(),
        "best_strict_json": best_strict.to_dict(),
        "best_parallel": best_parallel.to_dict(),
        "best_generation_speed": best_speed.to_dict(),
        "best_latency": best_latency.to_dict(),
    }

    json_path = os.path.join(OUT_DIR, "recommendation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, ensure_ascii=False, indent=2, default=str)

    md_path = os.path.join(OUT_DIR, "recommendation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Recomendacao de modelo\n\n")
        f.write(
            f"Melhor modelo geral: **{best.model}** "
            f"(score={best.score:.3f}, semantic={best.semantic_acc*100:.0f}%, "
            f"strict={best.strict_acc*100:.0f}%, codigo={best.codigo_acc*100:.0f}%, "
            f"parallel={best.parallel_score*100:.0f}%, "
            f"cobertura={best.run_coverage*100:.0f}%, "
            f"lat.med={best.lat_median:.2f}s, tps={best.tps_mean:.1f}, "
            f"erros={best.error_rate*100:.0f}%, swap={best.swap_contamination_rate*100:.0f}%).\n\n"
        )
        f.write(
            f"Melhor para JSON estrito: **{best_strict.model}** "
            f"(strict={best_strict.strict_acc*100:.0f}%, semantic={best_strict.semantic_acc*100:.0f}%).\n\n"
        )
        f.write(
            f"Melhor sob paralelismo: **{best_parallel.model}** "
            f"(parallel={best_parallel.parallel_score*100:.0f}%, "
            f"eficiencia={best_parallel.parallel_efficiency*100:.0f}%, "
            f"lat_growth={best_parallel.latency_growth:.1f}x, "
            f"rps_speedup={best_parallel.rps_speedup:.1f}x).\n\n"
        )
        f.write(
            f"Mais rapido em geracao: **{best_speed.model}** "
            f"(tps={best_speed.tps_mean:.1f}).\n\n"
        )
        f.write(
            f"Menor latencia mediana: **{best_latency.model}** "
            f"(lat.med={best_latency.lat_median:.2f}s).\n\n"
        )
        f.write(
            "Regra pratica: use o melhor geral para producao equilibrada; "
            "use o melhor JSON estrito se o seu fluxo quebra com formato invalido; "
            "use o melhor sob paralelismo quando agentes/n8n chamam o mesmo modelo em paralelo; "
            "use o mais rapido apenas quando throughput pesa mais que confiabilidade. "
            "Modelos com cobertura incompleta nao disputam recomendacoes principais.\n"
        )

    console.print(Panel(
        f"[bold green]Melhor geral:[/bold green] {best.model}  score={best.score:.3f}\n"
        f"[bold]JSON estrito:[/bold] {best_strict.model}  strict={best_strict.strict_acc*100:.0f}%\n"
        f"[bold]Paralelismo:[/bold] {best_parallel.model}  par={best_parallel.parallel_score*100:.0f}%  eff={best_parallel.parallel_efficiency*100:.0f}%\n"
        f"[bold]Mais rapido:[/bold] {best_speed.model}  tps={best_speed.tps_mean:.1f}\n"
        f"[bold]Menor latencia:[/bold] {best_latency.model}  mediana={best_latency.lat_median:.2f}s\n"
        f"[bold]Swap:[/bold] {best.model}  contaminacao={best.swap_contamination_rate*100:.0f}%  pico={best.swap_peak_mb:.0f}MB",
        title="Qual modelo usar?", border_style="green"
    ))

# ============================================================================
#  MAIN
# ============================================================================
raw_records = []

console.print(Panel(
    f"[bold cyan]Benchmark LLM — Qualidade por Campo e Paralelismo[/bold cyan]\n"
    f"URL publica: [yellow]{mask_sensitive(BASE_URL)}[/yellow]\n"
    f"URLs diretas env: [yellow]{', '.join(mask_sensitive(u) for u in OLLAMA_DIRECT_URLS) or 'auto'}[/yellow]\n"
    f"Modelos: [green]{', '.join(MODELS)}[/green]\n"
    f"Niveis: {REQUEST_LEVELS}  Warmup: {WARMUP_REPS}x  Retry: {MAX_RETRIES}x  Timeout: {TIMEOUT}s\n"
    f"Cold start: {COLD_START}  keep_alive_cold={KEEP_ALIVE_COLD}  unload_after_model={UNLOAD_AFTER_MODEL}\n\n"
    f"Recuperacao Ollama: {ENABLE_OLLAMA_CONTAINER_RECOVERY}  modo={EASYPANEL_RESTART_MODE}  "
    f"service={mask_sensitive(OLLAMA_SERVICE_HINT) or 'auto'}  container_match={OLLAMA_CONTAINER_MATCH}  "
    f"rerun_swap={SWAP_CONTAMINATION_RERUNS}x\n"
    f"Swap: clear_start={CLEAR_SWAP_BEFORE_START} clear_round={CLEAR_SWAP_BEFORE_EACH_ROUND} "
    f"clear_after_model={CLEAR_SWAP_AFTER_MODEL} max={SWAP_STABLE_MAX_MB:g}MB "
    f"delta={SWAP_DELTA_THRESHOLD_MB:g}MB min_clear={SWAP_CLEAR_MIN_MB:g}MB\n\n"
    f"[bold]Score = {W_SEM}*sem + {W_STR}*strict + {W_PAR}*parallel + {W_TPS}*tps + {W_LAT}*lat + {W_CTL}*codigo[/bold]\n"
    f"[dim]Semantica real, controle numerico e paralelismo sao metricas separadas[/dim]",
    title="Config", border_style="blue"
))

if CLEAR_SWAP_BEFORE_START:
    had_swap_to_clear = get_swap_used_mb() >= SWAP_CLEAR_MIN_MB
    if clear_swap_if_possible(force=had_swap_to_clear) and had_swap_to_clear:
        time.sleep(SWAP_RECOVERY_SLEEP)
    reset_swap_baseline("inicio do benchmark")

for model in MODELS:
    if stop_all:
        break
    console.rule(f"[bold cyan]{model}[/bold cyan]")
    last_global = False
    if UNLOAD_BEFORE_MODEL:
        unload_loaded_benchmark_models(f"antes de testar {model}")
        if not prepare_for_round(f"antes de testar {model}"):
            stop_all = True
            break

    if WARMUP_REPS and not COLD_START:
        warmup(model)
        if not prepare_for_round(f"apos warmup inicial de {model}"):
            stop_all = True
            break

    for n in REQUEST_LEVELS:
        if stop_all:
            break
        for rep in range(REPEATS[n]):
            if stop_all: break
            kind = "cold" if rep == 0 else "warm"
            console.print(f"  [dim]concorrencia={n}  rep={rep+1}/{REPEATS[n]}  ({kind})[/dim]")
            if kind == "cold" and COLD_START:
                unload_model(model)
                if not prepare_for_round(f"{model} concorrencia={n} rep={rep+1} cold"):
                    stop_all = True
                    break
            elif kind == "warm" and WARMUP_REPS:
                warmup(model)
                if not prepare_for_round(f"{model} concorrencia={n} rep={rep+1} warm"):
                    stop_all = True
                    break
            elif not prepare_for_round(f"{model} concorrencia={n} rep={rep+1} {kind}"):
                stop_all = True
                break

            records, wall_time = run_concurrent(model, n, kind)
            stats = compute_stats(records, wall_time)
            print_round_table(model, n, rep, kind, stats, records)

            for r in records:
                raw_records.append({
                    "model":       model,
                    "requests":    n,
                    "rep":         rep,
                    "kind":        kind,
                    "idx":         r["idx"],
                    "control_code": r.get("control_code"),
                    "wall_time":   wall_time,
                    "dur":         r["dur"],
                    "ttft":        r["ttft"],
                    "tps":         r["tps"],
                    "prompt_tps":  r["prompt_tps"],
                    "eval_count":  r["eval_count"],
                    "semantic_ok": r["semantic_ok"],
                    "strict_ok":   r["strict_ok"],
                    "estado_ok":   r.get("sem_details", {}).get("estado_ok", False),
                    "capital_ok":  r.get("sem_details", {}).get("capital_ok", False),
                    "consistency_ok": r.get("sem_details", {}).get("consistency_ok", False),
                    "codigo_ok":   r.get("sem_details", {}).get("ibge_ok", False),
                    "ibge_ok":     r.get("sem_details", {}).get("ibge_ok", False),
                    "ibge_value":  r.get("sem_details", {}).get("ibge_value"),
                    "error_class": r["error_class"],
                    "strict_reason": r.get("strict_reason",""),
                    "swap_used_mb": r.get("swap_used_mb", 0.0),
                    "swap_delta_mb": r.get("swap_delta_mb", 0.0),
                    "swap_baseline_mb": r.get("swap_baseline_mb", SWAP_BASELINE_MB),
                    "swap_contaminated": r.get("swap_contaminated", False),
                    "rerun_used": r.get("rerun_used", False),
                    "response":    r["response"],
                })

            if stop_all: break
            last_rep_for_model = (n == REQUEST_LEVELS[-1] and rep == REPEATS[n]-1)
            last_global = (model == MODELS[-1] and last_rep_for_model)
            if not last_rep_for_model:
                console.print(f"  [dim]cooldown inteligente (substitui {COOLDOWN}s fixos)...[/dim]")
                if not wait_system_stable(f"entre rodadas de {model}"):
                    stop_all = True
                    break

    if UNLOAD_AFTER_MODEL:
        unload_model(model)
    if CLEAR_SWAP_AFTER_MODEL:
        had_swap_to_clear = get_swap_used_mb() >= SWAP_CLEAR_MIN_MB
        if clear_swap_if_possible(force=had_swap_to_clear) and had_swap_to_clear:
            time.sleep(SWAP_RECOVERY_SLEEP)
        reset_swap_baseline("apos modelo")
    if not stop_all and not last_global:
        if not wait_system_stable(f"apos modelo {model}"):
            stop_all = True
            break

# ============================================================================
#  CONSOLIDACAO
# ============================================================================
# A consolidacao transforma respostas individuais em metricas por modelo/nivel,
# calcula score final, salva CSVs, amostras e relatórios visuais.
console.rule("[bold yellow]Consolidando...[/bold yellow]")

if not raw_records:
    console.print("[bold red]Nenhum resultado coletado: sistema nao atingiu estabilidade suficiente.[/bold red]")
    sys.exit(1)

df_raw = pd.DataFrame(raw_records)
df_raw.to_csv(os.path.join(OUT_DIR, "results_raw.csv"), index=False)
save_samples(df_raw)

# RPS real: apenas requisicoes com resposta HTTP/stream bem sucedida
rps_agg = (
    df_raw.groupby(["model","requests","rep"])
    .agg(n_success=("dur", lambda x: x.notna().sum()), wall_time=("wall_time","first"))
    .assign(rps=lambda x: x.n_success / x.wall_time)
    .groupby(["model","requests"])["rps"].mean()
    .reset_index()
)

rps_attempted_agg = (
    df_raw.groupby(["model","requests","rep"])
    .agg(n_reqs=("semantic_ok","count"), wall_time=("wall_time","first"))
    .assign(rps_attempted=lambda x: x.n_reqs / x.wall_time)
    .groupby(["model","requests"])["rps_attempted"].mean()
    .reset_index()
)

ok_df = df_raw[df_raw["dur"].notna()]

perf_cols = [
    "model", "requests", "lat_mean", "lat_median", "lat_p95", "lat_std",
    "lat_min", "lat_max", "ttft_mean", "ttft_p95", "tps_mean", "tps_max",
    "tokens_total",
]
if ok_df.empty:
    perf = pd.DataFrame(columns=perf_cols)
else:
    perf = ok_df.groupby(["model","requests"]).agg(
        lat_mean   = ("dur",  "mean"),
        lat_median = ("dur",  "median"),
        lat_p95    = ("dur",  lambda x: pct(list(x), 95)),
        lat_std    = ("dur",  lambda x: statistics.stdev(list(x)) if len(x) > 1 else 0.0),
        lat_min    = ("dur",  "min"),
        lat_max    = ("dur",  "max"),
        ttft_mean  = ("ttft", "mean"),
        ttft_p95   = ("ttft", lambda x: pct(list(x), 95)),
        tps_mean   = ("tps",  "mean"),
        tps_max    = ("tps",  "max"),
        tokens_total = ("eval_count", "sum"),
    ).reset_index()

# dual accuracy calculada sobre TODOS os registros (incluindo erros HTTP)
dual = df_raw.groupby(["model","requests"]).agg(
    semantic_acc = ("semantic_ok", "mean"),
    strict_acc   = ("strict_ok",   "mean"),
    estado_acc   = ("estado_ok",   "mean"),
    capital_acc  = ("capital_ok",  "mean"),
    consistency_acc = ("consistency_ok", "mean"),
    codigo_acc   = ("codigo_ok", "mean"),
    ibge_valid_acc = ("ibge_ok", "mean"),
    error_rate   = ("dur", lambda x: 1 - x.notna().mean()),
    swap_contamination_rate = ("swap_contaminated", "mean"),
    swap_peak_mb = ("swap_used_mb", "max"),
    swap_delta_peak_mb = ("swap_delta_mb", "max"),
    rerun_rate = ("rerun_used", "mean"),
).reset_index()

diversity = df_raw.groupby(["model","requests"]).agg(
    ibge_diversity = ("ibge_value", lambda x: x.dropna().nunique() / len(x.dropna()) if len(x.dropna()) else 0.0),
    forbidden_ibge_rate = ("ibge_value", lambda x: x.dropna().isin(FORBIDDEN_IBGE_CODES).mean() if len(x.dropna()) else 0.0),
).reset_index()

wall_agg = (
    df_raw.groupby(["model","requests","rep"])
    .agg(wall_time=("wall_time", "first"))
    .groupby(["model","requests"])["wall_time"].sum()
    .reset_index(name="wall_time_total")
)

agg = (
    dual.merge(perf, on=["model","requests"], how="left")
       .merge(diversity, on=["model","requests"])
       .merge(rps_agg, on=["model","requests"])
       .merge(rps_attempted_agg, on=["model","requests"])
       .merge(wall_agg, on=["model","requests"])
)
failed_perf_defaults = {
    "lat_mean": TIMEOUT, "lat_median": TIMEOUT, "lat_p95": TIMEOUT,
    "lat_std": 0.0, "lat_min": TIMEOUT, "lat_max": TIMEOUT,
    "ttft_mean": TIMEOUT, "ttft_p95": TIMEOUT,
    "tps_mean": 0.0, "tps_max": 0.0, "tokens_total": 0,
}
agg = agg.fillna(failed_perf_defaults)
agg["throughput"] = agg["tokens_total"] / (agg["wall_time_total"] + 1e-9)


def add_parallel_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Mede se o modelo ganha vazao quando a concorrencia aumenta.
    A referencia e o menor nivel de concorrencia testado por modelo.
    """
    out = frame.copy()
    defaults = {
        "rps_speedup": 1.0,
        "parallel_efficiency": 1.0,
        "latency_growth": 1.0,
        "latency_retention": 1.0,
        "semantic_drop": 0.0,
        "strict_drop": 0.0,
        "error_growth": 0.0,
        "quality_retention": 1.0,
        "parallel_score": 1.0,
    }
    for col, value in defaults.items():
        out[col] = value

    for model, sub in out.groupby("model", sort=False):
        sub = sub.sort_values("requests")
        base = sub.iloc[0]
        base_n = max(float(base["requests"]), 1.0)
        base_rps = max(float(base["rps"]), 1e-9)
        base_lat = max(float(base["lat_mean"]), 1e-9)
        base_sem = float(base["semantic_acc"])
        base_str = float(base["strict_acc"])
        base_err = float(base["error_rate"])

        for idx, row in sub.iterrows():
            n = max(float(row["requests"]), 1.0)
            rps = max(float(row["rps"]), 0.0)
            lat = max(float(row["lat_mean"]), 1e-9)
            sem = float(row["semantic_acc"])
            strict = float(row["strict_acc"])
            err = float(row["error_rate"])

            expected_rps = base_rps * (n / base_n)
            rps_speedup = rps / base_rps
            parallel_efficiency = min(rps / max(expected_rps, 1e-9), 1.0)
            latency_growth = lat / base_lat
            latency_retention = min(base_lat / lat, 1.0)
            semantic_drop = max(0.0, base_sem - sem)
            strict_drop = max(0.0, base_str - strict)
            error_growth = max(0.0, err - base_err)
            semantic_retention = sem / base_sem if base_sem > 1e-9 else 1.0
            strict_retention = strict / base_str if base_str > 1e-9 else 1.0
            quality_retention = max(0.0, min(semantic_retention, strict_retention, 1.0))
            error_retention = max(0.0, 1.0 - error_growth)
            parallel_score = (
                0.45 * parallel_efficiency +
                0.25 * latency_retention +
                0.20 * quality_retention +
                0.10 * error_retention
            )

            out.loc[idx, "rps_speedup"] = rps_speedup
            out.loc[idx, "parallel_efficiency"] = parallel_efficiency
            out.loc[idx, "latency_growth"] = latency_growth
            out.loc[idx, "latency_retention"] = latency_retention
            out.loc[idx, "semantic_drop"] = semantic_drop
            out.loc[idx, "strict_drop"] = strict_drop
            out.loc[idx, "error_growth"] = error_growth
            out.loc[idx, "quality_retention"] = quality_retention
            out.loc[idx, "parallel_score"] = max(0.0, min(parallel_score, 1.0))

    return out


agg = add_parallel_metrics(agg)
agg.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

# score equilibrado com normalizacao min-max
def _norm_high(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if abs(hi - lo) < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (series - lo) / (hi - lo)


def _norm_low(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if abs(hi - lo) < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (hi - series) / (hi - lo)


agg["tps_norm"]  = _norm_high(agg["tps_mean"])
agg["lat_norm"]  = _norm_low(agg["lat_mean"])
agg["score_raw"] = (
    W_SEM * agg["semantic_acc"] +
    W_STR * agg["strict_acc"]   +
    W_PAR * agg["parallel_score"] +
    W_TPS * agg["tps_norm"]     +
    W_LAT * agg["lat_norm"]     +
    W_CTL * agg["codigo_acc"]
)
agg["score"] = agg["score_raw"]
agg.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

run_coverage = (
    df_raw.groupby("model").agg(
        observed_records=("model", "size"),
        tested_levels=("requests", "nunique"),
    ).reset_index()
)
run_coverage["expected_records"] = EXPECTED_RECORDS_PER_MODEL
run_coverage["expected_levels"] = EXPECTED_LEVELS_PER_MODEL
run_coverage["run_coverage"] = (
    run_coverage["observed_records"] / run_coverage["expected_records"]
).clip(upper=1.0)
run_coverage["level_coverage"] = (
    run_coverage["tested_levels"] / run_coverage["expected_levels"]
).clip(upper=1.0)
run_coverage["complete_run"] = (
    (run_coverage["observed_records"] >= run_coverage["expected_records"]) &
    (run_coverage["tested_levels"] >= run_coverage["expected_levels"])
)

ranking = (
    agg.groupby("model").agg(
        score_raw    = ("score_raw",    "mean"),
        semantic_acc = ("semantic_acc", "mean"),
        strict_acc   = ("strict_acc",   "mean"),
        estado_acc   = ("estado_acc",   "mean"),
        capital_acc  = ("capital_acc",  "mean"),
        codigo_acc   = ("codigo_acc",   "mean"),
        consistency_acc = ("consistency_acc", "mean"),
        ibge_valid_acc = ("ibge_valid_acc", "mean"),
        ibge_diversity = ("ibge_diversity", "mean"),
        forbidden_ibge_rate = ("forbidden_ibge_rate", "mean"),
        parallel_score = ("parallel_score", "mean"),
        parallel_efficiency = ("parallel_efficiency", "mean"),
        rps_speedup = ("rps_speedup", "max"),
        latency_growth = ("latency_growth", "max"),
        semantic_drop = ("semantic_drop", "max"),
        strict_drop = ("strict_drop", "max"),
        error_growth = ("error_growth", "max"),
        tps_mean     = ("tps_mean",     "mean"),
        lat_median   = ("lat_median",   "mean"),
        lat_p95      = ("lat_p95",      "mean"),
        ttft_mean    = ("ttft_mean",    "mean"),
        rps          = ("rps",          "mean"),
        rps_attempted = ("rps_attempted", "mean"),
        error_rate   = ("error_rate",   "mean"),
        swap_contamination_rate = ("swap_contamination_rate", "mean"),
        swap_peak_mb = ("swap_peak_mb", "max"),
        swap_delta_peak_mb = ("swap_delta_peak_mb", "max"),
        rerun_rate = ("rerun_rate", "mean"),
    ).reset_index()
)
ranking = ranking.merge(run_coverage, on="model", how="left")
ranking["run_coverage"] = ranking["run_coverage"].fillna(0.0)
ranking["level_coverage"] = ranking["level_coverage"].fillna(0.0)
ranking["complete_run"] = ranking["complete_run"].fillna(False)
ranking["score"] = ranking["score_raw"] * ranking["run_coverage"]
ranking = ranking.sort_values(["complete_run", "score"], ascending=[False, False])
ranking.to_csv(os.path.join(OUT_DIR, "ranking.csv"), index=False)

print_final_ranking(ranking)
write_recommendation(ranking)
console.print("[dim]Gerando graficos...[/dim]")
make_plots(df_raw, agg, ranking)
console.print("[dim]Gerando HTML...[/dim]")
make_html(df_raw, agg, ranking)

console.print(Panel(
    "[green]ok[/green] results_raw.csv\n"
    "[green]ok[/green] summary.csv\n"
    "[green]ok[/green] ranking.csv\n"
    "[green]ok[/green] recommendation.md / recommendation.json\n"
    "[green]ok[/green] samples/  (outputs completos e exemplos de erro)\n"
    "[green]ok[/green] benchmark_full.png  (15 graficos)\n"
    "[green]ok[/green] report.html",
    title=f"Saida em {OUT_DIR}/", border_style="green"
))

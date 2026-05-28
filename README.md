# Benchmarks Ollama

Este projeto contem tres scripts para comparar modelos servidos por Ollama. As configuracoes sensiveis ficam no arquivo local `.env`, que e ignorado pelo Git.

## Arquivos

- `paralelismo.py`: benchmark principal de qualidade, formato JSON, latencia, TTFT, tokens/s, RPS, paralelismo, erros e swap.
- `paralelismo2.py`: benchmark complementar para chatbot, RAG e automacao, com testes de raciocinio, instrucao, JSON, memoria multi-turn, recusa por falta de contexto e controle de alucinacao.
- `benchmark_threads.py`: benchmark de CPU para escolher o melhor `options.num_thread` por modelo.

## Configuracao

Crie ou edite o arquivo `.env` na raiz do projeto:

```env
OLLAMA_BASE_URL=https://seu-servidor-ollama
OLLAMA_DIRECT_URL=
OLLAMA_CONTAINER_ID=
OLLAMA_CONTAINER_MATCH=ollama
OLLAMA_SERVICE_NAME=
EASYPANEL_DEPLOY_WEBHOOK=
EASYPANEL_RESTART_MODE=swarm
```

O `.env` nao deve ser enviado ao Git. Os scripts carregam esse arquivo automaticamente e mascaram o endpoint nos logs e relatorios.

## Como executar

Execute a partir da raiz do projeto, com Python 3.10 ou superior.

```powershell
python .\paralelismo.py
```

Gera `benchmark_results/` com dados brutos, resumo, ranking, recomendacao, amostras e relatorio HTML.

```powershell
python .\paralelismo2.py
```

Gera `benchmark_chatbot_results/` com ranking para chatbot/RAG, resumos por categoria, recomendacao e relatorios.

```powershell
python .\benchmark_threads.py
```

Gera `benchmark_thread_results/` com a recomendacao de `num_thread` por modelo.

## Variaveis uteis

Voce pode sobrescrever parametros pelo `.env` ou pelo ambiente antes de executar:

```env
BENCH_MODELS=gemma2:9b,qwen3:8b
BENCH_REPEATS=3
THREAD_MODELS=qwen3:8b,gemma2:9b,llama3.2:3b,granite3.3:8b
THREAD_LEVELS=6,8,10,12,14,16,18,20,22,24
THREAD_REPEATS=2
THREAD_CLEAR_SWAP_BEFORE_START=1
THREAD_CLEAR_SWAP_BEFORE_EACH_CALL=1
THREAD_CLEAR_SWAP_AFTER_MODEL=1
THREAD_SWAP_CONTAMINATION_RERUNS=1
```

Para a limpeza automatica de swap em Linux, execute o script como root ou permita `sudo -n swapoff -a` e `sudo -n swapon -a` para o usuario que roda o benchmark.

## Git

O `.gitignore` ignora tudo por padrao e libera apenas:

- `.gitignore`
- `README.md`
- `paralelismo.py`
- `paralelismo2.py`
- `benchmark_threads.py`

Para preparar o envio:

```powershell
git init
git add .gitignore README.md paralelismo.py paralelismo2.py benchmark_threads.py
git status
```

As pastas de resultados, caches, `.env` e demais arquivos locais ficam fora do commit.

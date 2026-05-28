# Benchmarks locais de SLMs com Ollama

Versao em portugues abaixo. Version en espanol al final del documento.

Este projeto reune tres benchmarks para avaliar Small Language Models (SLMs)
executados localmente com Ollama, especialmente em cenarios sem GPU dedicada e
com foco em uso administrativo. A ideia nao e medir apenas "qual modelo acerta
mais", mas observar se o modelo tambem entrega respostas em formato correto,
mantem estabilidade sob carga, evita alucinacoes e roda bem em CPU.

As configuracoes sensiveis ficam no arquivo local `.env`, que nao deve ser
enviado ao GitHub.

## Objetivo

O projeto foi criado para apoiar a escolha de modelos locais em ambientes com
restricoes de infraestrutura, como orgaos publicos, universidades, prefeituras
ou laboratorios que precisam manter maior controle sobre dados e processamento.

Os benchmarks ajudam a responder perguntas praticas:

- O modelo segue JSON estrito quando isso e necessario para automacao?
- A qualidade cai quando varias requisicoes rodam ao mesmo tempo?
- O modelo funciona bem como chatbot, assistente RAG ou apoio administrativo?
- Ele recusa responder quando a informacao nao esta no contexto?
- Qual configuracao de `num_thread` melhora a inferencia em CPU?

## Arquivos principais

- `paralelismo.py`: avalia automacao estruturada. Mede acerto semantico,
  aderencia a JSON estrito, codigo de controle, latencia, TTFT, tokens/s, RPS,
  paralelismo, erros e uso de swap.
- `paralelismo2.py`: avalia tarefas de chatbot, RAG e automacao. Inclui testes
  de raciocinio, seguimento de instrucao, extracao em JSON, memoria multi-turn,
  resposta baseada em contexto, recusa por falta de informacao e controle de
  alucinacao.
- `benchmark_threads.py`: avalia desempenho em CPU variando
  `options.num_thread`, para encontrar a melhor configuracao por modelo.

## Estrutura dos resultados

Cada script gera uma pasta propria de resultados:

- `benchmark_results/`: saidas do benchmark de paralelismo e JSON.
- `benchmark_chatbot_results/`: saidas do benchmark de chatbot, RAG e automacao.
- `benchmark_thread_results/`: saidas do benchmark de threads em CPU.

Essas pastas incluem CSVs, rankings, recomendacoes em Markdown, relatorios HTML
e figuras. Elas ficam fora do Git porque podem ser grandes, conter informacoes
do ambiente de execucao e variar a cada rodada.

## Configuracao

Crie ou edite um arquivo `.env` na raiz do projeto:

```env
OLLAMA_BASE_URL=https://seu-servidor-ollama
OLLAMA_DIRECT_URL=
OLLAMA_CONTAINER_ID=
OLLAMA_CONTAINER_MATCH=ollama
OLLAMA_SERVICE_NAME=
EASYPANEL_DEPLOY_WEBHOOK=
EASYPANEL_RESTART_MODE=swarm
```

Os scripts carregam o `.env` automaticamente. O endpoint e mascarado nos logs e
relatorios para evitar exposicao acidental.

## Como executar

Execute os comandos a partir da raiz do projeto, com Python 3.10 ou superior.

```powershell
python .\paralelismo.py
```

Gera o benchmark de automacao estruturada em `benchmark_results/`.

```powershell
python .\paralelismo2.py
```

Gera o benchmark complementar de chatbot/RAG em `benchmark_chatbot_results/`.

```powershell
python .\benchmark_threads.py
```

Gera o benchmark de `num_thread` em CPU em `benchmark_thread_results/`.

## Variaveis uteis

Parametros podem ser sobrescritos pelo `.env` ou pelo ambiente:

```env
BENCH_MODELS=gemma4:e2b,gemma2:9b,qwen3:8b
BENCH_REPEATS=3
BENCH_TIMEOUT=420

THREAD_MODELS=gemma2:9b,llama3.1:8b,qwen2.5:7b
THREAD_LEVELS=6,8,10,12,14,16,18,20,22,24
THREAD_REPEATS=5
THREAD_TIMEOUT=900
THREAD_CLEAR_SWAP_BEFORE_START=1
THREAD_CLEAR_SWAP_BEFORE_EACH_CALL=1
THREAD_CLEAR_SWAP_AFTER_MODEL=1
```

Para limpeza automatica de swap em Linux, rode com permissao adequada ou permita
`sudo -n swapoff -a` e `sudo -n swapon -a` para o usuario que executa o
benchmark.

## Como interpretar

Os rankings devem ser lidos por caso de uso. Um modelo pode ser bom em JSON
estrito e ruim em latencia, ou rapido em CPU e fraco em raciocinio. Por isso, os
relatorios sempre combinam qualidade, formato, erros, cobertura, latencia,
throughput e estabilidade.

No benchmark de threads, a comparacao principal e dentro de cada modelo. Antes
de aceitar uma recomendacao de `num_thread`, verifique se `error_rate` e
`response_valid_rate` indicam execucoes validas. Configuracoes com respostas
vazias ou invalidas nao devem ser escolhidas apenas pelo score agregado.

## Git

O repositorio versiona somente:

- `.gitignore`
- `README.md`
- `benchmark_threads.py`
- `paralelismo.py`
- `paralelismo2.py`

Arquivos sensiveis, caches, resultados dos benchmarks e materiais auxiliares
ficam locais e sao ignorados pelo `.gitignore`.

---

# Benchmarks locales de SLMs con Ollama

Este proyecto reune tres benchmarks para evaluar Small Language Models (SLMs)
ejecutados localmente con Ollama, especialmente en escenarios sin GPU dedicada y
con foco en usos administrativos. La idea no es medir solamente "que modelo
acierta mas", sino observar si el modelo tambien entrega respuestas en el
formato correcto, mantiene estabilidad bajo carga, evita alucinaciones y se
ejecuta bien en CPU.

Las configuraciones sensibles quedan en el archivo local `.env`, que no debe
subirse a GitHub.

## Objetivo

El proyecto fue creado para apoyar la seleccion de modelos locales en entornos
con restricciones de infraestructura, como organismos publicos, universidades,
municipios o laboratorios que necesitan mantener mayor control sobre los datos y
el procesamiento.

Los benchmarks ayudan a responder preguntas practicas:

- El modelo sigue JSON estricto cuando es necesario para automatizacion?
- La calidad cae cuando varias solicitudes se ejecutan al mismo tiempo?
- El modelo funciona bien como chatbot, asistente RAG o apoyo administrativo?
- Rechaza responder cuando la informacion no esta en el contexto?
- Que configuracion de `num_thread` mejora la inferencia en CPU?

## Archivos principales

- `paralelismo.py`: evalua automatizacion estructurada. Mide acierto semantico,
  adherencia a JSON estricto, codigo de control, latencia, TTFT, tokens/s, RPS,
  paralelismo, errores y uso de swap.
- `paralelismo2.py`: evalua tareas de chatbot, RAG y automatizacion. Incluye
  pruebas de razonamiento, seguimiento de instrucciones, extraccion en JSON,
  memoria multi-turn, respuesta basada en contexto, rechazo por falta de
  informacion y control de alucinaciones.
- `benchmark_threads.py`: evalua el rendimiento en CPU variando
  `options.num_thread`, para encontrar la mejor configuracion por modelo.

## Estructura de resultados

Cada script genera su propia carpeta de resultados:

- `benchmark_results/`: salidas del benchmark de paralelismo y JSON.
- `benchmark_chatbot_results/`: salidas del benchmark de chatbot, RAG y
  automatizacion.
- `benchmark_thread_results/`: salidas del benchmark de threads en CPU.

Estas carpetas incluyen CSVs, rankings, recomendaciones en Markdown, reportes
HTML y figuras. Permanecen fuera de Git porque pueden ser grandes, contener
informacion del entorno de ejecucion y variar en cada corrida.

## Configuracion

Cree o edite un archivo `.env` en la raiz del proyecto:

```env
OLLAMA_BASE_URL=https://su-servidor-ollama
OLLAMA_DIRECT_URL=
OLLAMA_CONTAINER_ID=
OLLAMA_CONTAINER_MATCH=ollama
OLLAMA_SERVICE_NAME=
EASYPANEL_DEPLOY_WEBHOOK=
EASYPANEL_RESTART_MODE=swarm
```

Los scripts cargan el `.env` automaticamente. El endpoint se enmascara en logs y
reportes para evitar exposicion accidental.

## Como ejecutar

Ejecute los comandos desde la raiz del proyecto, con Python 3.10 o superior.

```powershell
python .\paralelismo.py
```

Genera el benchmark de automatizacion estructurada en `benchmark_results/`.

```powershell
python .\paralelismo2.py
```

Genera el benchmark complementario de chatbot/RAG en
`benchmark_chatbot_results/`.

```powershell
python .\benchmark_threads.py
```

Genera el benchmark de `num_thread` en CPU en `benchmark_thread_results/`.

## Variables utiles

Los parametros pueden sobrescribirse por `.env` o por variables de entorno:

```env
BENCH_MODELS=gemma4:e2b,gemma2:9b,qwen3:8b
BENCH_REPEATS=3
BENCH_TIMEOUT=420

THREAD_MODELS=gemma2:9b,llama3.1:8b,qwen2.5:7b
THREAD_LEVELS=6,8,10,12,14,16,18,20,22,24
THREAD_REPEATS=5
THREAD_TIMEOUT=900
THREAD_CLEAR_SWAP_BEFORE_START=1
THREAD_CLEAR_SWAP_BEFORE_EACH_CALL=1
THREAD_CLEAR_SWAP_AFTER_MODEL=1
```

Para limpieza automatica de swap en Linux, ejecute con permisos adecuados o
permita `sudo -n swapoff -a` y `sudo -n swapon -a` para el usuario que ejecuta
el benchmark.

## Como interpretar

Los rankings deben leerse segun el caso de uso. Un modelo puede ser bueno en
JSON estricto y malo en latencia, o rapido en CPU y debil en razonamiento. Por
eso, los reportes combinan calidad, formato, errores, cobertura, latencia,
throughput y estabilidad.

En el benchmark de threads, la comparacion principal ocurre dentro de cada
modelo. Antes de aceptar una recomendacion de `num_thread`, verifique si
`error_rate` y `response_valid_rate` indican ejecuciones validas.
Configuraciones con respuestas vacias o invalidas no deben elegirse solo por el
score agregado.

## Git

El repositorio versiona solamente:

- `.gitignore`
- `README.md`
- `benchmark_threads.py`
- `paralelismo.py`
- `paralelismo2.py`

Archivos sensibles, caches, resultados de los benchmarks y materiales auxiliares
permanecen locales y son ignorados por `.gitignore`.

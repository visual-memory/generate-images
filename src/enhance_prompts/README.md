# enhance_prompts

Executa o PromptEnhancer vendorizado em `3rdparty/PromptEnhancer` sobre uma
coluna textual de um dataset do Hugging Face. Dataset, split, colunas, template,
backend e parâmetros de geração são configuráveis no módulo; opções de execução
como GPUs, caminhos e limite de registros são parametrizadas pela CLI.

O pacote apenas adapta a interface do projeto vendorizado e não altera nem
reimplementa o código do modelo.

## Estrutura

- `enhance_dataset_shards.py`: CLI que distribui o dataset entre processos/GPU,
  mantém checkpoints e reúne os shards no dataset final.
- `vendor_prompt_enhancer.py`: adaptador para os backends `7b` e `v2` do
  PromptEnhancer vendorizado.
- `utils.py`: configuração de geração, montagem e limpeza de prompts e leitura
  e escrita dos checkpoints JSONL.

## Preparação

Inicialize o submódulo, instale as dependências e baixe o modelo de reprompt:

```bash
git submodule update --init --recursive
uv sync --extra prompt-enhancing
hf download tencent/HunyuanImage-2.1 \
  --include "reprompt/*" \
  --local-dir /raid/aluno_paulosantana/models/promptenhancer-7b
```

O caminho padrão esperado pelo script é:

```text
/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt
```

Use `--model-path` caso o modelo esteja em outro local.

## Execução

O comando recomendado recebe um identificador por processo. Identificadores
repetidos iniciam vários processos na mesma GPU e só devem ser usados quando
houver VRAM suficiente:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,1
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,0,0,0
```

Para uma única GPU/processo:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0
```

O comando `run`:

1. cria um shard para cada item informado em `--gpus`;
2. inicia um subprocesso por shard com `CUDA_VISIBLE_DEVICES` configurado;
3. aguarda todos os subprocessos;
4. reúne os checkpoints na ordem original do dataset;
5. salva o dataset final e, opcionalmente, abre um pull request no Hub.

Teste rápido com poucas linhas, descartando checkpoints anteriores do mesmo
número de shards:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run \
  --gpus 0,1 --limit 10 --force
```

Execução com abertura de pull request no Hugging Face Hub:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run \
  --gpus 0,1 --push-pr
```

## Comandos manuais

Os subcomandos abaixo permitem executar e reunir shards separadamente, o que é
útil para depuração ou retomada manual:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run-shard \
  --num-shards 2 --shard-index 0

PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards merge \
  --num-shards 2
```

Em `run-shard`, `--device` é repassado somente ao backend `v2`. O fluxo padrão
usa o backend `7b` e seleciona a GPU por meio de `CUDA_VISIBLE_DEVICES`.

## Parametrização

Há dois níveis de configuração.

### Dataset e geração

Os parâmetros que definem o conteúdo processado ficam nas constantes no início
de `enhance_dataset_shards.py`. Para adaptar o fluxo a outro dataset, configure:

- `DEFAULT_DATASET_ID`: identificador do dataset no Hugging Face Hub;
- `DEFAULT_SPLIT`: split que será processado;
- `DEFAULT_DESCRIPTION_COLUMN`: coluna textual usada como entrada;
- `DEFAULT_ENHANCED_COLUMN`: nome da nova coluna que receberá o resultado;
- `DEFAULT_TEMPLATE`: template do prompt, obrigatoriamente com o marcador
  `{description}`;
- `DEFAULT_BACKEND`: backend do PromptEnhancer (`7b` ou `v2`);
- `DEFAULT_DEVICE_MAP`: estratégia de alocação do modelo;
- `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_MAX_NEW_TOKENS` e
  `DEFAULT_SYS_PROMPT`: parâmetros de geração;
- `DEFAULT_CHECKPOINT_INTERVAL`: quantidade de amostras entre gravações do
  checkpoint;
- `DEFAULT_TARGET_REPO_ID`: repositório de destino usado com `--push-pr`;
- `DEFAULT_COMMIT_MESSAGE` e `DEFAULT_COMMIT_DESCRIPTION`: metadados do pull
  request no Hub.

Por exemplo, para processar a coluna `caption` e gravar o resultado em
`enhanced_caption`, ajuste as constantes desta forma:

```python
DEFAULT_DATASET_ID = "organizacao/meu-dataset"
DEFAULT_SPLIT = "train"
DEFAULT_DESCRIPTION_COLUMN = "caption"
DEFAULT_ENHANCED_COLUMN = "enhanced_caption"
DEFAULT_TEMPLATE = "Enhance this image description:\n{description}"
DEFAULT_TARGET_REPO_ID = "organizacao/meu-dataset-enriquecido"
```

O valor da coluna de entrada é inserido em `{description}`. Um template possível
é:

```text
Enhance the following image description:
{description}
```

A saída do modelo é normalizada antes de ser salva: tags `answer` e `think`, o
conteúdo de blocos `think`, caracteres CJK e espaços repetidos são removidos.
Se o modelo falhar em uma linha, o prompt de entrada é usado como fallback.

### Parâmetros operacionais da CLI

Esses argumentos permitem variar a execução sem editar o módulo. Em especial,
`--model-path`, `--output-dir`, `--final-output-dir` e `--target-repo-id`
sobrescrevem seus respectivos valores padrão quando disponíveis no subcomando.

`run` aceita:

```text
--gpus (obrigatório)
--model-path
--limit
--output-dir
--final-output-dir
--target-repo-id
--push-pr
--force
```

`run-shard` aceita:

```text
--num-shards (obrigatório)
--shard-index (obrigatório)
--model-path
--device
--limit
--output-dir
--force
```

`merge` aceita:

```text
--num-shards (obrigatório)
--limit
--output-dir
--final-output-dir
--target-repo-id
--push-pr
```

`--force` remove somente o checkpoint do shard que será executado. No comando
`run`, a opção é propagada para todos os shards iniciados.

## Checkpoints e saídas

Cada checkpoint é um arquivo JSONL que preserva o índice da linha original:

```json
{"row_index": 123, "enhanced_description": "..."}
```

Com o valor padrão de `DEFAULT_CHECKPOINT_INTERVAL`, os registros são gravados a
cada 50 amostras concluídas e novamente ao final do processo. Uma interrupção
inesperada pode, portanto, exigir o reprocessamento de até 49 linhas por
processo. Esse comportamento acompanha o intervalo configurado.

Por padrão, os checkpoints ficam em:

```text
outputs/enhanced-dataset-shards/num_shards-N/shard-XXXXX/enhanced_description.jsonl
```

O resultado reunido é salvo em:

```text
outputs/enhanced-dataset/dataset
outputs/enhanced-dataset/<split>.parquet
```

O merge exige o checkpoint de todos os shards e todas as linhas esperadas. A
coluna final é reconstruída por `row_index`, preservando a ordem do dataset de
origem.

## Verificação

```bash
uv run pytest
uv run python -m compileall src tests
```

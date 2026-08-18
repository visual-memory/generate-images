# Generate Images

Pipeline experimental para enriquecer descrições de personas, gerar retratos a
partir delas e publicar os datasets resultantes no Hugging Face Hub. O projeto
trabalha principalmente com os datasets `visual-memory/*-Mapping`, distribui o
processamento entre GPUs e mantém checkpoints locais para permitir retomadas.

## O que existe hoje

O repositório reúne quatro frentes:

1. **Melhoria de prompts** com o PromptEnhancer mantido como submódulo Git.
2. **Geração de imagens** por uma matriz configurável de datasets, modelos e
   variantes de prompt (`original` e `enhanced`).
3. **Pós-processamento de datasets**, incluindo associação de imagens por ID e
   validação da cobertura dos mappings.
4. **Preparação e exploração de dados** em notebooks para Synthetic Persona
   Chat, PersonaChat e ConvAI2.

O fluxo esperado é:

```text
dataset Mapping
    -> PromptEnhancer
    -> dataset com descrição melhorada
    -> modelo text-to-image
    -> dataset com coluna persona-image
    -> pull request no Hugging Face Hub (opcional)
```

Este é um projeto de pesquisa em evolução, não uma biblioteca estável. Os IDs
de datasets, nomes de colunas e caminhos de modelos ainda são parcialmente
específicos do ambiente em que o projeto foi desenvolvido.

## Estrutura

```text
.
├── main.py                                  # experimento direto com PromptEnhancer V2
├── add_images_by_id.py                      # associa imagens a mappings por persona-id
├── validate_persona_mapping_coverage.py     # verifica cobertura de IDs no Hub
├── src/
│   ├── enhance_prompts/
│   │   ├── enhance_dataset_shards.py        # pipeline multi-GPU de prompts
│   │   ├── utils.py                         # prompts, limpeza e checkpoints
│   │   └── vendor_prompt_enhancer.py        # adaptador do submódulo
│   ├── generate_images/
│   │   ├── generate_persona_images_shards.py
│   │   ├── orchestrate_persona_images_vllm.py
│   │   └── persona_image_generation.yaml    # matriz de execução
│   └── dataset_persona_mapping/             # notebooks de preparação/mapeamento
├── tests/
└── 3rdparty/PromptEnhancer/                  # submódulo; código de terceiros
```

Não edite `3rdparty/PromptEnhancer` como parte de mudanças comuns no projeto.
Atualizações dessa pasta devem ser feitas como atualizações intencionais do
submódulo.

## Requisitos

- Linux;
- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- Git com suporte a submódulos;
- GPUs NVIDIA e uma instalação CUDA compatível para os pipelines de modelos;
- conta e token do Hugging Face para modelos/datasets restritos ou publicação.

O perfil de geração fixa `vllm==0.26.0`, `vllm-omni==0.26.0` e Transformers 5.
O lockfile atual resolve dependências CUDA 13, portanto confirme a
compatibilidade do driver e da GPU antes de baixar o ambiente, que é grande.

## Instalação

Clone o repositório e inicialize o PromptEnhancer:

```bash
git submodule update --init --recursive
```

Para instalar somente o ambiente base, usado pelos notebooks:

```bash
uv sync
```

Há dois perfis pesados e incompatíveis entre si no `pyproject.toml`. Instale o
perfil correspondente à tarefa atual; não passe os dois extras juntos.

### Perfil de melhoria de prompts

```bash
uv sync --extra prompt-enhancing
```

O pipeline usa por padrão o PromptEnhancer 7B neste caminho:

```text
/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt
```

Baixe o modelo para esse local ou informe outro diretório com `--model-path`:

```bash
uv run hf download tencent/HunyuanImage-2.1 \
  --include "reprompt/*" \
  --local-dir /raid/aluno_paulosantana/models/promptenhancer-7b
```

O `main.py` é apenas um experimento isolado com o PromptEnhancer V2 e possui um
caminho hardcoded diferente. Se quiser executá-lo, baixe também:

```bash
uv run hf download PromptEnhancer/PromptEnhancer-32B \
  --local-dir /raid/aluno_paulosantana/models/promptenhancer-32b
```

### Perfil de geração de imagens

```bash
uv sync --extra image-generation --extra transformers5
```

Autentique a CLI antes de acessar datasets privados, publicar resultados ou
usar modelos gated:

```bash
uv run hf auth login
```

Para modelos como `black-forest-labs/FLUX.2-dev`, também é necessário aceitar
as condições de acesso na página do modelo antes da execução.

## Configuração da geração

A matriz de geração fica em
`src/generate_images/persona_image_generation.yaml`. Ela define:

- `output_root`, `split`, coluna de ID e coluna de imagem;
- parâmetros globais como resolução e seed;
- datasets e as colunas usadas pelas variantes `original` e `enhanced`;
- modelos, sufixos dos datasets de saída e parâmetros específicos.

No estado atual, somente esta combinação está ativa no YAML:

| Chave | Recurso |
| --- | --- |
| Dataset `synthetic-persona-chat` | `visual-memory/Synthetic-Persona-Chat-Mapping_1k` |
| Modelo `ernie` | `baidu/ERNIE-Image` |
| Variantes | `description` e `enhanced_description` |

As entradas de PersonaChat, ConvAI2, Qwen e FLUX estão presentes, porém
comentadas. As opções `--dataset` e `--model` recebem as **chaves do YAML**, não
os IDs completos do Hub.

O nome de saída é derivado automaticamente no formato:

```text
<namespace>/<dataset>-<dataset_suffix>-<variant>
```

## Gerar imagens

### Matriz completa com vLLM-Omni

Este é o orquestrador para executar todos os datasets e modelos ativos no YAML,
sempre nas variantes `original` e `enhanced`:

```bash
uv run python -m generate_images.orchestrate_persona_images_vllm \
  --config src/generate_images/persona_image_generation.yaml \
  --gpus 0,1 \
  --push-pr
```

Cada GPU selecionada recebe uma réplica do modelo. Os jobs são executados em
sequência e as imagens de cada lote são produzidas em paralelo pelas réplicas.
Use `--limit 2` para um smoke test. Sem `--push-pr`, o resultado permanece
somente em disco.

As saídas ficam em:

```text
<output_root>/<target_dataset>/persona-images-vllm/
├── images/*.png
├── persona_images.jsonl
├── dataset/
└── train.parquet
```

O checkpoint considera uma linha concluída somente se o arquivo de imagem
ainda existir. Uma nova execução retoma as linhas restantes. `--force` apaga a
pasta de saída de cada job selecionado antes de regenerá-lo.

### Um job em shards independentes

O gerador alternativo inicia um processo por GPU e executa uma única combinação
de dataset, modelo e variante:

```bash
uv run python -m generate_images.generate_persona_images_shards run \
  --config src/generate_images/persona_image_generation.yaml \
  --dataset synthetic-persona-chat \
  --model ernie \
  --variant enhanced \
  --gpus 0,1 \
  --limit 10
```

Ao fim, os shards são reunidos preservando a ordem original. Acrescente
`--push-pr` para abrir um pull request no dataset de destino. Os subcomandos
`run-shard` e `merge` permitem operar as duas etapas manualmente.

As saídas desse modo ficam sob:

```text
<output_root>/<target_dataset>/persona-images-shards/
<output_root>/<target_dataset>/persona-images/
```

## Melhorar descrições de personas

O pipeline atual é configurado por constantes no início de
`src/enhance_prompts/enhance_dataset_shards.py`. Por padrão ele lê:

- dataset `visual-memory/ConvAI2-Mapping`;
- split `train`;
- coluna `persona_revised`;
- modelo 7B;
- nova coluna `enhanced_persona_revised`.

Esses padrões não correspondem ao dataset Synthetic Persona Chat ativo no YAML
de geração. Ajuste as constantes ou o dataset de entrada antes de encadear os
dois pipelines.

Execute um teste limitado em duas GPUs:

```bash
uv run python -m enhance_prompts.enhance_dataset_shards \
  run \
  --gpus 0,1 \
  --limit 10
```

Para usar outro local de modelo e publicar por pull request:

```bash
uv run python -m enhance_prompts.enhance_dataset_shards \
  run \
  --gpus 0,1 \
  --model-path /caminho/para/reprompt \
  --target-repo-id organizacao/dataset \
  --push-pr
```

Os checkpoints são JSONL e ficam, por padrão, em
`outputs/enhanced-dataset-shards/`. O merge cria
`outputs/enhanced-dataset/dataset/` e `outputs/enhanced-dataset/train.parquet`.
Se uma linha falhar, o pipeline registra o erro e usa o prompt preparado como
fallback. Consulte `src/enhance_prompts/README.md` para os detalhes internos e
os subcomandos `run-shard` e `merge`.

## Utilitários de datasets

### Associar imagens por ID

`add_images_by_id.py` transfere `persona-image` de um dataset para outro usando
`persona-id`, valida IDs nulos/duplicados/ausentes e preserva a ordem do mapping:

```bash
uv run python add_images_by_id.py \
  --mapping-dataset visual-memory/Meu-Mapping \
  --image-dataset visual-memory/Meu-Dataset-Com-Imagens \
  --target-dataset visual-memory/Meu-Mapping-Com-Imagens
```

Esse comando **sempre publica um pull request** no dataset de destino; não há
modo local ou dry-run na implementação atual.

### Validar cobertura dos mappings

```bash
uv run python validate_persona_mapping_coverage.py --max-missing 20
```

O script é somente leitura e verifica pares hardcoded de Synthetic Persona
Chat, PersonaChat e ConvAI2 no Hugging Face Hub. Ele termina com código `1` se
algum ID referenciado não existir no mapping correspondente.

## Testes e estado conhecido

O comando previsto é:

```bash
uv run pytest
```

No estado atual do diretório de trabalho, a suíte ainda não está verde:

- `tests/test_enhance_dataset.py` importa o módulo removido
  `enhance_prompts.enhance_dataset`, causando erro durante a coleta;
- os testes do gerador sharded ainda refletem a API anterior à configuração por
  YAML e três deles falham;
- os demais 9 testes dos módulos sharded passam.

As CLIs principais foram importadas e tiveram o `--help` validado. Uma execução
fim a fim exige acesso ao Hub, pesos dos modelos e GPUs compatíveis, e por isso
não faz parte dos testes unitários locais.

## Cuidados operacionais

- Modelos e saídas podem ocupar centenas de gigabytes; mantenha-os fora do Git.
- `--push-pr` cria mudanças externas no Hugging Face Hub.
- `--force` remove checkpoints e imagens locais dentro da pasta do job ou shard.
- O `output_root` padrão aponta para `/raid/aluno_paulosantana`; altere-o no YAML
  em outras máquinas.
- Não versionar tokens, pesos, `.venv`, `outputs/` ou logs.

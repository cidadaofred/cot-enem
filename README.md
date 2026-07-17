# CoT-ENEM

Projeto acadêmico inspirado em ChainLM/CoTGenius para gerar um dataset de questões do
ENEM com cadeias de raciocínio e genealogia rastreável.

## Arquitetura

```text
XML bruto -> parser -> normalizador -> filtros -> JSONL normalizado
                                                |-> Specify
                                                |-> Complicate -> validação -> CoT-ENEM
                                                `-> Diversify
```

`Specify` aprimora o raciocínio sem alterar a questão. `Complicate` cria uma questão
mais difícil no mesmo domínio. `Diversify` cria uma questão estruturalmente diferente
na mesma competência. As três estratégias partem da questão raiz.

As Fases 1, 2 e 3 estão implementadas: fundação do dataset, infraestrutura
independente de provedor e primeiro fluxo funcional de CoT inicial + Specify.

## Instalação e uso

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m cot_enem.cli prepare --input data/raw/enem.xml --output data/processed/enem_normalized.jsonl
pytest
```

Baixe o XML da [página oficial](https://www.ime.usp.br/~ddm/project/enem/) e preserve-o
em `data/raw/`. O comando não altera o XML. Itens dependentes de imagem permanecem no
JSONL com `eligible=false` para auditoria.

## Decisões e limitações

Pydantic v2 estabiliza os contratos normalizado e evoluído; `src` layout separa lógica
de scripts e notebooks; JSONL UTF-8 permite escrita progressiva. O XML histórico pode
ter variações ainda não representadas pelos aliases atuais.

## Provedores de LLM

- `MockLLMProvider`: respostas determinísticas, sem rede, usado nos testes.
- `OpenAICompatibleProvider`: usa `LLM_BASE_URL`, `LLM_API_KEY` e `LLM_MODEL`; não
  depende de SDK proprietário.
- `HuggingFaceProvider`: adaptador opcional com carregamento tardio para execução local
  ou Colab. Instale com `pip install -e ".[huggingface]"`.

Todos retornam `LLMResponse`, suportam JSON estruturado e preservam metadados de uso.
Retries com backoff exponencial são aplicados a falhas do provedor remoto. Agentes,
métricas semânticas e Colab serão adicionados nas fases seguintes.

## CoT inicial e Specify

O projeto inclui um baseline local em `.env.example`:

```text
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
JUDGE_MODEL=qwen2.5:7b
```

Copie-o para `.env` e disponibilize o modelo no Ollama. A CLI carrega `.env`
automaticamente, mas variáveis definidas no sistema, IntelliJ, Colab ou CI têm
precedência. Para um serviço remoto, substitua URL, modelos e chave no `.env` local;
ele é ignorado pelo Git.

Depois execute:

```bash
python -m cot_enem.cli evolve \
  --strategies specify \
  --input data/processed/enem_normalized.jsonl \
  --output outputs/datasets/cot_enem_v1.jsonl \
  --limit 10
```

O CoT inicial é gerado sem envio do gabarito. A resposta é comparada
programaticamente, e o `SpecifyAgent` melhora as etapas sem alterar questão ou
alternativas. Juízes independentes avaliam evolução e correção. Aceitos e rejeitados
são escritos progressivamente; IDs determinísticos permitem retomar a execução sem
duplicatas. Nesta fase, a CLI aceita apenas `specify`.

## Execução portátil

A configuração segue a precedência:

```text
configs/default.yaml
        ↓
perfil local/colab/server/cpu/gpu
        ↓
variáveis COTENEM_* e LLM_*
        ↓
argumentos da CLI
```

Inspecione o ambiente e valide pré-requisitos:

```bash
python -m cot_enem.cli info --config configs/local.yaml
python -m cot_enem.cli verify --config configs/local.yaml
```

Seleção explícita:

```bash
python -m cot_enem.cli info --config configs/cpu.yaml --device cpu
python -m cot_enem.cli info --config configs/gpu.yaml --device cuda --precision auto
```

`device=auto` escolhe CUDA, depois MPS e por fim CPU. `precision=auto` usa BF16
somente quando a GPU declara suporte; caso contrário usa FP16 em aceleradores e FP32
em CPU. O projeto suporta Python 3.11 e 3.12; Python 3.14 é rejeitado por `verify`
para evitar incompatibilidades com bibliotecas científicas.

### Qual arquivo faz o quê?

- `configs/default.yaml`: configuração integral e reproduzível.
- `configs/local.yaml`, `colab.yaml`, `server.yaml`: sobrescritas por ambiente.
- `configs/cpu.yaml`, `gpu.yaml`: sobrescritas de dispositivo e precisão.
- `configuration/schemas.py`: contrato Pydantic da configuração.
- `configuration/loader.py`: merge hierárquico e mensagens de erro com origem.
- `runtime/environment.py`: detecção Windows, Linux, WSL, Colab, SLURM, RAM e GPU.
- `runtime/device.py`: escolha validada de CPU, CUDA, MPS e precisão.
- `runtime/context.py`: reúne configuração, ambiente, dispositivo e saída.
- `runtime/diagnostics.py`: implementa as verificações de `verify` e dados de `info`.
- `observability/logging_config.py`: logs no console e JSONL, sem ler segredos.

O código de domínio continua independente:

```text
CLI → configuração → ExecutionContext
                         ↓
                  pipeline/agentes
                         ↓
             Local | Colab | servidor
```

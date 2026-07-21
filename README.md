# CoT-ENEM

## Resultado experimental atual

O baseline concluído utiliza exclusivamente questões do **ENEM 2017**. Foram
normalizadas 86 questões: 33 elegíveis para o pipeline textual e 53 inelegíveis por
dependência de imagem. O Specify processou todas as 33 questões e produziu 20
registros aceitos e 13 rejeitados, uma taxa de aceitação de 60,6% entre os itens
elegíveis. Os rejeitados permanecem armazenados para auditoria.

O protocolo, as versões, os resultados, o consumo observado, as estimativas e as
limitações estão documentados em `docs/RELATORIO_FASE3_ENEM2017.md`.

### Independência dos agentes

Specify, Complicate e Diversify são ramos independentes de um CoT inicial comum:

```text
Questão-raiz
    |
    +--> CoT inicial congelado
            |
            +--> Specify
            +--> Complicate
            `--> Diversify
```

Os agentes não são encadeados entre si. Complicate não recebe a saída de Specify, e
Diversify não recebe a saída de Complicate. Isso reduz propagação de erros e preserva
uma condição inicial comparável. Os três ramos já foram executados para o ENEM 2017.

### Modelo e ambiente

Geração do CoT inicial, Specify, Evolution Success Judge e Correctness Judge usaram
`Qwen/Qwen2.5-7B-Instruct`. Uma única instância do modelo foi compartilhada entre
esses papéis, com quantização NF4 em 4 bits e FP16 numa NVIDIA Tesla T4 do Google
Colab. O parser e o normalizador são determinísticos e não utilizam LLM. A execução
não consumiu API paga; tokens, energia e pico de VRAM ainda não foram instrumentados.

### Filtragem por maioria, fiel ao ChainLM

O comando `ensemble-specify` implementa uma segunda versão da Fase 3 com três juízes
heterogêneos: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 e
Phi-3.5-mini-instruct. Cada modelo vota separadamente sobre sucesso evolutivo e
correção; dois votos aprovam por maioria. Candidatos, votos e consolidação são
persistidos em JSONL distintos. Na T4, os modelos são carregados sequencialmente e
descarregados antes do próximo juiz.

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

As Fases 1 a 6 estão implementadas: fundação, infraestrutura de LLM, Specify,
Complicate, Diversify e consolidação final com métricas e relatórios.

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

## Fase 3 remotamente no Colab

Para máquinas Windows com poucos recursos, use
`notebooks/fase3_colab.ipynb`. Ele não depende do Colab CLI nem do Ollama:
clona o repositório para `/content`, usa `Qwen/Qwen2.5-7B-Instruct`
quantizado em 4 bits na GPU e persiste entrada, cache e resultados no Google
Drive. Consulte o procedimento completo em `docs/EXECUCAO_COLAB.md`.

## Fase 4: Complicate e Diversify

A Fase 4 reutiliza o CoT inicial congelado da Fase 3 e executa `Complicate` e
`Diversify` como ramos independentes, cada um com candidatos, votos e resultados
próprios. A linhagem é registrada por `root_id`, `parent_id` e `generation`. O mesmo
contrato de sementes aceita resultados aprovados de uma geração anterior, permitindo
uma futura reprodução das rodadas iterativas do ChainLM sem acoplar as fases.

Use `notebooks/fase4_colab.ipynb` para execução remota e consulte `docs/FASE4.md`
para arquitetura, arquivos, comandos e auditoria.

## Fase 6: consolidação final

O comando `finalize` reúne os resultados das três estratégias sem carregar LLMs,
separa aceitos e rejeitados e gera métricas em JSON/CSV e relatório Markdown. A
execução é CPU-only, idempotente e preserva todos os artefatos anteriores. Consulte
`docs/FASE6.md` ou execute `notebooks/fase6_analysis.ipynb` no Colab.

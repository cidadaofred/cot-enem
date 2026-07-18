# Fase 4 — Complicate e Diversify

## Objetivo

A Fase 4 implementa os ramos `Complicate` e `Diversify` sem alterar os artefatos do
`Specify`. Os dois ramos reutilizam o CoT inicial persistido na Fase 3 e escrevem
candidatos, votos e resultados em arquivos próprios.

```text
CoT inicial congelado
        |
        +--> Specify    (Fase 3)
        +--> Complicate (Fase 4)
        `--> Diversify  (Fase 4)
```

Essa configuração permite comparar as três estratégias sob a mesma entrada. Ela não
impõe uma cascata fixa entre agentes.

## Sementes e linhagem

O pipeline extrai de `phase3_candidates.jsonl` somente a questão-raiz, o CoT inicial,
o modelo gerador e o identificador do artefato. O campo `improved` do Specify não é
usado. Cada CoT inicial recebe um ID neutro `cotseed_*`.

Os resultados mantêm `root_id`, `parent_id`, `generation` e `strategy`, registrando a
questão original, a semente efetivamente utilizada e a geração evolutiva.

## Comportamento

`Complicate` cria uma questão mais difícil, acrescentando restrições ou aprofundando
o raciocínio. `Diversify` cria uma questão inspirada na entrada, mas com cenário ou
núcleo temático diferente.

Ambos retornam cabeçalho, enunciado, alternativas A–E, etapas de raciocínio e resposta
final. Cada candidato recebe votos sequenciais de Qwen, Mistral e Phi sobre sucesso
evolutivo e correção. Cada critério exige dois votos favoráveis.

## Arquivos

```text
outputs/datasets/
├── shared/
│   └── initial_cot_seeds.jsonl
├── complicate/
│   ├── candidates.jsonl
│   ├── judge_votes.jsonl
│   └── results.jsonl
└── diversify/
    ├── candidates.jsonl
    ├── judge_votes.jsonl
    └── results.jsonl
```

Os arquivos são incrementais e retomáveis. Reexecutar o mesmo comando não duplica
sementes, candidatos, votos ou resultados.

## Execução

Use `notebooks/fase4_colab.ipynb`. Ele inicia com `LIMIT = 1` e uma estratégia por
sessão para reduzir consumo de GPU.

Exemplo para Complicate:

```bash
python -m cot_enem.cli ensemble-evolve \
  --config configs/colab.yaml \
  --strategy complicate \
  --initial-candidates outputs/datasets/phase3_candidates.jsonl \
  --seeds outputs/datasets/shared/initial_cot_seeds.jsonl \
  --candidates outputs/datasets/complicate/candidates.jsonl \
  --votes outputs/datasets/complicate/judge_votes.jsonl \
  --output outputs/datasets/complicate/results.jsonl \
  --limit 1
```

Para Diversify, altere a estratégia e os três caminhos do ramo.

## Evolução iterativa futura

O artigo ChainLM evolui dados por quatro rodadas. O mesmo pipeline aceita resultados
aprovados de uma geração anterior usando:

```bash
--parent-results outputs/datasets/complicate/results.jsonl
```

Somente resultados aceitos são convertidos em novas sementes. O novo registro
preserva o `root_id`, usa o resultado anterior como pai e incrementa `generation`.
Isso permite alternar estratégias em rodadas futuras sem criar dependência entre as
implementações das fases.

## Auditoria

O notebook termina executando `audit_phase4_branch` em CPU. A auditoria valida
schemas, duplicidades, linhagem, estratégia, geração, três votos, maioria e
completude. Com `LIMIT` definido, um subconjunto consistente é aceito. Sem limite, a
execução integral deve retornar `complete: True`.

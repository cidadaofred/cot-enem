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

### Conteúdo e significado dos artefatos

| Arquivo | Conteúdo | O que representa |
|---|---|---|
| `shared/initial_cot_seeds.jsonl` | Uma questão-raiz, seu CoT inicial congelado, modelo gerador, IDs de origem e geração | Entrada comum e neutra dos agentes. Permite comparar `Complicate` e `Diversify` sem usar a saída do `Specify` como entrada e sem criar dependência entre estratégias. |
| `complicate/candidates.jsonl` | Questão original, CoT inicial, questão mais difícil gerada, novo CoT, resposta, estratégia, modelo, prompt, temperatura, execução e linhagem | Os 33 candidatos brutos produzidos pelo agente `Complicate`, antes da votação. Não significa que todos foram aprovados. |
| `complicate/judge_votes.jsonl` | Um registro por par candidato–juiz, contendo decisões e justificativas separadas para sucesso evolutivo e correção | Evidência auditável dos 99 votos de `Complicate`: 33 candidatos × 3 modelos julgadores. |
| `complicate/results.jsonl` | Questão evoluída consolidada, CoTs inicial e melhorado, metadados, três votos, maiorias, motivos de rejeição e `status` | Resultado final de `Complicate`. Contém tanto os 30 aceitos quanto os 3 rejeitados; somente registros com `validation.accepted=true` devem entrar no dataset aprovado. |
| `diversify/candidates.jsonl` | Questão original, CoT inicial, questão diversificada gerada, novo CoT, resposta, estratégia, modelo, prompt, temperatura, execução e linhagem | Os 33 candidatos brutos produzidos pelo agente `Diversify`, antes da votação. |
| `diversify/judge_votes.jsonl` | Um registro por par candidato–juiz, com decisões e justificativas para sucesso evolutivo e correção | Evidência auditável dos 99 votos de `Diversify`: 33 candidatos × 3 modelos julgadores. |
| `diversify/results.jsonl` | Questão evoluída consolidada, raciocínio, metadados, votos, maiorias, motivos de rejeição e `status` | Resultado final de `Diversify`. Contém os 29 aceitos e os 4 rejeitados; somente os aceitos compõem o dataset aprovado. |

Os arquivos `candidates.jsonl` preservam a produção original do gerador; os arquivos
`judge_votes.jsonl` preservam a avaliação individual; e os arquivos `results.jsonl`
materializam a decisão majoritária. Essa separação evita perda de evidência e permite
refazer a agregação sem regenerar candidatos nem repetir inferências já persistidas.

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

## Relatório de execução integral — ENEM 2017

A execução integral da Fase 4 foi concluída e auditada sobre as 33 sementes obtidas
dos candidatos da Fase 3. Os dois agentes partiram independentemente do mesmo CoT
inicial congelado. O `Qwen/Qwen2.5-7B-Instruct` gerou as evoluções. Os modelos
`Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` e
`microsoft/Phi-3.5-mini-instruct` atuaram sequencialmente como juízes. Para cada candidato, cada juiz
avaliou dois critérios: sucesso da estratégia de evolução e correção da nova questão.
Um critério foi aprovado quando recebeu ao menos dois votos favoráveis; a aceitação
final também exigiu formato válido.

| Estratégia | Sementes | Candidatos | Votos | Aceitos | Rejeitados | Taxa de aceitação | Auditoria |
|---|---:|---:|---:|---:|---:|---:|---|
| Complicate | 33 | 33 | 99 | 30 | 3 | 90,9% | Aprovada |
| Diversify | 33 | 33 | 99 | 29 | 4 | 87,9% | Aprovada |
| **Total dos ramos** | **66 execuções de agente** | **66** | **198** | **59** | **7** | **89,4%** | **Completa** |

O total de 66 na linha consolidada representa 33 sementes processadas uma vez por
cada estratégia, e não 66 questões-raiz distintas. Como resultado concreto, a Fase
4 produziu 59 novas amostras aprovadas: 30 de `Complicate` e 29 de `Diversify`.
Também preservou sete rejeições para rastreabilidade metodológica.

As duas auditorias retornaram `complete: True`, com zero candidatos ausentes, zero
votos incompletos e zero resultados ausentes. Em cada ramo, a igualdade
`votes = candidates × 3` foi satisfeita. Portanto, os artefatos estão completos para
análise, consolidação posterior e eventual uso como sementes de novas rodadas.

### Interpretação dos resultados

`Complicate` apresentou a maior taxa de aceitação, com 90,9%, enquanto `Diversify`
alcançou 87,9%. A diferença é de 3,0 pontos percentuais e, neste recorte pequeno, deve
ser tratada de forma descritiva, não como evidência estatística de superioridade.

Os 59 aceitos podem seguir para uma consolidação de dataset, mantendo `strategy`,
`root_id`, `parent_id` e `generation`. Os sete rejeitados não devem ser descartados
dos artefatos de pesquisa: eles documentam a atuação do filtro, seus motivos e os
votos individuais, embora não devam integrar o subconjunto final aprovado.

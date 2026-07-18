# Relatório técnico — Fase 3 do CoT-ENEM

**Data da execução:** 17–18 de julho de 2026  
**Escopo:** ENEM 2017  
**Estratégia executada:** Specify  
**Estado do projeto:** implementação da Fase 3 concluída; Complicate e Diversify ainda
não executados. A conclusão do processamento integral do ensemble é atestada pela
auditoria descrita na Seção 13.

## 1. Objetivo

Esta etapa avaliou um baseline para construção do CoT-ENEM a partir de questões
objetivas do ENEM 2017. O fluxo normaliza o XML, gera uma cadeia de raciocínio inicial
sem expor o gabarito ao gerador, aplica uma estratégia evolutiva e submete o resultado
a validações programáticas e julgamentos por LLM.

O experimento descrito aqui utilizou **somente a prova de 2017**. Portanto, os números
não devem ser generalizados para outras edições do ENEM sem experimentos adicionais.

## 2. Dados

| Indicador | Resultado |
|---|---:|
| Questões normalizadas | 86 |
| Questões elegíveis para processamento textual | 33 (38,4%) |
| Questões inelegíveis | 53 (61,6%) |
| Motivo de inelegibilidade implementado | dependência de imagem |
| Registros Specify produzidos | 33 |
| Cobertura das questões elegíveis | 100% |

As 53 questões inelegíveis permanecem no JSONL normalizado para auditoria, com
`eligible=false` e a respectiva justificativa. Elas não são enviadas ao modelo porque
o baseline atual é textual e não possui entrada multimodal.

## 3. Arquitetura experimental

As estratégias evolutivas são independentes e formam ramos irmãos. Elas não devem ser
executadas em cascata:

```text
Questão-raiz elegível
        |
        +--> CoT inicial congelado
                |
                +--> Specify
                +--> Complicate
                `--> Diversify
```

Cada agente recebe a questão-raiz e o mesmo CoT inicial. Dessa forma, o resultado de
Specify não é entrada de Complicate, e o resultado de Complicate não é entrada de
Diversify. Essa decisão reduz propagação de erros e permite comparar as estratégias
sob a mesma condição inicial.

Nesta fase, somente o ramo Specify foi implementado e executado. O fluxo por questão
realizou quatro inferências principais:

1. geração do CoT inicial;
2. evolução pelo Specify;
3. julgamento de sucesso da evolução;
4. julgamento de correção.

Além dos juízes por LLM, validadores determinísticos conferem formato e coincidência
da alternativa prevista com o gabarito.

## 4. Modelos e configurações

| Componente | Modelo | Função |
|---|---|---|
| Gerador do CoT inicial | `Qwen/Qwen2.5-7B-Instruct` | Resolver a questão sem receber o gabarito |
| Specify | `Qwen/Qwen2.5-7B-Instruct` | Tornar o raciocínio mais explícito sem alterar a questão |
| Evolution Success Judge | `Qwen/Qwen2.5-7B-Instruct` | Comparar CoT inicial e evoluído |
| Correctness Judge | `Qwen/Qwen2.5-7B-Instruct` | Avaliar correção com acesso controlado ao gabarito |
| Parser e normalizador | código determinístico | Converter o XML do ENEM 2017 para o schema canônico |

Uma única instância do Qwen foi compartilhada entre geração e julgamento para evitar
carregar o modelo duas vezes na VRAM. A configuração utilizou quantização NF4 em 4
bits, computação FP16 e respostas estruturadas em JSON. Respostas com escapes LaTeX
inválidos são reparadas de forma restrita e respostas malformadas têm até três
tentativas.

## 5. Ambiente de execução

| Item | Valor observado |
|---|---|
| Ambiente | Google Colab convencional |
| GPU | NVIDIA Tesla T4 |
| Precisão | FP16 |
| Quantização | 4 bits, NF4, `bitsandbytes` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| Accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| Cache do modelo | `/content/hf-cache` |
| Persistência | Google Drive, JSONL incremental |

O cache foi mantido no disco temporário do Colab, que apresentava 65,54 GB livres. O
Drive apresentava 3,2 GB livres e foi reservado para entrada, logs e resultados. IDs
determinísticos permitem retomar a execução sem duplicar registros após interrupções.

## 6. Resultados da estratégia Specify

| Resultado | Quantidade | Percentual entre elegíveis |
|---|---:|---:|
| Aceitos | 20 | 60,6% |
| Rejeitados | 13 | 39,4% |
| Total | 33 | 100% |

Considerando as 86 questões normalizadas, os 20 registros aceitos representam 23,3%
do corpus original. Esse percentual combina duas barreiras diferentes: elegibilidade
textual e aprovação do exemplo evoluído. Os 13 rejeitados não são descartados da
trilha de auditoria; eles preservam os critérios e motivos de reprovação.

## 7. Consumo e tempo

O experimento não utilizou API paga: a inferência ocorreu localmente na GPU do Colab.
Consequentemente, não houve cobrança por tokens. A versão atual ainda não registra
automaticamente contagem de tokens, energia ou pico de VRAM; esses valores não devem
ser apresentados como medições.

### Valores observados

- download inicial dos quatro shards: aproximadamente 2,3 a 2,8 minutos nas tentativas
  registradas;
- carregamento quantizado do modelo: aproximadamente 63 a 70 segundos por processo;
- quatro inferências principais por questão, totalizando no mínimo 132 chamadas para
  as 33 questões; tentativas de correção de JSON podem elevar esse número;
- escrita incremental de um registro por questão concluída.

### Estimativas para planejamento

- cache dos pesos originais do Qwen 7B: cerca de 15 GB;
- ocupação esperada em VRAM após quantização, incluindo sobrecargas: aproximadamente
  5 a 8 GB;
- tempo de inferência por questão na T4: ordem de 2 a 4 minutos, dependendo do tamanho
  dos prompts, respostas e tentativas;
- tempo total estimado para 33 questões: aproximadamente 1,1 a 2,2 GPU-horas, além do
  download e carregamento inicial.

Essas faixas são estimativas de engenharia baseadas na configuração e nos logs
parciais. Para publicação como benchmark, a próxima execução deverá registrar
timestamps por item, tokens de entrada/saída, pico de VRAM e duração total.

## 8. Estimativa para as próximas estratégias

Não há resultados medidos de Complicate ou Diversify. Se, apenas como cenário de
planejamento, cada uma obtiver taxa de aceitação entre 50% e 70%, seriam esperados
aproximadamente 17 a 23 exemplos aceitos por estratégia entre as 33 raízes elegíveis.
Essa faixa não é hipótese confirmada nem resultado experimental.

As três estratégias juntas exigiriam ao menos 396 inferências principais se cada ramo
mantiver quatro chamadas por questão, sem contar retries. O custo real de Complicate
e Diversify poderá ser maior porque também precisam produzir ou validar novos
enunciados e alternativas.

## 9. Limitações

- corpus restrito ao ENEM 2017;
- ausência de suporte multimodal para 53 itens dependentes de imagem;
- mesmo modelo usado como gerador e juiz, o que pode introduzir viés correlacionado;
- amostra pequena, com apenas 33 raízes textuais;
- ausência de avaliação humana nesta fase;
- métricas de tokens, energia, VRAM máxima e latência por item ainda não instrumentadas;
- Complicate e Diversify ainda não possuem resultados.

## 10. Próximos passos

1. congelar `cot_enem_specify_v1_all.jsonl`, a partição aceita e a rejeitada;
2. auditar qualitativamente os 13 exemplos rejeitados e uma amostra dos aceitos;
3. registrar commit, configuração resolvida e hashes dos artefatos;
4. implementar Complicate e Diversify como ramos independentes do CoT inicial;
5. adicionar métricas de tempo, tokens, memória e falhas por item;
6. repetir o protocolo em outras edições do ENEM;
7. considerar um juiz diferente ou avaliação humana para reduzir viés.

## 11. Resumo para o artigo

No baseline do ENEM 2017, 86 questões foram normalizadas. Destas, 33 eram elegíveis
para o pipeline textual e todas foram processadas pela estratégia Specify. O sistema
produziu 20 exemplos aceitos e 13 rejeitados, correspondendo a uma taxa de aceitação
de 60,6% entre os itens elegíveis. A execução utilizou Qwen2.5-7B-Instruct em uma GPU
Tesla T4, com quantização NF4 em 4 bits e computação FP16. Geração, evolução e
julgamento compartilharam uma única instância do modelo. A arquitetura mantém
Specify, Complicate e Diversify independentes a partir de um CoT inicial comum, de
modo a evitar propagação de erros e permitir comparação direta entre estratégias.

## 12. Revisão metodológica: ensemble de juízes

Após o baseline de juiz único, a Fase 3 foi ampliada para aproximar a filtragem do
artigo ChainLM. Os 33 candidatos já gerados pelo Qwen são reutilizados sem nova
amostragem. Cada candidato recebe votos independentes de:

- `Qwen/Qwen2.5-7B-Instruct`;
- `mistralai/Mistral-7B-Instruct-v0.3`;
- `microsoft/Phi-3.5-mini-instruct`.

Cada juiz avalia sucesso evolutivo e correção. A aprovação em cada dimensão exige dois
dos três votos. Os modelos são carregados sequencialmente na T4 e os votos são
persistidos antes da troca de modelo. Essa revisão permite comparar diretamente juiz
único e maioria heterogênea sobre os mesmos candidatos. Os resultados do ensemble
devem ser relatados separadamente após a nova execução; a taxa de 60,6% permanece
identificada como resultado do baseline de juiz único.

## 13. Entregáveis concretos da Fase 3

A Fase 3 entrega um pipeline reproduzível para gerar candidatos CoT com a estratégia
Specify e filtrá-los por maioria heterogênea. O gerador
`Qwen/Qwen2.5-7B-Instruct` produz o CoT inicial e sua evolução Specify. Em seguida,
três modelos avaliam cada candidato de forma independente:

- `Qwen/Qwen2.5-7B-Instruct`;
- `mistralai/Mistral-7B-Instruct-v0.3`;
- `microsoft/Phi-3.5-mini-instruct`.

Cada juiz emite duas decisões estruturadas: sucesso da evolução e correção do
raciocínio. Assim, cada candidato completo possui seis decisões, correspondentes a
três modelos multiplicados por dois critérios. A decisão agregada em cada critério
exige pelo menos dois votos favoráveis.

### 13.1 Artefatos persistentes

| Artefato | Conteúdo |
|---|---|
| `data/processed/enem_normalized.jsonl` | 86 questões normalizadas, incluindo elegibilidade e justificativas |
| `outputs/datasets/phase3_candidates.jsonl` | CoT inicial e candidato produzido pelo Specify |
| `outputs/datasets/phase3_judge_votes.jsonl` | Votos individuais, modelo julgador e justificativas |
| `outputs/datasets/cot_enem_specify_ensemble_v1.jsonl` | Resultado agregado por maioria, incluindo aceitos e rejeitados |

Os arquivos intermediários fazem parte do resultado experimental. Eles permitem
recalcular a maioria, estudar concordância entre juízes e retomar uma execução
interrompida sem gerar novamente etapas concluídas.

### 13.2 Garantias operacionais

A implementação inclui:

- gravação incremental em JSONL;
- IDs determinísticos e prevenção de duplicidades;
- retomada de candidatos, votos e resultados persistidos;
- carregamento sequencial e liberação de VRAM entre os modelos;
- quantização NF4 em 4 bits e computação FP16 para a Tesla T4;
- aplicação de `--limit` à geração, aos três juízes e à agregação;
- validação de tipos antes da persistência;
- tentativas de correção para respostas estruturadas inválidas;
- normalização controlada de variações JSON retornadas pelos modelos;
- rejeição de respostas que contenham apenas o molde do JSON Schema;
- preservação explícita de votos recuperados de saídas truncadas;
- auditoria final executável inteiramente em CPU.

### 13.3 Auditoria e critério de conclusão

O notebook `notebooks/fase3_colab.ipynb` termina com uma auditoria que valida os
schemas dos quatro JSONL, duplicidades, referências entre questão, candidato, voto e
resultado, presença dos três modelos, recálculo independente da maioria e coerência
entre `validation.accepted` e o status final.

Durante smoke tests, com `LIMIT` definido, a auditoria aceita um subconjunto
estruturalmente consistente. Para uma execução integral, sem `LIMIT`, a Fase 3 somente
é considerada completamente processada quando a saída apresenta:

```text
AUDITORIA ESTRUTURAL: APROVADA
PROCESSAMENTO COMPLETO: True
```

`AUDITORIA ESTRUTURAL: APROVADA` com `PROCESSAMENTO COMPLETO: False` significa que
os artefatos existentes são válidos, mas ainda há candidatos, votos ou resultados
pendentes. Dessa forma, conclusão da implementação e conclusão de uma execução
integral são registradas separadamente.

### 13.4 Síntese

Concretamente, a Fase 3 estabeleceu a infraestrutura de geração, evolução,
julgamento heterogêneo, persistência, retomada, votação majoritária e auditoria do
ramo Specify para o ENEM 2017. O baseline anterior, com 20 aceitos e 13 rejeitados,
permanece identificado como resultado de juiz único. As métricas do ensemble devem
ser extraídas do arquivo `cot_enem_specify_ensemble_v1.jsonl` após a auditoria
integral, sem substituir ou misturar os dois protocolos.

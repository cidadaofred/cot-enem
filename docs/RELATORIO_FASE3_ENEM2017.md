# Relatório técnico — Fase 3 do CoT-ENEM

**Data da execução:** 17 de julho de 2026  
**Escopo:** ENEM 2017  
**Estratégia executada:** Specify  
**Estado do projeto:** Fase 3 concluída; Complicate e Diversify ainda não executados

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

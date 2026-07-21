# Fase 6 — Consolidação, métricas e relatório final

## Objetivo

A Fase 6 fecha o primeiro ciclo do CoT-ENEM sem realizar novas inferências. Ela lê
os resultados persistidos das Fases 3 e 4, valida a integridade global e produz uma
visão consolidada. Nenhum candidato, voto ou resultado anterior é alterado.

```text
normalized + Specify/results + Complicate/results + Diversify/results
                              |
                              v
                    validação de schemas
                    estratégia e linhagem
                              |
                 +------------+------------+
                 |                         |
             aceitos.jsonl             rejeitados.jsonl
                 |                         |
                 +------------+------------+
                              |
                    métricas JSON/CSV + relatório MD
```

## Execução no Colab

Esta etapa usa apenas CPU e pode ser executada mesmo sem acesso a GPU. Considerando
o mesmo `DRIVE_ROOT` das fases anteriores:

```bash
python -m cot_enem.cli finalize \
  --normalized /content/drive/MyDrive/cot-enem/data/processed/enem_normalized.jsonl \
  --specify /content/drive/MyDrive/cot-enem/outputs/datasets/cot_enem_specify_ensemble_v1.jsonl \
  --complicate /content/drive/MyDrive/cot-enem/outputs/datasets/complicate/results.jsonl \
  --diversify /content/drive/MyDrive/cot-enem/outputs/datasets/diversify/results.jsonl \
  --output-dir /content/drive/MyDrive/cot-enem/outputs/final
```

O notebook `notebooks/fase6_analysis.ipynb` executa o mesmo procedimento e mostra
as métricas sem carregar modelos.

## Arquivos produzidos

| Arquivo | Finalidade |
|---|---|
| `cot_enem_2017_accepted.jsonl` | Dataset final de registros aprovados, ordenado por raiz, estratégia e ID |
| `cot_enem_2017_rejected.jsonl` | Rejeitados preservados para auditoria e análise dos filtros |
| `metrics.json` | Métricas completas, motivos de rejeição, passos de CoT e concordância dos juízes |
| `strategy_metrics.csv` | Resumo tabular por estratégia para análise estatística |
| `REPORT.md` | Relatório gerado automaticamente a partir dos artefatos reais |

Os arquivos são reprodutíveis e idempotentes: executar novamente com as mesmas
entradas substitui atomicamente as saídas pelo mesmo conteúdo, sem anexar duplicatas.

## Verificações

O comando falha antes da publicação das saídas quando encontra:

- JSONL ausente ou inválido;
- registro que não corresponde à estratégia declarada pelo arquivo;
- ID duplicado entre os três ramos;
- referência a questão-raiz inexistente no dataset normalizado;
- divergência entre `status` e `validation.accepted`.

O resumo final informa `complete: true` somente após essas verificações. As métricas
de concordância são descritivas e calculadas sobre os votos já persistidos. Não há
avaliação humana nem inferência de similaridade semântica nesta etapa.

## Resultado esperado do experimento atual

Com os resultados auditados até a Fase 4, a consolidação deve ler 99 registros:
33 de Specify, 33 de Complicate e 33 de Diversify. Espera-se obter 79 aceitos e 20
rejeitados, correspondendo a 79,8% de aceitação global. O comando recalcula esses
números dos JSONL e deve ser tratado como a fonte definitiva.

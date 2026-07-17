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

Esta entrega implementa a Fase 1: schemas, parser XML, normalização, filtros,
persistência JSONL, amostras e testes.

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
ter variações ainda não representadas pelos aliases atuais. Provedores, agentes,
métricas semânticas e Colab serão adicionados nas fases seguintes.

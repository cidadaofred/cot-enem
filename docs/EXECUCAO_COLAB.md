# Execução remota da Fase 3 no Google Colab

Esta solução não usa Colab CLI, Ollama nem a porta `11434`. O Windows serve apenas
para enviar o código ao GitHub e o arquivo normalizado ao Google Drive. O Colab
clona o projeto, carrega o Qwen na GPU e executa a mesma CLI do projeto.

## Preparação única

1. Publique este repositório no GitHub, público ou privado.
2. No Google Drive, crie `cot-enem/data/processed/`.
3. Envie `data/processed/enem_normalized.jsonl` para essa pasta.
4. Abra `notebooks/fase3_colab.ipynb` no Colab.
5. Selecione **Ambiente de execução > Alterar tipo de ambiente > GPU**.
6. Na primeira célula de configuração, informe `REPOSITORY_URL` e `BRANCH`.

Para repositório privado, não coloque token no notebook. Use um repositório privado
somente se configurar uma credencial temporária/secreta no Colab; a alternativa mais
simples para o código acadêmico sem dados sensíveis é um repositório público.

## Execução segura

Na primeira execução, defina `LIMIT = 1`. Execute todas as células e valide o arquivo:

```text
Meu Drive/cot-enem/outputs/datasets/cot_enem_v1.jsonl
```

Depois altere para `LIMIT = None` e execute novamente. A pipeline calcula IDs
determinísticos, lê o JSONL já existente e pula registros concluídos. Se a sessão for
interrompida, reconecte e execute as células novamente; no máximo o item que estava em
processamento será refeito.

## Perfil utilizado

`configs/colab.yaml` seleciona:

- `Qwen/Qwen2.5-7B-Instruct`;
- Hugging Face Transformers, sem servidor HTTP intermediário;
- quantização NF4 em 4 bits;
- uma única cópia do modelo compartilhada entre geração e juízes;
- CUDA com BF16 quando suportado e FP16 como fallback;
- escrita incremental no Google Drive.

O cache do modelo fica em `Meu Drive/cot-enem/cache/huggingface`. A primeira sessão
faz o download; as seguintes reutilizam os arquivos. Não mova o repositório inteiro
para o Drive: muitos arquivos pequenos tornam instalação e imports mais lentos.

## Diagnóstico

O notebook executa antes do pipeline:

```bash
python -m cot_enem.cli verify --config configs/colab.yaml
```

Todos os itens devem aparecer como `[OK]`. Se `device` ou `bitsandbytes` falhar,
confirme que a sessão é GPU e reinstale as dependências executando novamente a célula
de instalação. Não tente iniciar `ollama serve`: ele não participa desta arquitetura.

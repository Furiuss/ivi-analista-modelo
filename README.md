# Ivi Analista Modelo

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) usando Ollama para embeddings e LLM, com ChromaDB como vector store.

## Pré-requisitos

1. **Python 3.8+**
2. **Ollama**
   - [Instale o Ollama](https://ollama.ai/download)
   - Após a instalação, baixe os modelos necessários:
   ```bash
   # Modelo para embeddings
   ollama pull nomic-embed-text
   
   # Modelo LLM
   ollama pull mistral
   ```

## Instalação

1. **Clone o repositório**
   ```bash
   git clone [URL_DO_REPOSITÓRIO]
   cd [NOME_DO_PROJETO]
   ```

2. **Crie e ative um ambiente virtual**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/MacOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -e .
   ```

## Estrutura do Projeto
```
projeto/
├── src/
│   ├── cache/            # Sistema de cache
│   ├── core/             # Interfaces e configurações core
│   ├── document_processing/  # Processamento de documentos
│   ├── embeddings/       # Providers de embedding
│   ├── llm/             # Providers de LLM
│   ├── utils/           # Utilitários
│   ├── vectorstores/    # Implementações de vector stores
│   └── scripts/         # Scripts de execução
├── data/                # Diretório para dados
├── chroma/             # Vector store persistence
├── config.yaml         # Configuração do sistema
└── setup.py           
```

## Configuração

1. **Crie o arquivo de configuração**
   
   Crie um arquivo `config.yaml` na raiz do projeto:
   ```yaml
   vector_store:
     provider: chroma
     persist_directory: chroma
     collection_name: default
     distance_metric: cosine

   embeddings:
     provider: ollama
     model_name: nomic-embed-text
     cache_size: 1000
     batch_size: 32

   llm:
     provider: ollama
     model_name: mistral
     temperature: 0.7
     max_tokens: 2000
     timeout: 180
     max_retries: 3
     retry_delay: 1.0

   cache:
     type: lru
     capacity: 1000
     default_ttl: 3600

   logging:
     level: INFO
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     file: logs/app.log
   ```

2. **Prepare seu diretório de dados**
   - Crie uma pasta `data` na raiz do projeto
   - Coloque seus arquivos JSON no formato esperado na pasta data

## Uso

1. **Treinamento do Sistema**
   ```bash
   # Treinar com dados novos (mantendo dados existentes)
   python src/scripts/train.py --config config.yaml --data data

   # Resetar e treinar do zero
   python src/scripts/train.py --config config.yaml --data data --reset
   ```

2. **Realizando Consultas**
   ```bash
   # Consulta básica
   python src/scripts/query.py "Sua pergunta aqui"

   # Consulta com idioma específico
   python src/scripts/query.py "Sua pergunta aqui" --language pt

   # Consulta com threshold de similaridade
   python src/scripts/query.py "Sua pergunta aqui" --similarity_threshold 0.7
   ```

## Formato dos Dados

O sistema espera arquivos JSON com a seguinte estrutura:
```json
{
    "Article": {
        "Title": "Título do Documento",
        "ContentHtml": "Conteúdo em HTML",
        "ContentText": "Conteúdo em texto puro",
        "Categories": [
            {
                "Name": "Categoria",
                "Id": "id_categoria"
            }
        ],
        "Id": "id_documento"
    }
}
```

## Troubleshooting

1. **Erro de conexão com Ollama**
   - Verifique se o serviço Ollama está rodando
   - Verifique se os modelos foram baixados corretamente

2. **Erros de memória com ChromaDB**
   - Ajuste o `batch_size` nas configurações
   - Reduza o número de documentos processados simultaneamente

3. **Respostas não relevantes**
   - Aumente o `similarity_threshold` nas consultas
   - Verifique a qualidade dos documentos de entrada
   - Ajuste o `temperature` do LLM nas configurações


## Contato

- Github: https://github.com/Furiuss
- Linkedin: https://www.linkedin.com/in/andre-messias-125b0720a/
# E-commerce Recommendation System

Sistema de recomendação de produtos para e-commerce utilizando TensorFlow.js e ChromaDB para armazenamento vetorial.

## Contexto do Projeto

Esse é um projeto de exemplo apresentado no módulo 03 da minha pós graduação.

O professor sugeriu como desafio experimentar um banco de dados vetorial como Pinecone, ChromaDB ou até mesmo a extensão vetorial do PostgreSQL. Com isso, você pode armazenar os vetores gerados do modelo (usuários e produtos) diretamente no banco, e realizar buscas por similaridade de forma escalável.

A ideia é substituir a comparação com todos os produtos por uma busca pelos "top N" produtos mais próximos do vetor do cliente. Assim, você otimiza performance e torna o sistema viável para uso com grandes volumes de dados. Na hora de realizar a predição, basta recuperar os produtos mais próximos do banco e aplicar a função predict da rede neural, gerando assim as recomendações personalizadas com base em dados reais.

## Arquitetura

O sistema segue uma arquitetura MVC (Model-View-Controller) com comunicação baseada em eventos:

- **Frontend**: Aplicação modular ES6+ com controllers, views e services
- **Backend**: API Express que atua como proxy para o ChromaDB
- **Database**: ChromaDB (banco de dados vetorial) rodando em container Docker
- **ML**: TensorFlow.js com Web Worker para treinamento do modelo de recomendação
- **Event System**: Sistema de eventos customizado para comunicação entre componentes

## Estrutura do Projeto

```
ecommerce-recomendations-with-tensorflow/
├── index.html              # Página principal
├── style.css               # Estilos da aplicação
├── server.js               # Backend Express (porta 3001)
├── package.json            # Dependências e scripts
├── chroma_setup.js         # Script de setup do ChromaDB
├── data/
│   ├── products.json       # Lista de produtos
│   └── users.json          # Lista de usuários
└── src/
    ├── index.js            # Ponto de entrada da aplicação
    ├── controller/         # Controllers (lógica de controle)
    │   ├── UserController.js
    │   ├── ProductController.js
    │   ├── ModelTrainingController.js
    │   ├── TFVisorController.js
    │   └── WorkerController.js
    ├── view/               # Views (apresentação)
    │   ├── UserView.js
    │   ├── ProductView.js
    │   ├── ModelTrainingView.js
    │   ├── TFVisorView.js
    │   ├── View.js
    │   └── templates/
    │       ├── product-card.html
    │       └── past-purchase.html
    ├── service/            # Services (lógica de negócio)
    │   ├── UserService.js
    │   ├── ProductService.js
    │   └── VectorService.js
    ├── events/             # Sistema de eventos
    │   ├── events.js
    │   └── constants.js
    └── workers/            # Web Workers
        └── modelTrainingWorker.js
```

## Tecnologias Utilizadas

- **Frontend**:
  - HTML5 + CSS3 (Bootstrap 5.3)
  - JavaScript ES6+ (módulos)
  - TensorFlow.js 4.22.0
  - tfjs-vis 1.5.1 (visualização)

- **Backend**:
  - Node.js + Express 4.18.2
  - ChromaDB client 3.3.1

- **Ferramentas**:
  - BrowserSync (hot reload)
  - Concurrently (execução paralela)
  - Docker (ChromaDB)
  - Nodemon (backend dev)

## Setup e Execução

### Pré-requisitos

- Node.js 18+ instalado
- Docker e Docker Compose instalados

### Instalação

1. Clone o repositório e instale as dependências:
```bash
npm install
```

### Execução

**Opção 1: Desenvolvimento completo (recomendado)**
```bash
npm run dev
```
Este comando inicia em paralelo:
- ChromaDB container (porta 8000)
- Backend Express (porta 3001)
- Frontend com BrowserSync (porta 3000)

**Opção 2: Manual**

1. Inicie o ChromaDB:
```bash
npm run chroma
```

2. Em outro terminal, inicie o backend:
```bash
npm run backend
```

3. Em outro terminal, inicie o frontend:
```bash
npm start
```

4. Acesse `http://localhost:3000` no navegador

### Testes
```bash
npm test
```

## Funcionalidades

### Atuais

- **Seleção de Usuário**: Escolha entre usuários pré-cadastrados
- **Histórico de Compras**: Visualização de compras passadas do usuário selecionado
- **Catálogo de Produtos**: Listagem de produtos disponíveis
- **Compra de Produtos**: Funcionalidade "Buy Now" para adicionar produtos ao carrinho
- **Treinamento de Modelo**: Botão para treinar modelo de recomendação com dados atuais
- **Recomendações**: Geração de recomendações personalizadas baseadas em:
  - Categoria do produto (peso: 0.4)
  - Cor do produto (peso: 0.3)
  - Preço (peso: 0.2)
  - Idade do usuário (peso: 0.1)

### Como Usar

1. Selecione um usuário no dropdown "User Profile"
2. Clique em "Train Model" para treinar o modelo com os dados atuais
3. Após o treinamento, clique em "Run Recommendation" para ver produtos recomendados
4. Compre produtos para atualizar o histórico e gerar novas recomendações

### Armazenamento Vetorial

- O sistema utiliza ChromaDB para armazenar embeddings de produtos
- Vetores são criados a partir de features: categoria, cor, preço e idade do usuário
- Recomendações são geradas via consulta por similaridade vetorial

## Scripts Disponíveis

| Script | Descrição |
|--------|-----------|
| `npm start` | Inicia frontend com BrowserSync na porta 3000 |
| `npm run backend` | Inicia backend Express na porta 3001 |
| `npm run chroma` | Inicia container ChromaDB na porta 8000 |
| `npm run chroma:stop` | Para o container ChromaDB |
| `npm run dev` | Inicia todos os serviços em paralelo |
| `npm test` | Executa testes automatizados |

## Variáveis de Ambiente

O backend suporta as seguintes variáveis (opcional):

- `CHROMA_URL`: URL do ChromaDB (padrão: http://localhost:8000)
- `BACKEND_URL`: URL do backend (padrão: http://localhost:3001)
- `BACKEND_PORT`: Porta do backend (padrão: 3001)

## Fluxo de Dados

1. Frontend carrega usuários e produtos dos arquivos JSON
2. Usuário seleciona perfil e realiza compras
3. Compras são rastreadas e atualizam o estado
4. Ao treinar o modelo:
   - Dados são processados no Web Worker
   - Vetores são criados e enviados ao ChromaDB via backend
5. Recomendações são geradas consultando vetores similares no ChromaDB

## Melhorias Futuras

- Interface de visualização do treinamento do modelo
- Histórico de recomendações
- Métricas de avaliação do modelo
- Suporte a múltiplas sessões de usuário
- API REST completa para integração externa
- Dashboard administrativo

## Licença

ISC

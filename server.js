import express from 'express';
import dotenv from 'dotenv';

/**
 * Carrega variáveis de ambiente a partir do arquivo `.env`, caso exista.
 *
 * Permite configurar localmente variáveis como:
 * - CHROMA_URL
 * - BACKEND_URL
 * - BACKEND_PORT
 *
 * Caso o arquivo não exista, o Node utilizará apenas `process.env`.
 *
 * @function
 */
dotenv.config();

/**
 * Aplicação Express que atua como proxy entre o frontend e o ChromaDB.
 *
 * Responsabilidades:
 * - habilitar CORS
 * - validar requisições
 * - criar/garantir coleções no Chroma
 * - encaminhar operações de vetor (upsert/query)
 *
 * @type {import('express').Express}
 */
const app = express();

/**
 * Middleware para:
 * - interpretar JSON no corpo da requisição
 * - limitar tamanho do payload
 * - capturar o corpo bruto para logs/debug
 */
app.use(express.json({
  limit: '1mb',

  /**
   * Intercepta o buffer bruto recebido antes do parse do JSON.
   *
   * @param {import('express').Request} req
   * @param {import('express').Response} res
   * @param {Buffer} buf
   */
  verify: (req, res, buf) => {
    req.rawBody = buf.toString();
  }
}));

/**
 * Middleware de logging simples.
 *
 * Registra:
 * - método HTTP
 * - URL
 * - corpo bruto recebido
 *
 * Útil para diagnóstico de requisições enviadas pelo frontend.
 */
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url} raw body=`, JSON.stringify(req.rawBody));
  next();
});

/**
 * Middleware para capturar erros de parsing JSON.
 *
 * Quando o cliente envia JSON inválido, o Express lança
 * um `SyntaxError`. Este middleware converte o erro em
 * resposta HTTP 400 com mensagem estruturada.
 *
 * @param {Error} err
 * @param {import('express').Request} req
 * @param {import('express').Response} res
 * @param {import('express').NextFunction} next
 */
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {

    console.error('Erro ao interpretar JSON:', err.message);
    console.error('Corpo recebido:', req.rawBody);

    return res.status(400).json({
      error: 'JSON inválido',
      details: err.message
    });
  }

  next(err);
});

/**
 * Middleware CORS básico.
 *
 * Permite que aplicações frontend (ex: localhost:3000)
 * façam requisições para este backend (ex: localhost:3001).
 */
app.use((req, res, next) => {

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});

/**
 * Constrói e normaliza a URL do ChromaDB.
 *
 * Aceita URLs com ou sem `/api/v2` e garante que
 * o endpoint final esteja correto.
 *
 * Exemplos válidos:
 * - http://localhost:8000
 * - http://localhost:8000/api/v2
 *
 * @returns {string} URL normalizada do servidor Chroma
 */
function buildChromaUrl() {

  let u = process.env.CHROMA_URL || 'http://localhost:8000';

  // remove barra final
  u = u.replace(/\/$/, '');

  // garante presença do endpoint da API
  if (!u.includes('/api/v2')) {
    u += '/api/v2';
  }

  return u;
}

/**
 * URL final utilizada para comunicação com o ChromaDB.
 *
 * @constant {string}
 */
const CHROMA_URL = buildChromaUrl();

if (!process.env.CHROMA_URL) {
  console.warn('CHROMA_URL não definido - usando padrão', CHROMA_URL);
}

/**
 * URL pública opcional do backend.
 *
 * Pode ser usada apenas para logging ou configuração
 * de clientes externos.
 *
 * @type {string|undefined}
 */
const BACKEND_URL = process.env.BACKEND_URL;

/**
 * Configurações de tenant e database do ChromaDB.
 *
 * Caso não sejam definidos nas variáveis de ambiente,
 * utiliza valores padrão.
 *
 * @constant {string}
 */
const TENANT = process.env.CHROMA_TENANT || 'default';

/**
 * Nome do database no Chroma.
 *
 * @constant {string}
 */
const DATABASE = process.env.CHROMA_DB || 'default';

/**
 * Nome da coleção onde os embeddings de produtos são armazenados.
 *
 * @constant {string}
 */
const COLLECTION_NAME = 'products';

/**
 * Cache do ID da coleção já criada ou recuperada.
 *
 * Evita chamadas repetidas ao Chroma para buscar a mesma coleção.
 *
 * @type {string|null}
 */
let _cachedCollectionId = null;

/**
 * Garante que a coleção exista no ChromaDB.
 *
 * Também tenta garantir a existência do:
 * - tenant
 * - database
 *
 * Caso a coleção já exista, reutiliza o ID em cache.
 *
 * @async
 * @param {number} dimension Dimensão dos embeddings
 * @returns {Promise<string>} ID da coleção
 */
async function ensureCollection(dimension) {

  console.log('ensureCollection dimension', dimension, 'tenant', TENANT, 'db', DATABASE);

  // tenta criar tenant
  try {
    await fetch(`${CHROMA_URL}/tenants`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: TENANT })
    });
  } catch (e) {
    console.warn('falha ao criar tenant (pode já existir):', e.message);
  }

  // tenta criar database
  try {
    await fetch(`${CHROMA_URL}/tenants/${TENANT}/databases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: DATABASE })
    });
  } catch (e) {
    console.warn('falha ao criar database (pode já existir):', e.message);
  }

  if (_cachedCollectionId) return _cachedCollectionId;

  /**
   * Cria ou recupera a coleção usando `get_or_create`.
   */
  const collRes = await fetch(
    `${CHROMA_URL}/tenants/${TENANT}/databases/${DATABASE}/collections`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: COLLECTION_NAME,
        get_or_create: true
      })
    }
  );

  const collText = await collRes.text();

  let collBody = null;

  if (collText) {
    try {
      collBody = JSON.parse(collText);
    } catch (err) {

      console.error('falha ao interpretar resposta da coleção:', err.message);
      console.error('resposta bruta:', collText);

      throw new Error('JSON inválido na criação da coleção');
    }
  } else {
    console.warn('endpoint de coleção retornou corpo vazio (status', collRes.status + ')');
  }

  console.log('resposta criação/obtenção coleção', collRes.status, collBody);

  if (collBody && collBody.id) {
    _cachedCollectionId = collBody.id;
    return _cachedCollectionId;
  }

  throw new Error('não foi possível garantir a coleção; id não retornado');
}

/**
 * Endpoint para inserir ou atualizar embeddings de produtos.
 *
 * Espera receber:
 *
 * {
 *   vectors: [{ id, values }]
 * }
 *
 * @route POST /vector/upsert
 */
app.post('/vector/upsert', async (req, res) => {

  try {

    const { vectors } = req.body;

    console.log('upsert vectors', vectors && vectors.length);

    if (!Array.isArray(vectors)) {
      return res.status(400).json({ error: 'array vectors obrigatório' });
    }

    if (vectors.length) {
      await ensureCollection(vectors[0].values.length);
    }

    /**
     * Lista de IDs dos vetores.
     * @type {string[]}
     */
    const ids = vectors.map(v => v.id);

    /**
     * Converte embeddings para arrays simples caso
     * tenham sido serializados como objetos.
     *
     * @type {number[][]}
     */
    const embeddings = vectors.map(v => {

      if (Array.isArray(v.values)) return v.values;

      if (v.values && typeof v.values === 'object') {
        return Object.values(v.values);
      }

      return v.values;
    });

    const payload = { ids, embeddings };

    const collectionId = await ensureCollection(vectors[0].values.length);

    console.log(
      'proxy upsert para Chroma',
      CHROMA_URL,
      TENANT,
      DATABASE,
      collectionId,
      payload
    );

    const upstream = await fetch(
      `${CHROMA_URL}/tenants/${TENANT}/databases/${DATABASE}/collections/${collectionId}/upsert`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }
    );

    const raw = await upstream.text();

    let body;

    if (raw) {
      try {
        body = JSON.parse(raw);
      } catch (parseErr) {

        console.error('falha ao interpretar resposta do Chroma:', parseErr.message);
        console.error('resposta bruta:', raw);

        return res.status(502).json({
          error: 'Resposta inválida do Chroma',
          raw
        });
      }
    } else {
      body = {};
    }

    console.log('resposta upsert chroma', body);

    return res.status(upstream.status).json(body);

  } catch (err) {

    console.error('erro upsert', err);

    return res.status(500).json({
      error: err.message
    });
  }
});

/**
 * Endpoint para consulta de similaridade vetorial.
 *
 * Espera receber:
 *
 * {
 *   query: number[],
 *   topK?: number
 * }
 *
 * @route POST /vector/query
 */
app.post('/vector/query', async (req, res) => {

  try {

    const { query, topK = 50 } = req.body;

    if (!Array.isArray(query)) {
      return res.status(400).json({ error: 'query deve ser um array' });
    }

    if (!query.length) {
      console.warn('consulta vazia recebida');
    }

    const collectionId = await ensureCollection(query.length);

    const payload = {
      query_embeddings: [query],
      n_results: topK
    };

    console.log(
      'proxy query para Chroma',
      CHROMA_URL,
      TENANT,
      DATABASE,
      collectionId,
      payload
    );

    const upstream = await fetch(
      `${CHROMA_URL}/tenants/${TENANT}/databases/${DATABASE}/collections/${collectionId}/query`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }
    );

    const body = await upstream.json();

    console.log('resposta chroma', body);

    return res.status(upstream.status).json(body);

  } catch (err) {

    console.error('erro query', err);

    return res.status(500).json({
      error: err.message
    });
  }
});

/**
 * Rota proxy genérica para qualquer endpoint `/api`.
 *
 * Encaminha a requisição diretamente para o Chroma
 * preservando:
 * - método
 * - cabeçalhos
 * - corpo
 *
 * @route ALL /api/*
 */
app.use('/api', async (req, res) => {

  try {

    const upstreamUrl = CHROMA_URL + req.originalUrl;

    /**
     * Opções da requisição encaminhada ao upstream.
     */
    const options = {
      method: req.method,
      headers: { ...req.headers },
    };

    delete options.headers.host;
    delete options.headers['content-length'];

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      options.body = JSON.stringify(req.body);
      options.headers['Content-Type'] = 'application/json';
    }

    const upstreamRes = await fetch(upstreamUrl, options);

    const body = await upstreamRes.text();

    res.status(upstreamRes.status);

    upstreamRes.headers.forEach((value, key) => {

      if (![
        'transfer-encoding',
        'connection',
        'keep-alive',
        'content-length'
      ].includes(key.toLowerCase())) {

        res.setHeader(key, value);
      }
    });

    res.send(body);

  } catch (err) {

    console.error('erro proxy', err);

    res.status(500).json({
      error: err.message
    });
  }
});

/**
 * Porta utilizada pelo servidor backend.
 *
 * Ordem de prioridade:
 * 1. BACKEND_PORT
 * 2. PORT (compatibilidade com plataformas cloud)
 * 3. 3001
 *
 * @constant {number|string}
 */
const PORT = process.env.BACKEND_PORT || process.env.PORT || 3001;

/**
 * Inicializa o servidor HTTP Express.
 */
app.listen(PORT, () => {

  console.log(
    `backend proxy ouvindo em ${
      BACKEND_URL
        ? BACKEND_URL.replace(/\/api\/v2$/, '')
        : `http://localhost:${PORT}`
    }`
  );

  console.log(`proxy encaminhando requisições para Chroma em ${CHROMA_URL}`);

});
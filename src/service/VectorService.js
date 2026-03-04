/**
 * Endereço base do proxy do backend.
 *
 * `index.js` ou um `<script>` inline pode definir `window.BACKEND_URL`
 * antes deste módulo ser carregado.
 *
 * Caso não exista, utiliza `http://localhost:3001` como fallback
 * para permitir que o módulo funcione isoladamente durante
 * desenvolvimento ou testes.
 *
 * @constant {string}
 */
const BACKEND_URL =
    (typeof globalThis !== 'undefined' && globalThis.BACKEND_URL) ||
    'http://localhost:3001';

/**
 * Envia uma requisição HTTP POST para o backend e retorna a resposta em JSON.
 *
 * Função utilitária interna usada pelos métodos deste serviço para
 * centralizar chamadas POST à API.
 *
 * @async
 * @param {string} path Caminho do endpoint da API (ex: `/vector/upsert`)
 * @param {object} body Corpo da requisição que será serializado em JSON
 *
 * @returns {Promise<{status:number, body:object}>}
 * Objeto contendo:
 * - `status` código HTTP da resposta
 * - `body` corpo da resposta já convertido para JSON
 */
async function post(path, body) {
    const res = await fetch(`${BACKEND_URL}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    const json = await res.json();

    return {
        status: res.status,
        body: json
    };
}

/**
 * Insere ou atualiza (upsert) um lote de vetores de produtos
 * na coleção vetorial (ex: Chroma).
 *
 * Cada vetor deve conter:
 * - `id` identificador único
 * - `values` vetor numérico (embedding)
 *
 * @async
 * @param {Array<{id:string, values:number[]}>} vectors
 * Lista de vetores de produtos para inserção ou atualização.
 *
 * @returns {Promise<object>|undefined}
 * Resultado retornado pelo backend. Caso `vectors` esteja vazio,
 * nenhuma requisição é realizada.
 */
export async function upsertProductVectors(vectors) {
    // evita requisição desnecessária se não houver vetores
    if (!vectors.length) return;

    const result = await post('/vector/upsert', { vectors });

    console.log('upsertProductVectors result', result);

    return result;
}

/**
 * Consulta os vetores mais próximos ao vetor de busca informado.
 *
 * Realiza uma busca de similaridade no banco vetorial e retorna
 * os `topK` resultados mais próximos.
 *
 * @async
 * @param {number[] | TypedArray | object} query
 * Vetor de consulta (embedding).
 *
 * Pode ser:
 * - `Array<number>`
 * - `TypedArray` (ex: Float32Array)
 * - objeto com chaves numéricas
 *
 * @param {number} [topK=50]
 * Quantidade máxima de resultados a retornar.
 *
 * @returns {Promise<{ids:string[], distances:number[], raw:object}>}
 * Objeto contendo:
 * - `ids` IDs dos vetores encontrados
 * - `distances` distância ou similaridade de cada vetor
 * - `raw` resposta completa retornada pelo backend
 */
export async function queryNearest(query, topK = 50) {

    /**
     * Float32Array e outros TypedArrays são convertidos para objetos
     * com chaves numéricas quando serializados com JSON.stringify.
     *
     * Isso faz o backend rejeitar a requisição com erro
     * "query array required".
     *
     * Aqui garantimos que sempre seja enviado um Array simples.
     *
     * @type {number[]}
     */
    let q = query;

    if (
        ArrayBuffer.isView(query) ||
        (query && typeof query === 'object' && !Array.isArray(query))
    ) {
        // converte TypedArray ou objeto numérico para Array
        q = Array.from(query);
    }

    const res = await fetch(`${BACKEND_URL}/vector/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: q,
            topK
        })
    });

    const data = await res.json();

    /**
     * A resposta do mecanismo vetorial normalmente vem como
     * matriz bidimensional:
     *
     * ids: [[...]]
     * distances: [[...]]
     *
     * Aqui simplificamos retornando apenas o primeiro conjunto
     * de resultados.
     *
     * @type {string[]}
     */
    const ids = (data.ids && data.ids[0]) || [];

    /**
     * Distâncias (ou similaridades) correspondentes aos IDs retornados.
     *
     * @type {number[]}
     */
    const distances = (data.distances && data.distances[0]) || [];

    /**
     * Estrutura final retornada ao chamador.
     *
     * @type {{ids:string[], distances:number[], raw:object}}
     */
    const result = {
        ids,
        distances,
        raw: data
    };

    console.log('queryNearest result', result);

    return result;
}
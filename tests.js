/**
 * Script consolidado de testes básicos (sanity checks) para o backend.
 *
 * Executa três verificações principais:
 * 1. Serialização de TypedArrays (ex: Float32Array)
 * 2. Inserção de vetores (upsert) no backend
 * 3. Consulta de similaridade vetorial
 *
 * Pode ser executado diretamente com:
 *
 *   node tests.js
 *
 * ou via script do npm:
 *
 *   npm run test
 */

/**
 * URL do backend proxy.
 *
 * Deve apontar para o servidor Express responsável por encaminhar
 * requisições ao ChromaDB.
 *
 * Este valor deve ser o mesmo utilizado no `VectorService`.
 *
 * @constant {string}
 */
const BACKEND_URL = 'http://localhost:3001';

/**
 * Testa como um `Float32Array` é serializado ao usar `JSON.stringify`.
 *
 * Objetivo:
 * demonstrar que TypedArrays não são serializados como arrays
 * normais, mas como objetos com chaves numéricas.
 *
 * Exemplo esperado:
 * {
 *   "query": { "0":1, "1":2, "2":3 }
 * }
 *
 * Isso é importante porque o backend espera `number[]`
 * e não objetos, sendo necessário converter com `Array.from`.
 *
 * @async
 * @returns {Promise<void>}
 */
async function testSerialization() {

    const a = new Float32Array([1, 2, 3]);

    console.log(
        'serialização de Float32Array:',
        JSON.stringify({ query: a })
    );
}

/**
 * Testa o endpoint de inserção/atualização de vetores (`/vector/upsert`).
 *
 * Fluxo:
 * 1. Gera um vetor aleatório de dimensão 14
 * 2. Envia ao backend com um ID fixo (`test-vector`)
 * 3. O backend repassa para o ChromaDB
 *
 * Objetivo:
 * validar se:
 * - o backend está respondendo
 * - a coleção é criada automaticamente
 * - o Chroma aceita o vetor
 *
 * @async
 * @returns {Promise<void>}
 */
async function testUpsert() {

    console.log('\n=== teste de upsert ===');

    try {

        /**
         * Gera um vetor de embedding aleatório.
         *
         * @type {number[]}
         */
        const vector = Array.from(
            { length: 14 },
            () => Math.random()
        );

        const res = await fetch(`${BACKEND_URL}/vector/upsert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                vectors: [
                    {
                        id: 'test-vector',
                        values: vector
                    }
                ]
            })
        });

        /**
         * Usa `.text()` em vez de `.json()` para evitar erro caso
         * o backend retorne resposta não JSON.
         */
        const json = await res.text();

        console.log('status upsert:', res.status);
        console.log('resposta:', json);

    } catch (err) {

        console.error('erro no teste de upsert', err);
    }
}

/**
 * Testa o endpoint de consulta vetorial (`/vector/query`).
 *
 * Fluxo:
 * 1. Gera um vetor aleatório
 * 2. Envia ao backend
 * 3. O backend consulta o ChromaDB
 * 4. Retorna os `topK` vetores mais próximos
 *
 * O resultado normalmente contém:
 * - `ids`
 * - `distances`
 * - metadados do mecanismo vetorial
 *
 * @async
 * @returns {Promise<void>}
 */
async function testQuery() {

    console.log('\n=== teste de consulta vetorial ===');

    try {

        /**
         * Vetor de consulta aleatório.
         *
         * Deve ter a mesma dimensão dos vetores inseridos.
         *
         * @type {number[]}
         */
        const queryVec = Array.from(
            { length: 14 },
            () => Math.random()
        );

        const res = await fetch(`${BACKEND_URL}/vector/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                query: queryVec,
                topK: 5
            })
        });

        /**
         * Resultado retornado pelo backend/Chroma.
         *
         * Estrutura típica:
         * {
         *   ids: [[]],
         *   distances: [[]]
         * }
         */
        const json = await res.json();

        console.log('resultado da consulta:', json);

    } catch (err) {

        console.error('erro no teste de consulta', err);
    }
}

/**
 * Função principal que executa todos os testes em sequência.
 *
 * Ordem:
 * 1. teste de serialização
 * 2. teste de upsert
 * 3. teste de query
 *
 * Executa dentro de uma IIFE async para permitir uso de `await`
 * no nível superior do script.
 */
(async () => {

    await testSerialization();

    await testUpsert();

    await testQuery();

})();
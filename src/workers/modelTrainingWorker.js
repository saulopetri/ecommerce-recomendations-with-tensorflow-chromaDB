import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
// Helper para banco de dados vetorial (ChromaDB)
import * as vectorService from '../service/VectorService.js';

let _globalCtx = {};
let _model = null;

/**
 * @typedef {Object} Weights
 * @property {number} category - Peso para a categoria do produto.
 * @property {number} color - Peso para a cor do produto.
 * @property {number} price - Peso para o preço do produto.
 * @property {number} age - Peso para a idade do usuário.
 */
/** @type {Weights} */
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

/**
 * Normaliza um valor contínuo para o intervalo [0, 1].
 * Isso é útil para garantir que todas as features tenham a mesma escala,
 * evitando que uma feature domine o treinamento.
 * A fórmula utilizada é: (valor - mínimo) / (máximo - mínimo).
 * Exemplo: se o preço for 129.99, o preço mínimo for 39.99 e o máximo for 199.99,
 * o resultado normalizado será aproximadamente 0.56.
 *
 * @param {number} value - O valor a ser normalizado.
 * @param {number} min - O valor mínimo do intervalo.
 * @param {number} max - O valor máximo do intervalo.
 * @returns {number} O valor normalizado entre 0 e 1.
 */
const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

/**
 * Cria o contexto global para o treinamento do modelo.
 * Este contexto inclui informações sobre produtos e usuários,
 * como faixas de preço e idade, índices de cores e categorias,
 * e a idade média normalizada dos compradores por produto.
 *
 * @param {Array<Object>} products - Lista de produtos disponíveis.
 * @param {Array<Object>} users - Lista de usuários com seus históricos de compras.
 * @returns {Object} O contexto global com informações processadas.
 */
function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    );
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );

    // Calcula a média de idade dos compradores por produto para personalização.
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name]
                ? ageSums[product.name] / ageCounts[product.name]
                : midAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // Dimensões totais: preço + idade + categorias (one-hot) + cores (one-hot)
        dimentions: 2 + categories.length + colors.length
    };
}

/**
 * Cria um vetor one-hot ponderado para uma determinada feature.
 * Isso é usado para codificar features categóricas como cor e categoria.
 *
 * @param {number} index - O índice da categoria ou cor.
 * @param {number} length - O número total de categorias ou cores.
 * @param {number} weight - O peso a ser aplicado ao vetor one-hot.
 * @returns {tf.Tensor1D} O vetor one-hot ponderado.
 */
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

/**
 * Codifica um produto em um vetor numérico para ser usado pelo modelo.
 * Normaliza o preço e a idade, e aplica codificação one-hot às features
 * categóricas (categoria e cor), aplicando os pesos definidos.
 *
 * @param {Object} product - O produto a ser codificado.
 * @param {Object} context - O contexto global com informações de normalização e índices.
 * @returns {tf.Tensor1D} O vetor codificado do produto.
 */
function encodeProduct(product, context) {
    // Normaliza os dados para o intervalo [0, 1] e aplica o peso na recomendação.
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ]);

    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    // Concatena todos os vetores para formar o vetor final do produto.
    return tf.concat1d([price, age, category, color]);
}

/**
 * Codifica um usuário em um vetor numérico.
 * Se o usuário tiver compras, o vetor é a média dos vetores dos produtos comprados.
 * Caso contrário, usa a idade normalizada e zera as outras features.
 *
 * @param {Object} user - O usuário a ser codificado.
 * @param {Object} context - O contexto global com informações de normalização e índices.
 * @returns {tf.Tensor1D} O vetor codificado do usuário.
 */
function encodeUser(user, context) {
    if (user.purchases.length) {
        // Calcula a média dos vetores dos produtos comprados pelo usuário.
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([1, context.dimentions]);
    }

    // Se o usuário não tem compras, retorna um vetor padrão.
    return tf.concat1d([
        tf.zeros([1]), // Preço é ignorado.
        tf.tensor1d([normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age]),
        tf.zeros([context.numCategories]), // Categoria é ignorada.
        tf.zeros([context.numColors]), // Cor é ignorada.
    ]).reshape([1, context.dimentions]);
}

/**
 * Cria os dados de treinamento (features e labels) para o modelo.
 * Combina os vetores de usuários e produtos para criar pares de input/label.
 *
 * @param {Object} context - O contexto global com produtos e usuários processados.
 * @returns {Object} Um objeto contendo os tensores de features (xs), labels (ys) e a dimensão da entrada.
 */
function createTrainingData(context) {
    const inputs = [];
    const labels = [];
    context.users
        .filter(u => u.purchases.length) // Considera apenas usuários com histórico de compras.
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync();
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync();

                // O label é 1 se o produto foi comprado pelo usuário, 0 caso contrário.
                const label = user.purchases.some(
                    purchase => purchase.name === product.name ? 1 : 0
                );
                // Combina o vetor do usuário e do produto para formar a entrada.
                inputs.push([...userVector, ...productVector]);
                labels.push(label);
            });
        });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        // A dimensão da entrada é o dobro da dimensão original (usuário + produto).
        inputDimention: context.dimentions * 2
    };
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
/**
 * Configura a rede neural e treina o modelo com os dados fornecidos.
 * A rede possui camadas densas com ativação ReLU e uma camada de saída sigmoide
 * para predição de probabilidade.
 *
 * @param {Object} trainData - Os dados de treinamento, incluindo features (xs) e labels (ys).
 * @returns {Promise<tf.Sequential>} O modelo treinado.
 */
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential();
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    );
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    );

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    );
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    );

    // Compila o modelo com otimizador Adam, perda binária e métrica de acurácia.
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Treina o modelo com os dados, definindo épocas, tamanho do batch e shuffle.
    // Usa callbacks para postar logs de progresso durante o treinamento.
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

/**
 * Inicia o processo de treinamento do modelo.
 * Carrega os dados de produtos, cria o contexto, treina a rede neural e salva os vetores de produtos.
 *
 * @param {Object} data - Objeto contendo os dados necessários para o treinamento.
 * @param {Array<Object>} data.users - Lista de usuários com histórico de compras.
 */
async function trainModel({ users }) {

    console.log('Treinando modelo com usuários:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    // Carrega os dados dos produtos de um arquivo JSON.
    const products = await (await fetch('/data/products.json')).json();

    // Cria o contexto e os vetores dos produtos.
    const context = makeContext(products, users);
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        };
    });

    // Salva os vetores dos produtos no ChromaDB para consultas futuras.
    // `dataSync()` retorna um Float32Array; a serialização JSON o transforma em um objeto
    // com chaves numéricas, o que causa erro no backend. Converte para um Array real
    // para que o proxy possa encaminhar os dados corretamente.
    await vectorService.upsertProductVectors(
        context.productVectors.map(p => ({ id: p.name, values: Array.from(p.vector) }))
    );

    _globalCtx = context;

    // Cria os dados de treinamento e treina o modelo.
    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    // Notifica o progresso e a conclusão do treinamento.
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

/**
 * Gera recomendações de produtos para um determinado usuário.
 * Utiliza o modelo treinado e o banco de dados vetorial para encontrar os produtos mais relevantes.
 *
 * @param {Object} data - Objeto contendo os dados para recomendação.
 * @param {Object} data.user - O usuário para o qual as recomendações serão geradas.
 */
async function recommend({ user }) {
    if (!_model) return; // Retorna se o modelo ainda não foi treinado.
    const context = _globalCtx;

    // 1. Calcula o vetor do usuário (sem considerar o preço).
    const userVector = encodeUser(user, context).dataSync();

    // 2️⃣ use o banco vetorial (ChromaDB) para obter os "top N" produtos
    // mais próximos do vetor do usuário.  Dessa forma diminuímos drasticamente
    // a quantidade de pares que precisamos passar pela rede neural.
    const result = await vectorService.queryNearest(userVector, 200);
    console.log('vectorService.queryNearest result', result, 'userVector length', userVector.length);
    const { ids } = result;

    // manter apenas os candidatos retornados pelo Chroma (e na mesma ordem)
    const candidates = context.productVectors
        .filter(p => ids.includes(p.name))
        .sort((a, b) => ids.indexOf(a.name) - ids.indexOf(b.name));

    // 3️⃣ preparar tensores somente para os candidatos
    const inputs = candidates.map(({ vector }) => [...userVector, ...vector]);
    const inputTensor = tf.tensor2d(inputs);

    // 4️⃣ predição rápida usando apenas _model.predict()
    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    const recommendations = candidates.map((item, index) => ({
        ...item.meta,
        name: item.name,
        score: scores[index]
    }));

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = async e => {
    const { action, ...data } = e.data;
    if (handlers[action]) await handlers[action](data);
};



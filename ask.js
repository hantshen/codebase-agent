import fs from 'fs';
import readline from 'readline';
import axios from 'axios';
import { pipeline } from '@xenova/transformers';
import 'dotenv/config';
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// 🧠 Cosine similarity
function cosineSimilarity(vecA, vecB) {
    const dot = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    const normA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    return dot / (normA * normB);
}

// 🔍 Get top N matches
async function searchEmbeddings(query, topK = 3) {
    const raw = JSON.parse(fs.readFileSync('./embeddings.json', 'utf8'));
    const embeddings = raw.map(item => ({
        ...item,
        embedding: Array.isArray(item.embedding) ? item.embedding : Object.values(item.embedding),
    }));

    const queryVec = (await extractor(query, { pooling: 'mean', normalize: true })).data;

    return embeddings
        .map(item => ({
            ...item,
            score: cosineSimilarity(queryVec, item.embedding),
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

// 💬 Ask the agent
async function askAgent(question) {
    const matches = await searchEmbeddings(question);
    const context = matches.map(m => `File: ${m.filePath}\n\n${m.content}`).join('\n\n---\n\n');

    const prompt = `You are a helpful AI code assistant. Use the code snippets below to answer the user's question.\n\n${context}\n\n---\n\nUser question: "${question}"\n\nAnswer:`;

    const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
        model: 'deepseek/deepseek-r1-0528-qwen3-8b:free', // openai/gpt-3.5-turbo
        messages: [
            { role: 'system', content: 'You are a senior developer assistant. Answer clearly and concisely.' },
            { role: 'user', content: prompt }
        ],
    }, {
        headers: {
            'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
            'HTTP-Referer': 'http://localhost',
            'X-Title': 'CodeAgent',
        }
    });

    console.log('\n🤖 Answer:\n', response.data.choices[0].message.content);
}

// 🧪 Ask via terminal
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

rl.question('🧠 Ask your code: ', (question) => {
    askAgent(question).finally(() => rl.close());
});

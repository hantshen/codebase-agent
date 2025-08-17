import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { globby } from 'globby';
import { pipeline } from '@xenova/transformers';
import 'dotenv/config';

const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
if (!extractor) {
    console.error('❌ Failed to load model');
    process.exit(1);
}
console.log('✅ Model loaded');

const githubRepos = [
    '<ORG_NAME>/<REPO_NAME>',
    '<GITHUB_USER_NAME>/<REPO_NAME>',
];

const token = process.env.GITHUB_TOKEN;
if (!token) {
    console.error('❌ GITHUB_TOKEN missing in .env file.');
    process.exit(1);
}

const repoFolder = './repos';
if (!fs.existsSync(repoFolder)) fs.mkdirSync(repoFolder);

function ensureRepoCloned(repo) {
    const name = repo.split('/').pop();
    const repoPath = path.resolve(repoFolder, name);

    if (fs.existsSync(repoPath)) {
        console.log(`📁 Already cloned: ${name}`);
        return repoPath;
    }

    const url = `https://${token}@github.com/${repo}.git`;
    console.log(`⬇️ Cloning ${repo}...`);
    execSync(`git clone ${url} ${repoPath}`, { stdio: 'inherit' });

    return repoPath;
}

async function getCodeFiles(repoPath) {
    // Convert Windows-style path to POSIX for globby
    const posixPath = repoPath.replace(/\\/g, '/');

    const patterns = [
        `${posixPath}/**/*.js`,
        `${posixPath}/**/*.ts`,
        `${posixPath}/**/*.tsx`,
        `${posixPath}/**/*.jsx`,
        `${posixPath}/**/*.py`,
        `${posixPath}/**/*.mjs`,
        `!${posixPath}/node_modules/**`,
        `!${posixPath}/dist/**`,
        `!${posixPath}/build/**`,
        `!${posixPath}/.next/**`,
        `!${posixPath}/.venv/**`,
        `!${posixPath}/__pycache__/**`,
        `!${posixPath}/**/*.min.js`,
    ];

    const files = await globby(patterns);

    console.log(`📃 Matched ${files.length} files in ${repoPath}`);
    files.forEach(f => console.log('   ↪️ ', f));

    return files.map(filePath => {
        const content = fs.readFileSync(filePath, 'utf-8').trim();
        return { filePath, content };
    }).filter(file => file.content.length > 0);
}

const results = [];

for (const repo of githubRepos) {
    const localPath = ensureRepoCloned(repo);
    const codeFiles = await getCodeFiles(localPath);
    console.log(`📂 ${repo} - ${codeFiles.length} code files found`);

    for (const file of codeFiles) {
        try {
            const embedding = await extractor(file.content, {
                pooling: 'mean',
                normalize: true,
            });

            results.push({
                repo,
                filePath: file.filePath,
                content: file.content,
                embedding: embedding.data,
            });

            console.log(`✅ Embedded: ${file.filePath}`);
        } catch (err) {
            console.error(`❌ Error embedding ${file.filePath}:`, err.message);
        }
    }
}

if (results.length > 0) {
    fs.writeFileSync('./embeddings.json', JSON.stringify(results, null, 2));
    console.log(`💾 Saved ${results.length} embeddings to embeddings.json`);
} else {
    console.warn('⚠️ No embeddings generated.');
}

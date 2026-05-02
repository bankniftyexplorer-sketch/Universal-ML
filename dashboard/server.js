import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');

const app = express();
app.use(cors());

// Simple in-memory queue to prevent OOM
let isProcessing = false;
const jobQueue = [];

const processQueue = async () => {
    if (isProcessing || jobQueue.length === 0) return;
    isProcessing = true;
    const job = jobQueue.shift();
    try {
        await job();
    } catch (err) {
        console.error('Job error:', err);
    }
    isProcessing = false;
    processQueue();
};

// 1. Get List of Symbols
app.get('/api/symbols', (req, res) => {
    // Look for directories in the project root that might be symbols (uppercase, no dots)
    const dirs = fs.readdirSync(PROJECT_ROOT, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory() && /^[A-Z0-9]+$/.test(dirent.name))
        .map(dirent => dirent.name);
    res.json({ symbols: dirs });
});

// 2. Get latest report data for a symbol
app.get('/api/report/:symbol', (req, res) => {
    const symbol = req.params.symbol.toUpperCase();
    const symbolDir = path.join(PROJECT_ROOT, symbol);

    const jsonPath = path.join(symbolDir, `${symbol.toLowerCase()}_VOL_live_forecast.json`);
    const pngPath = path.join(symbolDir, `${symbol.toLowerCase()}_VOL_dashboard.png`);
    const backtestPath = path.join(symbolDir, `${symbol.toLowerCase()}_VOL_report.png`);

    let data = null;
    let pngExists = fs.existsSync(pngPath);
    let backtestExists = fs.existsSync(backtestPath);
    let lastUpdated = null;

    if (fs.existsSync(jsonPath)) {
        try {
            data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            const stat = fs.statSync(jsonPath);
            lastUpdated = stat.mtime;
        } catch (e) {
            console.error(e);
        }
    }

    res.json({
        symbol,
        data,
        pngExists,
        backtestExists,
        lastUpdated
    });
});

// 3. Serve the PNG image
app.get('/api/image/:symbol', (req, res) => {
    const symbol = req.params.symbol.toUpperCase();
    const type = req.query.type || 'dashboard';

    let filename = `${symbol.toLowerCase()}_VOL_dashboard.png`;
    if (type === 'backtest') {
        filename = `${symbol.toLowerCase()}_VOL_report.png`;
    }

    const pngPath = path.join(PROJECT_ROOT, symbol, filename);
    if (fs.existsSync(pngPath)) {
        res.sendFile(pngPath);
    } else {
        res.status(404).send('Image not found');
    }
});

// 4. SSE Endpoint to run the inference script and stream logs
app.get('/api/run/:symbol', (req, res) => {
    const symbol = req.params.symbol.toUpperCase();

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Function to run a command and return a promise
    const runCommand = (command, args) => {
        return new Promise((resolve) => {
            const cmd = spawn(command, args, {
                cwd: PROJECT_ROOT,
                env: { ...process.env, PYTHONUNBUFFERED: '1' }
            });

            const handleData = (data) => {
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    const text = line.replace(/\r/g, ''); // keep leading spaces, just remove carriage returns
                    if (text.trim().length > 0) { // only send if it's not completely empty
                        res.write(`data: ${JSON.stringify({ type: 'log', message: text })}\n\n`);
                    }
                }
            };

            cmd.stdout.on('data', handleData);
            cmd.stderr.on('data', handleData);

            cmd.on('close', (code) => resolve(code));
            req.on('close', () => cmd.kill());
        });
    };

    const executeChain = async () => {
        if (req.destroyed) return; // connection closed before job started

        res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'vault_sync', status: 'running' })}\n\n`);
        res.write(`data: ${JSON.stringify({ type: 'log', message: `[1/3] Pulling latest market data for ${symbol} into the vault...` })}\n\n`);
        let code = await runCommand('uv', ['run', 'python', 'data_vault/yfinance_vault.py', '--symbol', symbol]);
        if (req.destroyed) return;

        if (code !== 0) {
            res.write(`data: ${JSON.stringify({ type: 'log', message: `[!] Data vault sync failed for ${symbol}. Proceeding with existing data...` })}\n\n`);
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'vault_sync', status: 'failed' })}\n\n`);
        } else {
            res.write(`data: ${JSON.stringify({ type: 'log', message: `[✓] Data vault updated successfully.` })}\n\n`);
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'vault_sync', status: 'done' })}\n\n`);
        }

        res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'inference', status: 'running' })}\n\n`);
        res.write(`data: ${JSON.stringify({ type: 'log', message: `[2/3] Initializing live volatility inference for ${symbol}...` })}\n\n`);
        code = await runCommand('uv', ['run', 'python', 'live_volatility_inference.py', '--symbol', symbol]);
        if (req.destroyed) return;

        if (code !== 0) {
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'inference', status: 'failed' })}\n\n`);
            res.write(`data: ${JSON.stringify({ type: 'log', message: `[!] Inference failed. Artifacts missing. [3/3] Initiating full 15-year training for ${symbol}... (This may take several minutes)` })}\n\n`);
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'training', status: 'running' })}\n\n`);
            code = await runCommand('uv', ['run', 'python', 'daily_volatility_engine.py', '--symbol', symbol]);
            if (req.destroyed) return;

            if (code === 0) {
                res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'training', status: 'done' })}\n\n`);
                res.write(`data: ${JSON.stringify({ type: 'log', message: `[✓] Training complete. Re-running live inference...` })}\n\n`);
                res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'inference', status: 'running' })}\n\n`);
                code = await runCommand('uv', ['run', 'python', 'live_volatility_inference.py', '--symbol', symbol]);
                if (req.destroyed) return;
            } else {
                res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'training', status: 'failed' })}\n\n`);
            }
        } else {
            res.write(`data: ${JSON.stringify({ type: 'log', message: `[3/3] Live forecast and visual dashboard completed successfully.` })}\n\n`);
        }

        if (code === 0) {
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'inference', status: 'done' })}\n\n`);
        } else {
            res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'inference', status: 'failed' })}\n\n`);
        }

        res.write(`data: ${JSON.stringify({ type: 'done', code })}\n\n`);
        res.end();
    };

    jobQueue.push(executeChain);

    if (isProcessing) {
        res.write(`data: ${JSON.stringify({ type: 'log', message: `[Queue] Job added. Position: ${jobQueue.length}` })}\n\n`);
        res.write(`data: ${JSON.stringify({ type: 'stage', stage: 'queue', status: 'running', position: jobQueue.length })}\n\n`);
    }

    processQueue();
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Backend server running on http://localhost:${PORT}`);
});

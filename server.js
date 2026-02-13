import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { Audio2FaceSDK } from './audio2face-sdk-node.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = process.env.PORT || 8765;
const MAX_AUDIO_SAMPLES = 16000 * 60;

app.use(express.json({ limit: '10mb' }));
app.use(express.raw({ type: 'audio/*', limit: '10mb' }));
app.use((req, res, next) => {
  const origin = req.headers.origin || '*';
  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const sdk = new Audio2FaceSDK();

async function loadModel() {
  try {
    const configPath = path.join(__dirname, 'a2f_ms_config.json');
    if (fs.existsSync(configPath)) {
      await sdk.loadConfigFile(configPath);
      console.log('Config loaded from a2f_ms_config.json');
    }
    const modelPath = path.join(__dirname, 'network_actual.onnx');
    if (!fs.existsSync(modelPath)) {
      console.error('Model not found:', modelPath);
      return false;
    }
    console.log('Loading model...');
    await sdk.loadModel(modelPath, { useGPU: false });
    console.log(`Model loaded (${sdk.actualBackend}) | inputs: ${sdk.session.inputNames} | outputs: ${sdk.session.outputNames}`);
    return true;
  } catch (err) {
    console.error('Model load failed:', err.message);
    return false;
  }
}

function requireModel(req, res, next) {
  if (!sdk.session) return res.status(503).json({ error: 'MODEL_NOT_LOADED', message: 'Model is still loading. Retry in a few seconds.' });
  next();
}

function parseAudioBody(req) {
  if (req.is('json')) {
    const { audio, emotion } = req.body;
    if (!audio || !Array.isArray(audio)) return { error: 'JSON body must include "audio" array of float samples.' };
    if (audio.length > MAX_AUDIO_SAMPLES) return { error: `Audio exceeds max ${MAX_AUDIO_SAMPLES} samples (${Math.round(MAX_AUDIO_SAMPLES / 16000)}s at 16kHz).` };
    return { audioData: new Float32Array(audio), emotion };
  }
  const buf = req.body;
  if (!buf || !buf.length) return { error: 'Empty audio body. Send raw Float32 PCM or JSON with "audio" array.' };
  if (buf.byteLength % 4 !== 0) return { error: 'Audio body must be Float32 PCM (byte length must be multiple of 4).' };
  const samples = buf.byteLength / 4;
  if (samples > MAX_AUDIO_SAMPLES) return { error: `Audio exceeds max ${MAX_AUDIO_SAMPLES} samples.` };
  return { audioData: new Float32Array(buf.buffer, buf.byteOffset, samples) };
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: !!sdk.session, backend: sdk.actualBackend, timestamp: Date.now() });
});

app.get('/api/config', (req, res) => {
  res.json({
    emotions: Audio2FaceSDK.EMOTIONS,
    blendshapes: Audio2FaceSDK.BLENDSHAPE_NAMES,
    sampleRate: sdk.sampleRate,
    bufferLen: sdk.bufferLen,
    bufferOfs: sdk.bufferOfs,
    faceParams: sdk.faceParams
  });
});

app.post('/api/emotion', requireModel, (req, res) => {
  if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body))
    return res.status(400).json({ error: 'INVALID_INPUT', message: 'Body must be a JSON object of emotion:value pairs.' });
  try {
    sdk.setEmotions(req.body);
    res.json({ ok: true, emotions: Object.fromEntries(Audio2FaceSDK.EMOTIONS.map((e, i) => [e, sdk.emotionVector[i]])) });
  } catch (err) {
    res.status(400).json({ error: 'INVALID_EMOTION', message: err.message });
  }
});

app.post('/api/process', requireModel, async (req, res) => {
  const parsed = parseAudioBody(req);
  if (parsed.error) return res.status(400).json({ error: 'INVALID_INPUT', message: parsed.error });
  try {
    if (parsed.emotion) sdk.setEmotions(parsed.emotion);
    res.json(await sdk.processAudioFile(parsed.audioData));
  } catch (err) {
    console.error('Process error:', err.message);
    res.status(500).json({ error: 'PROCESSING_ERROR', message: err.message });
  }
});

app.post('/api/process-chunk', requireModel, async (req, res) => {
  const parsed = parseAudioBody(req);
  if (parsed.error) return res.status(400).json({ error: 'INVALID_INPUT', message: parsed.error });
  try {
    res.json(await sdk.processAudioChunk(parsed.audioData, parsed.emotion ? { emotion: parsed.emotion } : {}));
  } catch (err) {
    console.error('Chunk error:', err.message);
    res.status(500).json({ error: 'PROCESSING_ERROR', message: err.message });
  }
});

app.use(express.static(__dirname, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mjs')) res.setHeader('Content-Type', 'application/javascript');
    else if (filePath.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm');
  }
}));

app.get('/', (req, res) => { res.sendFile(path.join(__dirname, 'index.html')); });

async function start() {
  await loadModel();
  const server = app.listen(PORT, () => {
    console.log(`\nServer: http://localhost:${PORT}`);
    console.log('Endpoints: GET /api/health, /api/config | POST /api/emotion, /api/process, /api/process-chunk\n');
  });
  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    server.close(() => { sdk.dispose(); process.exit(0); });
    setTimeout(() => process.exit(0), 3000);
  });
}

start().catch(err => { console.error('Server failed:', err); process.exit(1); });

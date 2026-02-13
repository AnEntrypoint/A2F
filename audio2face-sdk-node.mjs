import ort from 'onnxruntime-node';
import { Audio2FaceCore } from './audio2face-core.mjs';

export class Audio2FaceSDK extends Audio2FaceCore {
  constructor(options = {}) {
    super({ ort, ...options });
    this.actualBackend = null;
  }

  static get BLENDSHAPE_NAMES() { return Audio2FaceCore.BLENDSHAPE_NAMES; }
  static get EMOTIONS() { return Audio2FaceCore.EMOTIONS; }

  async loadModel(modelFile, options = {}) {
    const { useGPU = false, threads } = options;
    let modelBuffer;
    if (modelFile instanceof ArrayBuffer) {
      modelBuffer = modelFile;
    } else if (Buffer.isBuffer(modelFile)) {
      modelBuffer = modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength);
    } else if (typeof modelFile === 'string') {
      const fs = await import('fs');
      const buf = fs.readFileSync(modelFile);
      modelBuffer = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    } else {
      throw new Error('Model must be a file path string, ArrayBuffer, or Buffer');
    }

    const os = await import('os');
    const numThreads = threads || Math.max(1, Math.floor(os.cpus().length / 2));
    const providers = useGPU ? ['cuda', 'cpu'] : ['cpu'];
    const sessionOpts = {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      executionMode: 'sequential',
      intraOpNumThreads: numThreads,
      interOpNumThreads: 1,
    };

    try {
      this.session = await ort.InferenceSession.create(modelBuffer, sessionOpts);
      this.actualBackend = providers[0];
    } catch (err) {
      console.warn('GPU failed, falling back to CPU:', err.message);
      sessionOpts.executionProviders = ['cpu'];
      this.session = await ort.InferenceSession.create(modelBuffer, sessionOpts);
      this.actualBackend = 'cpu';
    }
    return this.session;
  }

  async loadQuantizedModel(modelPath, options = {}) {
    console.warn('INT8 quantized model: 4x smaller but may be slower on some CPUs. Use for memory-constrained environments.');
    return this.loadModel(modelPath, options);
  }

  async loadConfigFile(configPath) {
    const fs = await import('fs');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    this.loadConfig(config);
    return config;
  }

  async processAudioFile(audioBuffer) {
    if (!this.session) throw new Error('Model not loaded. Call loadModel() first.');
    let audioData;
    if (Buffer.isBuffer(audioBuffer)) {
      audioData = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);
    } else if (audioBuffer instanceof ArrayBuffer) {
      audioData = new Float32Array(audioBuffer);
    } else if (audioBuffer instanceof Float32Array) {
      audioData = audioBuffer;
    } else {
      throw new Error('Audio must be Buffer, ArrayBuffer, or Float32Array');
    }
    const results = [];
    for (let i = 0; i < audioData.length - this.bufferLen; i += this.bufferOfs) {
      results.push(await this.runInference(audioData.slice(i, i + this.bufferLen)));
    }
    return this.aggregateResults(results);
  }
}

export default Audio2FaceSDK;

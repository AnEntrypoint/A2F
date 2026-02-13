const ARKIT_BLENDSHAPES = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
  'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
];
const EXPLICIT_EMOTIONS = [
  'amazement', 'anger', 'cheekiness', 'disgust', 'fear',
  'grief', 'joy', 'outofbreath', 'pain', 'sadness'
];
const UPPER_FACE_MAX = 19;
const RING_CAPACITY = 16000 * 4;
const sigmoid = x => 1 / (1 + Math.exp(-x));
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export class Audio2FaceCore {
  static get BLENDSHAPE_NAMES() { return ARKIT_BLENDSHAPES; }
  static get EMOTIONS() { return EXPLICIT_EMOTIONS; }

  constructor({ ort, sampleRate, smoothingFactor, config } = {}) {
    this.ort = ort;
    this.session = null;
    this.sampleRate = sampleRate || 16000;
    this.smoothingUpper = smoothingFactor ?? 0.3;
    this.smoothingLower = smoothingFactor ?? 0.3;
    this.bufferLen = 8320;
    this.bufferOfs = 4160;
    this.skinOffset = 0;
    this.skinSize = 140;
    this.tongueOffset = 140;
    this.tongueSize = 10;
    this.jawOffset = 150;
    this.jawSize = 15;
    this.eyesOffset = 165;
    this.eyesSize = 4;
    this.emotionVector = new Float32Array(26);
    this.bsWeightMultipliers = null;
    this.bsWeightOffsets = null;
    this.bsSolveActivePoses = null;
    this.faceParams = {};
    this.lastResult = null;
    this._ring = new Float32Array(RING_CAPACITY);
    this._ringLen = 0;
    this._chunkLock = null;
    this._audioKey = null;
    this._hasEmotion = false;
    this._audioBuf = null;
    this._emotionTensor = null;
    if (config) this.loadConfig(config);
  }

  loadConfig(config) {
    const { audio_params: ap, face_params: fp, network_params: np, blendshape_params: bp } = config;
    if (ap) {
      this.bufferLen = ap.buffer_len ?? this.bufferLen;
      this.bufferOfs = ap.buffer_ofs ?? this.bufferOfs;
      this.sampleRate = ap.samplerate ?? this.sampleRate;
      if (this.bufferOfs > this.bufferLen) this.bufferOfs = this.bufferLen;
    }
    if (fp) {
      this.faceParams = fp;
      this.smoothingUpper = fp.upper_face_smoothing ?? this.smoothingUpper;
      this.smoothingLower = fp.lower_face_smoothing ?? this.smoothingLower;
      if (Array.isArray(fp.emotion)) fp.emotion.forEach((v, i) => { this.emotionVector[i] = v; });
    }
    if (np) {
      this.skinSize = np.num_shapes_skin ?? this.skinSize;
      this.tongueSize = np.num_shapes_tongue ?? this.tongueSize;
      this.tongueOffset = this.skinSize;
      this.jawOffset = this.tongueOffset + this.tongueSize;
      this.jawSize = np.result_jaw_size ?? this.jawSize;
      this.eyesOffset = this.jawOffset + this.jawSize;
      this.eyesSize = np.result_eyes_size ?? this.eyesSize;
    }
    if (bp) {
      this.bsWeightMultipliers = bp.bsWeightMultipliers ?? null;
      this.bsWeightOffsets = bp.bsWeightOffsets ?? null;
      this.bsSolveActivePoses = bp.bsSolveActivePoses ?? null;
    }
  }

  setEmotion(name, value) {
    const idx = EXPLICIT_EMOTIONS.indexOf(name);
    if (idx === -1) throw new Error(`Unknown emotion: ${name}. Valid: ${EXPLICIT_EMOTIONS.join(', ')}`);
    this.emotionVector[idx] = clamp(value);
  }

  setEmotions(obj) { for (const [k, v] of Object.entries(obj)) this.setEmotion(k, v); }
  getEmotionVector() { return new Float32Array(this.emotionVector); }

  _initTensorCache() {
    const names = this.session.inputNames;
    this._audioKey = names.includes('audio') ? 'audio' : names.includes('input') ? 'input' : names[0];
    this._hasEmotion = names.includes('emotion');
    this._audioBuf = new Float32Array(this.bufferLen);
    if (this._hasEmotion)
      this._emotionTensor = new this.ort.Tensor('float32', new Float32Array(26), [1, 1, 26]);
  }

  async runInference(audioChunk) {
    if (!this.session) throw new Error('No session loaded. Call loadModel() first.');
    if (!this._audioKey) this._initTensorCache();
    const feeds = {};
    if (audioChunk.length === this._audioBuf.length) {
      this._audioBuf.set(audioChunk);
      feeds[this._audioKey] = new this.ort.Tensor('float32', this._audioBuf, [1, 1, this._audioBuf.length]);
    } else {
      feeds[this._audioKey] = new this.ort.Tensor('float32', audioChunk, [1, 1, audioChunk.length]);
    }
    if (this._hasEmotion) {
      this._emotionTensor.data.set(this.emotionVector);
      feeds.emotion = this._emotionTensor;
    }
    return this.parseOutputs(await this.session.run(feeds));
  }

  parseOutputs(outputs) {
    const data = outputs[this.session.outputNames[0]].data;
    const numBs = Math.min(ARKIT_BLENDSHAPES.length, this.skinSize);
    const blendshapes = new Array(numBs);
    for (let i = 0; i < numBs; i++) {
      let val = clamp(sigmoid(data[this.skinOffset + i] * 0.1));
      if (this.bsWeightMultipliers) val *= this.bsWeightMultipliers[i] ?? 1;
      if (this.bsWeightOffsets) val += this.bsWeightOffsets[i] ?? 0;
      if (this.bsSolveActivePoses && !this.bsSolveActivePoses[i]) val = 0;
      blendshapes[i] = { name: ARKIT_BLENDSHAPES[i], value: clamp(val) };
    }
    const jaw = clamp(sigmoid((data[this.jawOffset] || 0) * 0.1));
    const eo = this.eyesOffset;
    return {
      blendshapes, jaw, timestamp: Date.now(),
      eyes: { leftX: data[eo] || 0, leftY: data[eo + 1] || 0, rightX: data[eo + 2] || 0, rightY: data[eo + 3] || 0 }
    };
  }

  _ringAppend(audioData) {
    const needed = this._ringLen + audioData.length;
    if (needed > this._ring.length) {
      const newCap = Math.max(needed * 2, RING_CAPACITY);
      const grown = new Float32Array(newCap);
      grown.set(this._ring.subarray(0, this._ringLen));
      this._ring = grown;
    }
    this._ring.set(audioData, this._ringLen);
    this._ringLen += audioData.length;
  }

  _ringConsume(len) {
    this._ring.copyWithin(0, len, this._ringLen);
    this._ringLen -= len;
  }

  async processAudioChunk(audioData, options = {}) {
    if (!this.session) throw new Error('No session loaded. Call loadModel() first.');
    while (this._chunkLock) await this._chunkLock;
    let unlock;
    this._chunkLock = new Promise(r => { unlock = r; });
    try {
      if (options.emotion) this.setEmotions(options.emotion);
      if (audioData.length > 0) this._ringAppend(audioData);
      if (this._ringLen < this.bufferLen) return this.lastResult || this.getEmptyResult();
      const chunk = this._ring.slice(0, this.bufferLen);
      this._ringConsume(this.bufferOfs);
      const result = await this.runInference(chunk);
      if (this.lastResult)
        result.blendshapes = this.smoothBlendshapes(this.lastResult.blendshapes, result.blendshapes);
      this.lastResult = result;
      return result;
    } finally { this._chunkLock = null; unlock(); }
  }

  smoothBlendshapes(prev, curr) {
    if (!prev || !curr || prev.length !== curr.length) return curr;
    return curr.map((bs, i) => {
      const f = i <= UPPER_FACE_MAX ? this.smoothingUpper : this.smoothingLower;
      return { name: bs.name, value: prev[i].value * f + bs.value * (1 - f) };
    });
  }

  aggregateResults(results) {
    if (!results.length) return this.getEmptyResult();
    const blendshapes = results[0].blendshapes.map((bs, i) => ({
      name: bs.name, value: results.reduce((a, r) => a + r.blendshapes[i].value, 0) / results.length
    }));
    return {
      blendshapes, frameCount: results.length,
      jaw: results.reduce((a, r) => a + r.jaw, 0) / results.length,
      eyes: results[results.length - 1].eyes
    };
  }

  getEmptyResult() {
    return { blendshapes: ARKIT_BLENDSHAPES.map(name => ({ name, value: 0 })), jaw: 0, eyes: null, timestamp: Date.now() };
  }

  resampleAudio(audioData, fromRate, toRate) {
    const ratio = toRate / fromRate, newLen = Math.floor(audioData.length * ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const pos = i / ratio, idx = Math.floor(pos), frac = pos - idx;
      result[i] = idx >= audioData.length - 1 ? audioData[audioData.length - 1] : audioData[idx] * (1 - frac) + audioData[idx + 1] * frac;
    }
    return result;
  }

  setSmoothing(factor) { this.smoothingUpper = this.smoothingLower = clamp(factor); }

  setSmoothingRegion(region, factor) {
    if (region === 'upper') this.smoothingUpper = clamp(factor);
    else if (region === 'lower') this.smoothingLower = clamp(factor);
    else throw new Error(`Unknown region: ${region}. Valid: upper, lower`);
  }

  dispose() {
    try { if (this.session) this.session.release(); } catch (_) {}
    this.session = null;
    this._ring = new Float32Array(RING_CAPACITY);
    this._ringLen = 0;
    this.lastResult = null;
    this._audioKey = null;
    this._hasEmotion = false;
    this._audioBuf = null;
    this._emotionTensor = null;
  }
}

export default Audio2FaceCore;

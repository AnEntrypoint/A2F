import ort from 'onnxruntime-node';

export class Audio2FaceSDK {
    constructor() {
        this.session = null;
        this.sampleRate = 16000;
        this.bufferLen = 8320;
        this.bufferOfs = 4160;
        this.audioBuffer = new Float32Array(0);
        this.lastResult = null;
        this.smoothingFactor = 0.3;
        this.actualBackend = null;
        this.skinOffset = 0;
        this.skinSize = 140;
        this.tongueOffset = 140;
        this.tongueSize = 10;
        this.jawOffset = 150;
        this.jawSize = 15;
        this.eyesOffset = 165;
        this.eyesSize = 4;

        this.blendshapeNames = this.generateBlendshapeNames();
    }

    generateBlendshapeNames() {
        return [
            'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft',
            'browOuterUpRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft',
            'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
            'eyeLookOutRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft',
            'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'cheekPuff',
            'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight',
            'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
            'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
            'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
            'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
            'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
            'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
            'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
        ];
    }

    async loadModel(modelFile, options = {}) {
        const { useGPU = false } = options;
        
        let modelBuffer;
        if (modelFile instanceof ArrayBuffer) {
            modelBuffer = modelFile;
        } else if (Buffer.isBuffer(modelFile)) {
            modelBuffer = modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength);
        } else if (typeof modelFile === 'string') {
            // Read file from path
            const fs = await import('fs');
            const buffer = fs.readFileSync(modelFile);
            modelBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        } else if (modelFile instanceof Blob) {
            modelBuffer = await modelFile.arrayBuffer();
        } else {
            throw new Error('Model must be a File, Blob, ArrayBuffer, Buffer, or URL string');
        }

        const sessionOptions = {
            executionProviders: useGPU ? ['cuda', 'cpu'] : ['cpu'],
            graphOptimizationLevel: 'all',
        };

        try {
            this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
            this.actualBackend = sessionOptions.executionProviders[0];
        } catch (err) {
            console.warn('GPU initialization failed, falling back to CPU:', err);
            sessionOptions.executionProviders = ['cpu'];
            this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
            this.actualBackend = 'cpu';
        }

        return this.session;
    }

    async processAudioFile(audioBuffer) {
        if (!this.session) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Handle Node.js Buffer or ArrayBuffer
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
        const hopSize = this.bufferOfs;
        
        for (let i = 0; i < audioData.length - this.bufferLen; i += hopSize) {
            const chunk = audioData.slice(i, i + this.bufferLen);
            const result = await this.runInference(chunk);
            results.push(result);
        }

        return this.aggregateResults(results);
    }

    async processAudioChunk(audioData) {
        if (!this.session) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        const newBuffer = new Float32Array(this.audioBuffer.length + audioData.length);
        newBuffer.set(this.audioBuffer);
        newBuffer.set(audioData, this.audioBuffer.length);
        this.audioBuffer = newBuffer;

        if (this.audioBuffer.length < this.bufferLen) {
            return this.lastResult || this.getEmptyResult();
        }

        const chunk = this.audioBuffer.slice(0, this.bufferLen);
        this.audioBuffer = this.audioBuffer.slice(this.bufferOfs);

        const result = await this.runInference(chunk);
        
        if (this.lastResult) {
            result.blendshapes = this.smoothBlendshapes(
                this.lastResult.blendshapes,
                result.blendshapes
            );
        }
        
        this.lastResult = result;
        return result;
    }

    async runInference(audioChunk) {
        if (!this.session) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }
        const inputTensor = new ort.Tensor('float32', audioChunk, [1, 1, audioChunk.length]);
        
        const feeds = {};
        const inputNames = this.session.inputNames;
        
        // Feed audio input
        if (inputNames.includes('audio')) {
            feeds.audio = inputTensor;
        } else if (inputNames.includes('input')) {
            feeds.input = inputTensor;
        } else {
            feeds[inputNames[0]] = inputTensor;
        }
        
        if (inputNames.includes('emotion')) {
            const emotionData = new Float32Array(26);
            feeds.emotion = new ort.Tensor('float32', emotionData, [1, 1, 26]);
        }

        const results = await this.session.run(feeds);
        
        return this.parseOutputs(results);
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    parseOutputs(outputs) {
        const data = outputs[this.session.outputNames[0]].data;

        const blendshapes = [];
        const numBs = Math.min(this.blendshapeNames.length, this.skinSize);
        for (let i = 0; i < numBs; i++) {
            const raw = data[this.skinOffset + i];
            blendshapes.push({
                name: this.blendshapeNames[i],
                value: Math.max(0, Math.min(1, this.sigmoid(raw * 0.1)))
            });
        }

        const jawValues = [];
        for (let i = 0; i < this.jawSize; i++) {
            jawValues.push(data[this.jawOffset + i] || 0);
        }
        const jaw = Math.max(0, Math.min(1, this.sigmoid(jawValues[0] * 0.1)));

        const eo = this.eyesOffset;
        const eyes = {
            leftX: data[eo] || 0,
            leftY: data[eo + 1] || 0,
            rightX: data[eo + 2] || 0,
            rightY: data[eo + 3] || 0
        };

        return { blendshapes, jaw, eyes, timestamp: Date.now() };
    }

    smoothBlendshapes(prev, curr) {
        if (!prev || !curr || prev.length !== curr.length) {
            return curr;
        }

        return curr.map((bs, i) => ({
            name: bs.name,
            value: prev[i].value * this.smoothingFactor + bs.value * (1 - this.smoothingFactor)
        }));
    }

    aggregateResults(results) {
        if (results.length === 0) {
            return this.getEmptyResult();
        }

        const avgBlendshapes = [];
        const numBs = results[0].blendshapes.length;

        for (let i = 0; i < numBs; i++) {
            const sum = results.reduce((acc, r) => acc + r.blendshapes[i].value, 0);
            avgBlendshapes.push({
                name: results[0].blendshapes[i].name,
                value: sum / results.length
            });
        }

        return {
            blendshapes: avgBlendshapes,
            jaw: results.reduce((acc, r) => acc + r.jaw, 0) / results.length,
            eyes: results[results.length - 1].eyes,
            frameCount: results.length
        };
    }

    getEmptyResult() {
        return {
            blendshapes: this.blendshapeNames.map(name => ({ name, value: 0 })),
            jaw: 0,
            eyes: null,
            timestamp: Date.now()
        };
    }

    resampleAudio(audioData, fromRate, toRate) {
        const ratio = toRate / fromRate;
        const newLength = Math.floor(audioData.length * ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const pos = i / ratio;
            const index = Math.floor(pos);
            const frac = pos - index;

            if (index >= audioData.length - 1) {
                result[i] = audioData[audioData.length - 1];
            } else {
                result[i] = audioData[index] * (1 - frac) + audioData[index + 1] * frac;
            }
        }

        return result;
    }

    setSmoothing(factor) {
        this.smoothingFactor = Math.max(0, Math.min(1, factor));
    }

    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        this.audioBuffer = new Float32Array(0);
        this.lastResult = null;
    }
}

export default Audio2FaceSDK;
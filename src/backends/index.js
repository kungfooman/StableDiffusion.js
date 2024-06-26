// @ts-ignore
import * as ORT from '@aislamov/onnxruntime-web64';
import { Tensor         } from '@xenova/transformers';
import { replaceTensors } from '../util/Tensor.js';
// @ts-ignore
const ONNX = ORT.default ?? ORT
const isNode = typeof process !== 'undefined' && process?.release?.name === 'node'
const onnxSessionOptions = isNode
  ? {
    executionProviders: ['cpu'],
    executionMode: 'parallel',
  }
  : {
    executionProviders: ['webgpu'],
  }

export class Session {
  /** @type {import('onnxruntime-common').InferenceSession} */
  session;
  /**
   * @param {InferenceSession} session 
   * @param {Record<string, unknown>} [config] 
   */
  constructor (session, config = {}) {
    this.session = session;
    this.config = config || {};
  }
  /**
   * @param {string|ArrayBuffer} modelOrPath 
   * @param {string|ArrayBuffer} [weightsPathOrBuffer] 
   * @param {string} [weightsFilename] 
   * @param {Record<string, unknown>} [config] 
   * @param {InferenceSession.SessionOptions} [options] 
   */
  static async create(modelOrPath, weightsPathOrBuffer, weightsFilename, config, options) {
    const arg = typeof modelOrPath === 'string' ? modelOrPath : new Uint8Array(modelOrPath)

    const sessionOptions = {
      ...onnxSessionOptions,
      ...options,
    }

    const weightsParams = {
      externalWeights: weightsPathOrBuffer,
      externalWeightsFilename: weightsFilename,
    }
    const executionProviders = sessionOptions.executionProviders.map((provider) => {
      if (typeof provider === 'string') {
        return {
          name: provider,
          ...weightsParams,
        }
      }

      return {
        ...provider,
        ...weightsParams,
      }
    })

    // @ts-ignore
    const session = ONNX.InferenceSession.create(arg, {
      ...sessionOptions,
      executionProviders,
    })

    // @ts-ignore
    return new Session(await session, config)
  }
  /**
   * @param {Record<string, Tensor>} inputs 
   */
  async run (inputs) {
    // @ts-ignore
    const result = await this.session.run(inputs)
    return replaceTensors(result)
  }

  release () {
    return this.session.release()
  }
}

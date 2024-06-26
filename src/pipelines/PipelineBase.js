import {Tensor                 } from '@xenova/transformers';
import {Session                } from '../backends/index.js';
import {CLIPTokenizer          } from '../tokenizers/CLIPTokenizer.js';
import {SchedulerBase          } from '../schedulers/SchedulerBase.js';
import {cat, randomNormalTensor} from '../util/Tensor.js';
export class PipelineBase {
  /** @type {Session} */
  unet;
  /** @type {Session} */
  vaeDecoder;
  /** @type {Session} */
  vaeEncoder;
  /** @type {Session} */
  textEncoder;
  /** @type {CLIPTokenizer} */
  tokenizer;
  /** @type {SchedulerBase} */
  scheduler;
  /** @type {number} */
  vaeScaleFactor;
  /**
   * @param {string} prompt 
   * @returns {Promise<Tensor>}
   */
  async encodePrompt (prompt) {
    const tokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )
    const inputIds = tokens.input_ids
    // @ts-ignore
    const encoded = await this.textEncoder.run({ input_ids: new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length]) })
    return encoded.last_hidden_state;
  }
  /**
   * @param {string} prompt 
   * @param {string | undefined} negativePrompt 
   */
  async getPromptEmbeds (prompt, negativePrompt) {
    const promptEmbeds = await this.encodePrompt(prompt)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '')
    return cat([negativePromptEmbeds, promptEmbeds])
  }
  /**
   * 
   * @param {number} batchSize 
   * @param {number} numChannels 
   * @param {number} height 
   * @param {number} width 
   * @param {string} seed 
   * @returns 
   */
  prepareLatents (batchSize, numChannels, height, width, seed = '') {
    const latentShape = [
      batchSize,
      numChannels,
      Math.floor(width / this.vaeScaleFactor),
      height / this.vaeScaleFactor,
    ]
    return randomNormalTensor(latentShape, undefined, undefined, 'float32', seed)
  }
  /**
   * @param {Tensor} latents 
   */
  async makeImages (latents) {
    latents = latents.div(this.vaeDecoder.config.scaling_factor || 0.18215)
    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents },
    )
    const images = decoded.sample
      .div(2)
      .add(0.5)
      .clipByValue(0, 1)
    // .mul(255)
    // .round()
    // .clipByValue(0, 255)
    // .transpose(0, 2, 3, 1)
    return [images]
  }
  async release () {
    await this.unet?.release();
    await this.vaeDecoder?.release();
    await this.vaeEncoder?.release();
    await this.textEncoder?.release();
  }
}

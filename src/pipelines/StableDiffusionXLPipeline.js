import {Tensor                                                 } from '@xenova/transformers'
import {PNDMScheduler                                          } from '../schedulers/PNDMScheduler.js';
import {CLIPTokenizer                                          } from '../tokenizers/CLIPTokenizer.js';
import {cat, randomNormalTensor                                } from '../util/Tensor.js';
import {dispatchProgress, loadModel, ProgressStatus, sessionRun} from './common.js';
import {getModelFile, getModelJSON                             } from '../hub/index.js';
import {Session                                                } from '../backends/index.js';
import {PipelineBase                                           } from './PipelineBase.js';
/** @typedef {import('./common.js'                   ).PretrainedOptions  } PretrainedOptions   */
/** @typedef {import('./common.js'                   ).ProgressCallback   } ProgressCallback    */
/** @typedef {import('../hub/common.js'              ).GetModelFileOptions} GetModelFileOptions */
/** @typedef {import('../schedulers/PNDMScheduler.js').PNDMSchedulerConfig} PNDMSchedulerConfig */
/** @typedef {import('../schedulers/SchedulerBase.js').SchedulerConfig    } SchedulerConfig     */
/**
 * @typedef {Object} StableDiffusionXLInput
 * @property {string} prompt
 * @property {string} [negativePrompt]
 * @property {number} [guidanceScale]
 * @property {string} [seed]
 * @property {number} [width]
 * @property {number} [height]
 * @property {number} numInferenceSteps
 * @property {boolean} [sdV1]
 * @property {ProgressCallback} [progressCallback]
 * @property {boolean} [runVaeOnEachStep]
 * @property {boolean} [img2imgFlag]
 * @property {Float32Array} [inputImage]
 * @property {number} [strength]
 */
export class StableDiffusionXLPipeline extends PipelineBase {
  /** @type {Session} */
  textEncoder2;
  /** @type {CLIPTokenizer} */
  tokenizer2;
  /** @type {PNDMScheduler} */
  scheduler;
  /**
   * @param {Session} unet 
   * @param {Session} vaeDecoder 
   * @param {Session} textEncoder 
   * @param {Session} textEncoder2 
   * @param {CLIPTokenizer} tokenizer 
   * @param {CLIPTokenizer} tokenizer2 
   * @param {PNDMScheduler} scheduler 
   */
  constructor (unet, vaeDecoder, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler) {
    super();
    this.unet = unet;
    this.vaeDecoder = vaeDecoder;
    this.textEncoder = textEncoder;
    this.textEncoder2 = textEncoder2;
    this.tokenizer = tokenizer;
    this.tokenizer2 = tokenizer2;
    this.scheduler = scheduler;
  }
  /**
   * @param {PNDMSchedulerConfig} config 
   */
  static createScheduler (config) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }
  /**
   * @param {string} modelRepoOrPath 
   * @param {PretrainedOptions} [options] 
   */
  static async fromPretrained (modelRepoOrPath, options) {
    /** @type {GetModelFileOptions} */
    const opts = {
      ...options,
    };
    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    const tokenizer2 = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer_2' })
    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder2 = await loadModel(modelRepoOrPath, 'text_encoder_2/model.onnx', opts)
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)
    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionXLPipeline(unet, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)
  }
  /**
   * @param {string} prompt 
   * @param {CLIPTokenizer} tokenizer 
   * @param {Session} textEncoder 
   */
  async encodePromptXl (prompt, tokenizer, textEncoder) {
    const tokens = tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )
    const inputIds = tokens.input_ids;
    const tensor = new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length]);
    const result = await sessionRun(textEncoder, { input_ids: tensor })
    console.log(Object.keys(result));
    const {
      last_hidden_state: lastHiddenState,
      pooler_output: poolerOutput,
      // hidden_states: hiddenStates,
      'hidden_states.11': hiddenStates,
    } = result;
    return {lastHiddenState, poolerOutput, hiddenStates};
  }
  /**
   * @param {string} prompt 
   * @param {string|undefined} negativePrompt 
   */
  async getPromptEmbedsXl (prompt, negativePrompt) {
    const promptEmbeds          = await this.encodePromptXl(prompt              , this.tokenizer , this.textEncoder );
    const negativePromptEmbeds  = await this.encodePromptXl(negativePrompt || '', this.tokenizer , this.textEncoder );
    const promptEmbeds2         = await this.encodePromptXl(prompt              , this.tokenizer2, this.textEncoder2);
    const negativePromptEmbeds2 = await this.encodePromptXl(negativePrompt || '', this.tokenizer2, this.textEncoder2);
    return {
      hiddenStates: cat([
        cat([negativePromptEmbeds.hiddenStates, negativePromptEmbeds2.hiddenStates], -1),
        cat([promptEmbeds.hiddenStates, promptEmbeds2.hiddenStates], -1),
      ]),
      textEmbeds: cat([randomNormalTensor(negativePromptEmbeds2.lastHiddenState.dims), randomNormalTensor(promptEmbeds2.lastHiddenState.dims)]),
    }
  }
  /**
   * @param {number} width 
   * @param {number} height 
   */
  getTimeEmbeds (width, height) {
    return new Tensor(
      'float32',
      Float32Array.from([height, width, 0, 0, height, width, height, width, 0, 0, height, width]),
      [2, 6],
    )
  }
  /**
   * @param {StableDiffusionXLInput} input 
   */
  async run (input) {
    const width = input.width || 1024
    const height = input.height || 1024
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 5
    const seed = input.seed || ''
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })
    const promptEmbeds = await this.getPromptEmbedsXl(input.prompt, input.negativePrompt)
    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image
    const timesteps = this.scheduler.timesteps.data
    const doClassifierFreeGuidance = guidanceScale > 1
    let humanStep = 1;
    /** @type {Tensor[]|null} */
    let cachedImages = null;
    const timeIds = this.getTimeEmbeds(width, height)
    const hiddenStates = promptEmbeds.hiddenStates
    const textEmbeds = promptEmbeds.textEmbeds
    for (const step of timesteps) {
      const timestep = new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents
      console.log('running', {
        sample: latentInput,
        timestep,
        encoder_hidden_states: hiddenStates,
        text_embeds: textEmbeds,
        time_ids: timeIds,
      })
      const noise = await this.unet.run(
        {
          sample: latentInput,
          timestep,
          encoder_hidden_states: hiddenStates,
          text_embeds: textEmbeds,
          time_ids: timeIds,
        },
      )
      console.log('noise', noise)
      let noisePred = noise.out_sample
      if (doClassifierFreeGuidance) {
        const [noisePredUncond, noisePredText] = [
          noisePred.slice([0, 1]),
          noisePred.slice([1, 2]),
        ]
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      }
      latents = this.scheduler.step(
        noisePred,
        step,
        latents,
      )
      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(latents)
      }
      humanStep++
    }
    if (input.runVaeOnEachStep) {
      return cachedImages;
    }
    return this.makeImages(latents)
  }
  /**
   * @param {Tensor} latents 
   */
  async makeImages (latents) {
    latents = latents.mul(0.13025)
    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents },
    )
    const images = decoded.sample
      .div(2)
      .add(0.5)
    return [images]
  }
  async release () {
    await super.release()
    return this.textEncoder2?.release()
  }
}

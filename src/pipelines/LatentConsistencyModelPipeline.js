import {dispatchProgress, loadModel, ProgressStatus} from './common.js';
import {Session                                    } from '../backends/index.js';
import {LCMScheduler                               } from '../schedulers/LCMScheduler.js';
import {CLIPTokenizer                              } from '../tokenizers/CLIPTokenizer.js';
import {getModelJSON                               } from '../hub/index.js';
import {cat, linspace, randomNormalTensor, range   } from '../util/Tensor.js';
import {DiffusionPipeline                          } from './DiffusionPipeline.js';
import {PipelineBase                               } from './PipelineBase.js';
import {Tensor                                     } from '@xenova/transformers';
import {seedrandom                                 } from '../seedrandom.js';
/** @typedef {import('./common.js'                  ).PretrainedOptions  } PretrainedOptions   */
/** @typedef {import('./common.js'                  ).ProgressCallback   } ProgressCallback    */
/** @typedef {import('../schedulers/LCMScheduler.js').LCMSchedulerConfig } LCMSchedulerConfig  */
/** @typedef {import('../hub/common.js'             ).GetModelFileOptions} GetModelFileOptions */
/**
 * @typedef {Object} StableDiffusionInput
 * @property {string} prompt - The main input prompt for the stable diffusion process.
 * @property {string} [negativePrompt] - Optional negative prompt to guide the diffusion away from certain features.
 * @property {number} [guidanceScale] - Optional scale for influencing the guidance of the diffusion.
 * @property {string} [seed] - Optional seed for deterministic results.
 * @property {number} [width] - Optional width for the output image.
 * @property {number} [height] - Optional height for the output image.
 * @property {number} numInferenceSteps - The number of inference steps to perform.
 * @property {boolean} [sdV1] - Optional flag for using the version 1 of the model.
 * @property {ProgressCallback} [progressCallback] - Optional callback for progress updates.
 * @property {boolean} [runVaeOnEachStep] - Optional flag to run the variational autoencoder on each step.
 * @property {boolean} [img2imgFlag] - Optional flag for an image-to-image operation.
 * @property {Float32Array} [inputImage] - Optional input image as a Float32Array for image-to-image operations.
 * @property {number} [strength] - Optional strength parameter for image-to-image operations.
 */
export class LatentConsistencyModelPipeline extends PipelineBase {
  /** @type {LCMScheduler} */
  scheduler;
  /**
   * 
   * @param {Session} unet 
   * @param {Session} vaeDecoder 
   * @param {Session} vaeEncoder 
   * @param {Session} textEncoder 
   * @param {CLIPTokenizer} tokenizer 
   * @param {LCMScheduler} scheduler 
   */
  constructor (unet, vaeDecoder, vaeEncoder, textEncoder, tokenizer, scheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 2 ** ((this.vaeDecoder.config.block_out_channels).length - 1);
  }
  /**
   * @param {LCMSchedulerConfig} config 
   */
  static createScheduler(config) {
    return new LCMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }
  /**
   * 
   * @param {string} modelRepoOrPath 
   * @param {PretrainedOptions} [options] 
   * @returns 
   */
  static async fromPretrained (modelRepoOrPath, options) {
    /** @type {GetModelFileOptions} */
    const opts = {
      ...options,
    };
    // order matters because WASM memory cannot be decreased. so we load the biggest one first
    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)
    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = LatentConsistencyModelPipeline.createScheduler(schedulerConfig)
    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new LatentConsistencyModelPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }
  /**
   * @param {number} batchSize 
   * @param {number} guidanceScale 
   * @param {number} embeddingDim 
   */
  getWEmbedding (batchSize, guidanceScale, embeddingDim = 512) {
    let w = new Tensor('float32', new Float32Array([guidanceScale]), [1])
    w = w.mul(1000)
    const halfDim = embeddingDim / 2
    let log = Math.log(10000) / (halfDim - 1);
    /** @type {Tensor} */
    let emb = range(0, halfDim).mul(-log).exp()
    // TODO: support batch size > 1
    emb = emb.mul(w.data[0]);
    return cat([emb.sin(), emb.cos()]).reshape([batchSize, embeddingDim])
  }
  /**
   * @param {StableDiffusionInput} input 
   */
  async run (input) {
    const width = input.width || this.unet.config.sample_size * this.vaeScaleFactor;
    const height = input.height || this.unet.config.sample_size * this.vaeScaleFactor;
    const batchSize = 1;
    const guidanceScale = input.guidanceScale || 8.5;
    const seed = input.seed || Math.random().toString(16).slice(2);
    console.log("Seed", seed);
    const rngObject = seedrandom(seed);
    console.log("rngObject", rngObject);
    const rng = rngObject.prng;

    this.scheduler.setTimesteps(input.numInferenceSteps || 5);
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })
    const promptEmbeds = await this.encodePrompt(input.prompt)
    let latents = this.prepareLatents(
      batchSize,
      this.unet.config.in_channels || 4,
      height,
      width,
      rng,
    )
    let timesteps = this.scheduler.timesteps.data
    let humanStep = 1;
    /** @type {Tensor[] | null} */
    let cachedImages = null;
    const wEmbedding = this.getWEmbedding(batchSize, guidanceScale, 256);
    /** @type {Tensor} */
    let denoised;
    for (const step of timesteps) {
      const timestep = new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      const noise = await this.unet.run(
        {sample: latents, timestep, encoder_hidden_states: promptEmbeds, timestep_cond: wEmbedding},
      );
      [latents, denoised] = this.scheduler.step(
        noise.out_sample,
        step,
        humanStep - 1,
        latents,
        rng,
      )
      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Done,
    })
    if (input.runVaeOnEachStep) {
      return cachedImages;
    }
    return this.makeImages(denoised)
  }
}

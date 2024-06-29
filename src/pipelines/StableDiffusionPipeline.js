import {Tensor                                     } from '@xenova/transformers'
import {PNDMScheduler                              } from '../schedulers/PNDMScheduler.js';
import {CLIPTokenizer                              } from '../tokenizers/CLIPTokenizer.js';
import {cat, randomNormalTensor                    } from '../util/Tensor.js';
import {dispatchProgress, loadModel, ProgressStatus} from './common.js';
import {getModelJSON                               } from '../hub/index.js';
import {Session                                    } from '../backends/index.js';
import {PipelineBase                               } from './PipelineBase.js';
import {seedrandom                                 } from '../seedrandom.js';
/** @typedef {import('./common.js'                   ).PretrainedOptions  } PretrainedOptions   */
/** @typedef {import('./common.js'                   ).ProgressCallback   } ProgressCallback    */
/** @typedef {import('../schedulers/PNDMScheduler.js').PNDMSchedulerConfig} PNDMSchedulerConfig */
/** @typedef {import('../hub/common.js'              ).GetModelFileOptions} GetModelFileOptions */
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
export class StableDiffusionPipeline extends PipelineBase {
  /** @type {PNDMScheduler} */
  scheduler;
  /**
   * 
   * @param {Session} unet 
   * @param {Session} vaeDecoder 
   * @param {Session} vaeEncoder 
   * @param {Session} textEncoder 
   * @param {CLIPTokenizer} tokenizer 
   * @param {PNDMScheduler} scheduler 
   */
  constructor (unet, vaeDecoder, vaeEncoder, textEncoder, tokenizer, scheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
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
    }
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
    const scheduler = StableDiffusionPipeline.createScheduler(schedulerConfig)
    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }
  /**
   * @param {StableDiffusionInput} input 
   */
  async run(input) {
    const width = input.width || 512;
    const height = input.height || 512;
    const batchSize = 1;
    const guidanceScale = input.guidanceScale || 7.5;
    const seed = input.seed || Math.random().toString(16).slice(2);
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Seed,
      seed,
    });
    const {prng} = seedrandom(seed);
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    });
    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt);
    const latentShape = [batchSize, 4, width / 8, height / 8];
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', prng); // Normal latents used in Text-to-Image
    let timesteps = this.scheduler.timesteps.data;
    if (input.img2imgFlag) {
      const inputImage = input.inputImage || new Float32Array()
      const strength = input.strength || 0.8
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.EncodingImg2Img,
      })
      const imageLatent = await this.encodeImage(inputImage, input.width, input.height) // Encode image to latent space
      // Taken from https://towardsdatascience.com/stable-diffusion-using-hugging-face-variations-of-stable-diffusion-56fd2ab7a265#2d1d
      const initTimestep = Math.round(input.numInferenceSteps * strength)
      const timestep = timesteps.toReversed()[initTimestep]
      latents = this.scheduler.addNoise(imageLatent, latents, timestep)
      // Computing the timestep to start the diffusion loop
      const tStart = Math.max(input.numInferenceSteps - initTimestep, 0)
      timesteps = timesteps.slice(tStart)
    }
    const doClassifierFreeGuidance = guidanceScale > 1
    let humanStep = 1;
    /** @type {Tensor[] | null} */
    let cachedImages = null;
    for (const step of timesteps) {
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      // but currently onnxruntime-node does not give out types, only input names
      const timestep = input.sdV1
        ? new Tensor(BigInt64Array.from([BigInt(step)]))
        : new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents
      const noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds },
      )
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
    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Done,
    });
    if (input.runVaeOnEachStep) {
      return cachedImages;
    }
    return this.makeImages(latents);
  }
  /**
   * 
   * @param {Float32Array} inputImage 
   * @param {number} width 
   * @param {number} height 
   * @returns 
   */
  async encodeImage (inputImage, width, height) {
    const encoded = await this.vaeEncoder.run(
      { sample: new Tensor('float32', inputImage, [1, 3, width, height]) },
    );
    const encodedImage = encoded.latent_sample;
    return encodedImage.mul(0.18215);
  }
}

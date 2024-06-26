import { linspace, range } from '@/util/Tensor'
import { betasForAlphaBar } from '@/schedulers/common'
import { Tensor } from '@xenova/transformers'
/**
 * @typedef {Object} SchedulerConfig
 * @property {number} beta_end
 * @property {string} beta_schedule
 * @property {number} beta_start
 * @property {boolean} clip_sample
 * @property {number} num_train_timesteps
 * @property {'epsilon'|'v_prediction'|'sample'} [prediction_type]
 * @property {boolean} set_alpha_to_one
 * @property {number} steps_offset
 * @property {null} trained_betas
 */
export class SchedulerBase {
  /** @type {Tensor} */
  betas;
  /** @type {Tensor} */
  alphas;
  /** @type {Tensor} */
  alphasCumprod;
  /** @type {number} */
  finalAlphaCumprod;
  /** @type {Tensor} */
  timesteps;
  numInferenceSteps = 20;
  /**
   * @param {SchedulerConfig} config 
   */
  constructor (config) {
    if (config.trained_betas !== null) {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
    } else if (config.beta_schedule === 'linear') {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
    } else if (config.beta_schedule === 'scaled_linear') {
      this.betas = linspace(config.beta_start ** 0.5, config.beta_end ** 0.5, config.num_train_timesteps).pow(2)
    } else if (config.beta_schedule === 'squaredcos_cap_v2') {
      this.betas = betasForAlphaBar(config.num_train_timesteps)
    } else {
      throw new Error(`${config.beta_schedule} does is not implemented for ${this.constructor}`)
    }

    this.timesteps = range(0, config.num_train_timesteps).reverse()
    this.alphas = linspace(1, 1, config.num_train_timesteps).sub(this.betas)
    this.alphasCumprod = this.alphas.cumprod()
    this.finalAlphaCumprod = config.set_alpha_to_one ? 1.0 : this.alphasCumprod[0].data
    this.config = config
  }
  /**
   * @param {Tensor} input 
   * @param {number} [timestep] 
   */
  scaleModelInput (input, timestep) {
    return input;
  }
  /**
   * @param {Tensor} originalSamples 
   * @param {Tensor} noise 
   * @param {number} timestep 
   */
  addNoise (originalSamples, noise, timestep) {
    const sqrtAlphaProd = this.alphasCumprod.data[timestep] ** 0.5;
    const sqrtOneMinusAlphaProd = (1 - this.alphasCumprod.data[timestep]) ** 0.5;
    return originalSamples.mul(sqrtAlphaProd).add(noise.mul(sqrtOneMinusAlphaProd));
  }
}

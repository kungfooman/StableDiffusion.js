import {Tensor                          } from '@xenova/transformers'
import {SchedulerBase                   } from './SchedulerBase.js';
import {randomNormalTensor, scalarTensor} from '../util/Tensor.js';
/** @typedef {import('.SchedulerBase.js').SchedulerConfig} SchedulerConfig */
/* @todo
export interface LCMSchedulerConfig extends SchedulerConfig {
  rescale_betas_zero_snr: boolean,
  thresholding: boolean,
  original_inference_steps: number
}
*/
/** @typedef {object} LCMSchedulerConfig */
export class LCMScheduler extends SchedulerBase {
  /** @type {number} */
  initNoiseSigma;
  /**
   * @param {LCMSchedulerConfig} config 
   */
  constructor (config) {
    super({
      rescale_betas_zero_snr: false,
      beta_start: 0.0001,
      beta_end: 0.02,
      beta_schedule: 'linear',
      clip_sample: true,
      set_alpha_to_one: true,
      steps_offset: 0,
      prediction_type: 'epsilon',
      thresholding: false,
      ...config,
    })
    this.initNoiseSigma = 1.0
  }
  /**
   * @param {number} timestep 
   * @param {number} prevTimestep 
   */
  getVariance (timestep, prevTimestep) {
    const alphaProdT = this.alphasCumprod.data[timestep]
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod.data[prevTimestep] : this.finalAlphaCumprod
    const betaProdT = 1 - alphaProdT
    const betaProdTPrev = 1 - alphaProdTPrev
    return (betaProdTPrev / betaProdT) * (1 - alphaProdT / alphaProdTPrev)
  }
  /**
   * @param {number} timestep 
   */
  getScalingsForBoundaryConditionDiscrete (timestep) {
    const sigmaData = 0.5
    const cSkip = sigmaData ** 2 / ((timestep / 0.1) ** 2 + sigmaData ** 2)
    const cOut = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigmaData ** 2) ** 0.5
    return [cSkip, cOut]
  }
  /**
   * @param {Tensor} modelOutput 
   * @param {number} timestep 
   * @param {number} timeIndex 
   * @param {Tensor} sample 
   * @param {() => number} rng
   * @returns {Tensor[]}
   */
  step(modelOutput, timestep, timeIndex, sample, rng) {
    if (!this.numInferenceSteps) {
      throw new Error('numInferenceSteps is not set');
    }
    const prevTimeIndex = timeIndex + 1
    let prevTimeStep
    if (prevTimeIndex < this.timesteps.data.length) {
      prevTimeStep = this.timesteps.data[prevTimeIndex]
    } else {
      prevTimeStep = timestep
    }
    const alphaProdT = this.alphasCumprod[timestep].data[0]
    const alphaProdTPrev = prevTimeStep >= 0 ? this.alphasCumprod[prevTimeStep].data[0] : this.finalAlphaCumprod
    const betaProdT = 1 - alphaProdT
    const betaProdTPrev = 1 - alphaProdTPrev
    const [cSkip, cOut] = this.getScalingsForBoundaryConditionDiscrete(timestep);
    /** @type {Tensor} */
    let predX0;
    const parametrization = this.config.prediction_type;
    if (parametrization === 'epsilon') {
      predX0 = sample.sub(
        modelOutput.mul(Math.sqrt(betaProdT)),
      ).div(Math.sqrt(alphaProdT))
    } else if (parametrization === 'sample') {
      predX0 = sample
    } else if (parametrization === 'v_prediction') {
      predX0 = sample.mul(Math.sqrt(alphaProdT)).sub(modelOutput.mul(Math.sqrt(betaProdT)))
    }
    const denoised = predX0.mul(cOut).add(sample.mul(cSkip))
    let prevSample = denoised
    if (this.timesteps.data.length > 1) {
      const noise = randomNormalTensor(modelOutput.dims, undefined, undefined, undefined, rng);
      prevSample = denoised.mul(Math.sqrt(alphaProdTPrev)).add(noise.mul(Math.sqrt(betaProdTPrev)))
    }
    return [
      prevSample,
      denoised,
    ]
  }
  /**
   * @param {number} numInferenceSteps 
   */
  setTimesteps (numInferenceSteps) {
    this.numInferenceSteps = numInferenceSteps
    if (this.numInferenceSteps > this.config.num_train_timesteps) {
      throw new Error('numInferenceSteps must be less than or equal to num_train_timesteps')
    }
    const lcmOriginSteps = this.config.original_inference_steps
    // LCM Timesteps Setting: Linear Spacing
    const c = Math.floor(this.config.num_train_timesteps / lcmOriginSteps);
    /** @type {number[]} */
    const lcmOriginTimesteps = []
    for (let i = 1; i <= lcmOriginSteps; i++) {
      lcmOriginTimesteps.push(i * c - 1)
    }
    const skippingStep = Math.floor(lcmOriginTimesteps.length / numInferenceSteps)
    /** @type {number[]} */
    const timesteps = []
    for (let i = lcmOriginTimesteps.length - 1; i >= 0; i -= skippingStep) {
      timesteps.push(lcmOriginTimesteps[i])
      if (timesteps.length === numInferenceSteps) {
        break
      }
    }
    this.timesteps = new Tensor(
      new Int32Array(
        timesteps,
      ),
    )
  }
}

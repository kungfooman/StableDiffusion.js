import {getModelJSON                  } from '../hub/index.js';
import {StableDiffusionPipeline       } from './StableDiffusionPipeline.js';
import {StableDiffusionXLPipeline     } from './StableDiffusionXLPipeline.js';
import {LatentConsistencyModelPipeline} from './LatentConsistencyModelPipeline.js';
/** @typedef {import('./common.js'     ).PretrainedOptions  } PretrainedOptions   */
/** @typedef {import('../hub/common.js').GetModelFileOptions} GetModelFileOptions */
export class DiffusionPipeline {
  /**
   * @param {string} modelRepoOrPath 
   * @param {PretrainedOptions} [options] 
   */
  static async fromPretrained (modelRepoOrPath, options) {
    /** @type {GetModelFileOptions} */
    const opts = {
      ...options,
    };
    const index = await getModelJSON(modelRepoOrPath, 'model_index.json', true, opts);
    switch (index['_class_name']) {
      case 'StableDiffusionPipeline':
      case 'OnnxStableDiffusionPipeline':
        return StableDiffusionPipeline.fromPretrained(modelRepoOrPath, options)
      case 'StableDiffusionXLPipeline':
      case 'ORTStableDiffusionXLPipeline':
        return StableDiffusionXLPipeline.fromPretrained(modelRepoOrPath, options)
      case 'LatentConsistencyModelPipeline':
        return LatentConsistencyModelPipeline.fromPretrained(modelRepoOrPath, options)
      default:
        throw new Error(`Unknown pipeline type ${index['_class_name']}`)
    }
  }
}

import { Tensor } from '@xenova/transformers';
/**
 * @param {number} numDiffusionTimesteps 
 * @param {number} maxBeta 
 * @param {'exp'|'cosine'} [alphaTransformType] 
 */
export function betasForAlphaBar (
  numDiffusionTimesteps,
  maxBeta = 0.999,
  alphaTransformType = 'cosine',
) {
  /**
   * @param {number} timeStep 
   */
  function alphaBar (timeStep) {
    if (alphaTransformType === 'cosine') {
      return Math.cos((timeStep + 0.008) / 1.008 * Math.PI / 2) ** 2
    } else if (alphaTransformType === 'exp') {
      return Math.exp(timeStep * -12)
    }

    throw new Error('Unsupported alphaTransformType: ' + alphaTransformType)
  }
  const betas = [];
  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps
    const t2 = (i + 1) / numDiffusionTimesteps
    betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta))
  }
  return new Tensor(betas)
}

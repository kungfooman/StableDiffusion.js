// import 'module-alias/register.js'
import browserCache from './hub/browser.js';
import { setCacheImpl } from './hub/index.js';
export * from './pipelines/StableDiffusionPipeline.js';
export * from './pipelines/StableDiffusionXLPipeline.js';
export * from './pipelines/DiffusionPipeline.js';
export * from './pipelines/common.js';
export * from './hub/index.js';
export { setModelCacheDir } from './hub/browser.js';
setCacheImpl(browserCache);

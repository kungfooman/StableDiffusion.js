// import 'module-alias/register.js'
import {getModelFile      } from './hub/node.js';
import {setCacheImpl      } from './hub/index.js';
export {setModelCacheDir  } from './hub/browser.js';
import {onnxruntimeBackend} from 'onnxruntime-node/dist/backend.js';
import * as ORT from '@aislamov/onnxruntime-web64';
export * from './pipelines/StableDiffusionPipeline.js';
export * from './pipelines/StableDiffusionXLPipeline.js';
export * from './pipelines/DiffusionPipeline.js';
export * from './pipelines/common.js';
export * from './hub/index.js';
const ONNX = ORT.default ?? ORT;
ONNX.registerBackend('cpu', onnxruntimeBackend, 1002);
setCacheImpl(getModelFile);

import 'module-alias/register';
import {getModelFile    } from '../hub/node.js';
import {setCacheImpl    } from '../hub/index.js';
export {setModelCacheDir} from '../hub/browser.js';
setCacheImpl({getModelFile});

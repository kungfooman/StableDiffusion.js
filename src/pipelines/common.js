import {Tensor                    } from '@xenova/transformers';
import {replaceTensors            } from '../util/Tensor.js';
import {getModelFile, getModelJSON} from '../hub/index.js';
import {Session                   } from '../backends/index.js';
/** @typedef {import('../hub/common.js').GetModelFileOptions} GetModelFileOptions */
/**
 * @param {import('onnxruntime-common').InferenceSession} session 
 * @param {Record<string, Tensor>} inputs 
 * @returns {Promise<Record<string, Tensor>>}
 */
export async function sessionRun (session, inputs) {
  // @ts-ignore
  const result = await session.run(inputs)
  return replaceTensors(result)
}

/**
 * @typedef {Object} PretrainedOptions
 * @property {string} [revision] - Optional revision identifier.
 * @property {ProgressCallback} [progressCallback] - Optional callback to track progress.
 */

export const ProgressStatus = {
  Downloading: 'Downloading',
  Ready: 'Ready',
  Error: 'Error',
  EncodingImg2Img: 'EncodingImg2Img',
  EncodingPrompt: 'EncodingPrompt',
  RunningUnet: 'RunningUnet',
  RunningVae: 'RunningVae',
  Done: 'Done',
}

/**
 * @typedef {Object} ProgressDownloadStatus
 * @property {string} file - The file being downloaded.
 * @property {number} size - The total size of the file.
 * @property {number} downloaded - The amount of the file that has been downloaded.
 */

/**
 * @typedef {Object} ProgressCallbackPayload
 * @property {ProgressStatus} status - The current status of the process.
 * @property {ProgressDownloadStatus} [downloadStatus] - Optional status of the download progress.
 * @property {Tensor[]} [images] - Optional array of tensors representing images.
 * @property {string} [statusText] - Optional text describing the current status.
 * @property {number} [unetTotalSteps] - Optional total number of steps for a U-Net process.
 * @property {number} [unetTimestep] - Optional current timestep in a U-Net process.
 */

/**
 * @callback ProgressCallback
 * @param {ProgressCallbackPayload} cb
 * @returns {Promise<void>}
 */

/**
 * @param {ProgressCallbackPayload} payload
 * @returns {string}
 */
function setStatusText (payload) {
  switch (payload.status) {
    case ProgressStatus.Downloading:
      return `Downloading ${payload.downloadStatus.file} (${Math.round(payload.downloadStatus.downloaded / payload.downloadStatus.size * 100)}%)`
    case ProgressStatus.EncodingImg2Img:
      return `Encoding input image`
    case ProgressStatus.EncodingPrompt:
      return `Encoding prompt`
    case ProgressStatus.RunningUnet:
      return `Running UNet (${payload.unetTimestep}/${payload.unetTotalSteps})`
    case ProgressStatus.RunningVae:
      return `Running VAE`
    case ProgressStatus.Done:
      return `Done`
    case ProgressStatus.Ready:
      return `Ready`
  }
  return '';
}
/**
 * @param {ProgressCallback} cb 
 * @param {ProgressCallbackPayload} payload 
 */
export function dispatchProgress (cb, payload) {
  if (!payload.statusText) {
    payload.statusText = setStatusText(payload)
  }
  if (cb) {
    return cb(payload)
  }
}
/**
 * @param {string} modelRepoOrPath 
 * @param {string} filename 
 * @param {GetModelFileOptions} opts 
 * @returns {Promise<Session>}
 */
export async function loadModel (modelRepoOrPath, filename, opts) {
  const model = await getModelFile(modelRepoOrPath, filename, true, opts)
  let weights = await getModelFile(modelRepoOrPath, filename + '_data', false, opts)
  let weightsName = 'model.onnx_data';
  const dirName = filename.split('/')[0]
  if (!weights) {
    weights = await getModelFile(modelRepoOrPath, dirName + '/weights.pb', false, opts)
    weightsName = 'weights.pb'
  }
  const config = await getModelJSON(modelRepoOrPath, dirName + '/config.json', false, opts)
  return Session.create(model, weights, weightsName, config)
}

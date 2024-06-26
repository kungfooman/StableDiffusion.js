import {downloadFile    } from '@huggingface/hub';
import {DbCache         } from './indexed-db.js';
import {pathJoin        } from './common.js';
import {dispatchProgress} from '../pipelines/common.js';
/** @typedef {import('../pipelines/common.js').ProgressCallback} ProgressCallback */
/** @typedef {import('../pipelines/common.js').ProgressStatus} ProgressStatus */
let cacheDir = '';
/**
 * @param {string} dir 
 */
export function setModelCacheDir(dir) {
  cacheDir = dir;
}
/**
 * @param {string} modelRepoOrPath 
 * @param {string} fileName 
 * @param {string} revision 
 */
export function getCacheKey(modelRepoOrPath, fileName, revision) {
  return pathJoin(cacheDir, modelRepoOrPath, revision === 'main' ? '' : revision, fileName);
}
/**
 * @param {string} modelRepoOrPath 
 * @param {string} fileName 
 * @param {boolean} fatal 
 * @param {import('./common.js').GetModelFileOptions} [options] 
 */
export async function getModelFile(modelRepoOrPath, fileName, fatal = true, options = {}) {
  const revision = options.revision || 'main'
  const cachePath = getCacheKey(modelRepoOrPath, fileName, revision)
  const cache = new DbCache()
  await cache.init()
  const cachedData = await cache.retrieveFile(cachePath, options.progressCallback, fileName)
  if (cachedData) {
    if (options.returnText) {
      const decoder = new TextDecoder('utf-8')
      return decoder.decode(cachedData.file)
    }
    return cachedData.file
  }
  /** @type {Response|null|undefined} */
  let response;
  // now local cache
  if (cacheDir) {
    response = await fetch(cachePath)
    // create-react-app will return 200 with HTML for missing files
    if (!response || !response.body || response.status !== 200 || response.headers.get('content-type')?.startsWith('text/html')) {
      response = null
    }
  }
  try {
    // now try the hub
    if (!response) {
      response = await downloadFile({ repo: modelRepoOrPath, path: fileName, revision })
    }
    // read response
    if (!response || !response.body || response.status !== 200 || response.headers.get('content-type')?.startsWith('text/html')) {
      throw new Error(`Error downloading ${fileName}`)
    }
    const buffer = await readResponseToBuffer(response, options.progressCallback, fileName)
    await cache.storeFile(buffer, cachePath)
    if (options.returnText) {
      const decoder = new TextDecoder('utf-8')
      return decoder.decode(buffer)
    }
    return buffer
  } catch (e) {
    if (!fatal) {
      return null
    }
    throw e
  }
}
/**
 * @param {Response} response 
 * @param {ProgressCallback} progressCallback 
 * @param {string} displayName 
 * @returns {Promise<ArrayBuffer>}
 */
function readResponseToBuffer (response, progressCallback, displayName) {
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    return response.arrayBuffer()
  }
  /** @type {ArrayBuffer} */
  let buffer;
  const contentLengthNum = parseInt(contentLength, 10)
  if (contentLengthNum > 2 * 1024 * 1024 * 1024) {
    // @ts-ignore
    const memory = new WebAssembly.Memory({ initial: Math.ceil(contentLengthNum / 65536), index: 'i64' })
    buffer = memory.buffer
  } else {
    buffer = new ArrayBuffer(contentLengthNum)
  }
  let offset = 0;
  return new Promise((resolve, reject) => {
    const reader = response.body.getReader()
    async function pump() {
      const { done, value } = await reader.read()
      if (done) {
        return resolve(buffer)
      }
      const chunk = new Uint8Array(buffer, offset, value.byteLength)
      chunk.set(new Uint8Array(value))
      offset += value.byteLength
      await dispatchProgress(progressCallback, {
        status: ProgressStatus.Downloading,
        downloadStatus: {
          file: displayName,
          size: contentLengthNum,
          downloaded: offset,
        }
      })
      return pump();
    }
    pump().catch(reject)
  })
}
export default {
  getModelFile,
}

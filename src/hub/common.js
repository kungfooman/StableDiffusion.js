/**
 * @typedef {object} GetModelFileOptions
 * @property {import('../pipelines/common.js').ProgressCallback} [progressCallback] 
 * @property {string} [revision]
 * @property {boolean} [returnText]
 */
/**
 * @typedef {object} CacheImpl
 * @property {(modelRepoOrPath: string, fileName: string, fatal?: boolean, options?: GetModelFileOptions): Promise<string|ArrayBuffer>} getModelFile
 */
/**
 * @param  {...string} parts 
 * @returns {string}
 */
export function pathJoin (...parts) {
  // https://stackoverflow.com/a/55142565
  parts = parts.map((part, index) => {
    if (index) {
      part = part.replace(/^\//, '')
    }
    if (index !== parts.length - 1) {
      part = part.replace(/\/$/, '')
    }
    return part
  })
  return parts.filter(p => p !== '').join('/')
}

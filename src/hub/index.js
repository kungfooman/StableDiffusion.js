/** @typedef {import('./common.js').CacheImpl          } CacheImpl           */
/** @typedef {import('./common.js').GetModelFileOptions} GetModelFileOptions */
/** @type {CacheImpl} */
let cacheImpl = null;
/**
 * @param {CacheImpl} impl 
 */
export function setCacheImpl(impl) {
  cacheImpl = impl;
}
/**
 * @param {string} modelRepoOrPath 
 * @param {string} fileName 
 * @param {boolean} [fatal] 
 * @param {GetModelFileOptions} [options] 
 */
export async function getModelFile (modelRepoOrPath, fileName, fatal = true, options = {}) {
  return cacheImpl.getModelFile(modelRepoOrPath, fileName, fatal, options)
}
/**
 * @param {string} modelPath 
 * @param {string} fileName 
 * @param {boolean} fatal 
 * @param {GetModelFileOptions} options 
 * @returns {Promise<string>}
 */
export function getModelTextFile (modelPath, fileName, fatal, options) {
  return getModelFile(modelPath, fileName, fatal, { ...options, returnText: true });
}
/**
 * @param {string} modelPath 
 * @param {string} fileName 
 * @param {boolean} [fatal] 
 * @param {GetModelFileOptions} [options] 
 */
export async function getModelJSON (modelPath, fileName, fatal = true, options = {}) {
  const jsonData = await getModelTextFile(modelPath, fileName, fatal, options);
  return JSON.parse(jsonData);
}

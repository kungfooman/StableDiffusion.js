/** @typedef {import('./common.js').CacheImpl          } CacheImpl           */
/** @typedef {import('./common.js').GetModelFileOptions} GetModelFileOptions */
/** @type {CacheImpl} */
let getModelFile = null;
/**
 * @param {CacheImpl} impl 
 */
function setCacheImpl(impl) {
  getModelFile = impl;
}
/**
 * @param {string} modelPath 
 * @param {string} fileName 
 * @param {boolean} fatal 
 * @param {GetModelFileOptions} options 
 * @returns {Promise<string>}
 */
function getModelTextFile(modelPath, fileName, fatal, options) {
  return getModelFile(modelPath, fileName, fatal, { ...options, returnText: true });
}
/**
 * @param {string} modelPath 
 * @param {string} fileName 
 * @param {boolean} [fatal] 
 * @param {GetModelFileOptions} [options] 
 */
async function getModelJSON(modelPath, fileName, fatal = true, options = {}) {
  const jsonData = await getModelTextFile(modelPath, fileName, fatal, options);
  return JSON.parse(jsonData);
}
export {setCacheImpl, getModelTextFile, getModelJSON, getModelFile};

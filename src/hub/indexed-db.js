import {openDB                          } from 'idb';
import {dispatchProgress, ProgressStatus} from '../pipelines/common.js';
/** @typedef {import('../pipelines/common.js').ProgressCallback} ProgressCallback */
/**
 * @typedef {object} FileMetadata
 * @property {number} chunks
 * @property {number} chunkLength
 * @property {number} totalLength
 * @property {number} chunk
 * @property {ArrayBuffer} file
 */
const DEFAULT_CHUNK_LENGTH = 1024 * 1024 * 512
export class DbCache {
  dbName = 'diffusers-cache';
  dbVersion = 1;
  /** @type {import('idb').IDBPDatabase} */
  db;
  init = async () => {
    const openRequest = await openDB(this.dbName, this.dbVersion, {
      upgrade (db) {
        if (!db.objectStoreNames.contains('files')) {
          db.createObjectStore('files')
        }
      },
    })

    this.db = openRequest
  }
  /**
   * @param {ArrayBuffer} file 
   * @param {string} name 
   * @param {number} [chunkLength] 
   */
  storeFile = async (file, name, chunkLength = DEFAULT_CHUNK_LENGTH) => {
    const transaction = this.db.transaction(['files'], 'readwrite')
    const store = transaction.objectStore('files')

    const chunks = Math.ceil(file.byteLength / chunkLength)

    const fileMetadata = {
      chunks,
      chunkLength,
      totalLength: file.byteLength,
    }

    for (let i = 0; i < chunks; i++) {
      const chunk = file.slice(i * chunkLength, (i + 1) * chunkLength)
      const nameSuffix = i > 0 ? `-${i}` : ''
      const thisChunkLength = chunk.byteLength
      await store.put({ ...fileMetadata, chunkLength: thisChunkLength, file: chunk, chunk: i }, `${name}${nameSuffix}`)
    }
    await transaction.done
  }
  /**
   * 
   * @param {string} filename 
   * @param {ProgressCallback} progressCallback 
   * @param {string} displayName 
   * @returns {Promise<FileMetadata | null>}
   */
  retrieveFile = async (filename, progressCallback, displayName) => {
    const transaction = this.db.transaction(['files'], 'readonly')
    const store = transaction.objectStore('files');
    /** @type {FileMetadata} */
    const request = await store.get(filename);
    if (!request) {
      return null;
    }

    if (request.chunks === 1) {
      return request
    }
    /** @type {ArrayBuffer} */
    let buffer;
    if (request.totalLength > 2 * 1024 * 1024 * 1024) {
      // @ts-ignore
      const memory = new WebAssembly.Memory({ initial: Math.ceil(request.totalLength / 65536), index: 'i64' })
      buffer = memory.buffer
    } else {
      buffer = new ArrayBuffer(request.totalLength)
    }

    const baseChunkLength = request.chunkLength
    let view = new Uint8Array(buffer, 0, request.chunkLength)
    view.set(new Uint8Array(request.file))

    await dispatchProgress(progressCallback, {
      status: ProgressStatus.Downloading,
      downloadStatus: {
        file: displayName,
        size: request.totalLength,
        downloaded: request.chunkLength,
      }
    })

    for (let i = 1; i < request.chunks; i++) {
      /** @type {FileMetadata} */
      const file = await store.get(`${filename}-${i}`);
      view = new Uint8Array(buffer, i * baseChunkLength, file.file.byteLength)
      view.set(new Uint8Array(file.file));
      await dispatchProgress(progressCallback, {
        status: ProgressStatus.Downloading,
        downloadStatus: {
          file: displayName,
          size: request.totalLength,
          downloaded: i * baseChunkLength + file.file.byteLength
        }
      })
    }
    await transaction.done

    return { ...request, file: buffer }
  }
}

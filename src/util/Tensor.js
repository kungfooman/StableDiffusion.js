import {Tensor    } from '@xenova/transformers';
import {seedrandom} from '../seedrandom.js';
Tensor.prototype.reverse = function () {
  return new Tensor(this.type, this.data.reverse(), this.dims.slice());
}
/**
 * @param {Tensor} value 
 */
Tensor.prototype.sub = function (value) {
  return this.clone().sub_(value)
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.sub_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}
/**
 * @param {Tensor} value 
 */
Tensor.prototype.add = function (value) {
  return this.clone().add_(value)
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.add_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}
/**
 * @param {number} dim 
 */
Tensor.prototype.cumprod = function (dim) {
  return this.clone().cumprod_(dim)
}
/**
 * @param {number} dim 
 */
Tensor.prototype.cumprod_ = function (dim) {
  const newDims = this.dims.slice();
  // const newStrides = this.strides.slice();
  if (dim === undefined) {
    dim = this.dims.length - 1
  }
  if (dim < 0 || dim >= this.dims.length) {
    throw new Error(`Invalid dimension: ${dim}`)
  }
  const size = newDims[dim]
  for (let i = 1; i < size; ++i) {
    for (let j = 0; j < this.data.length / size; ++j) {
      const index = j * size + i
      this.data[index] *= this.data[index - 1]
    }
  }
  // newDims[dim] = 1;
  // newStrides[dim] = 0;
  return this
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.mul = function (value) {
  return this.clone().mul_(value)
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.mul_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.div = function (value) {
  return this.clone().div_(value)
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.div_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.pow = function (value) {
  return this.clone().pow_(value)
}
/**
 * @param {Tensor|number} value 
 */
Tensor.prototype.pow_ = function (value) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value)
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value.data[i])
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}
Tensor.prototype.round = function () {
  return this.clone().round_()
}
Tensor.prototype.reshape = function (dims) {
  return new Tensor(this.type, this.data, dims);
}
// from C:\xampp\htdocs\diffusers.js\node_modules\@aislamov\onnxruntime-web64\node_modules\onnxruntime-common\dist\ort-common.js
Tensor.prototype.toImageData = function (options) {
  const pixels2DContext = document.createElement('canvas').getContext('2d');
  let image;
  if (pixels2DContext != null) {
      // Default values for height and width & format
      let width;
      let height;
      let channels;
      if ((options === null || options === void 0 ? void 0 : options.tensorLayout) !== undefined && options.tensorLayout === 'NHWC') {
          width = this.dims[2];
          height = this.dims[1];
          channels = this.dims[3];
      }
      else { // Default layout is NCWH
          width = this.dims[3];
          height = this.dims[2];
          channels = this.dims[1];
      }
      const inputformat = options !== undefined ? (options.format !== undefined ? options.format : 'RGB') : 'RGB';
      const norm = options === null || options === void 0 ? void 0 : options.norm;
      let normMean;
      let normBias;
      if (norm === undefined || norm.mean === undefined) {
          normMean = [255, 255, 255, 255];
      }
      else {
          if (typeof (norm.mean) === 'number') {
              normMean = [norm.mean, norm.mean, norm.mean, norm.mean];
          }
          else {
              normMean = [norm.mean[0], norm.mean[1], norm.mean[2], 255];
              if (norm.mean[3] !== undefined) {
                  normMean[3] = norm.mean[3];
              }
          }
      }
      if (norm === undefined || norm.bias === undefined) {
          normBias = [0, 0, 0, 0];
      }
      else {
          if (typeof (norm.bias) === 'number') {
              normBias = [norm.bias, norm.bias, norm.bias, norm.bias];
          }
          else {
              normBias = [norm.bias[0], norm.bias[1], norm.bias[2], 0];
              if (norm.bias[3] !== undefined) {
                  normBias[3] = norm.bias[3];
              }
          }
      }
      const stride = height * width;
      if (options !== undefined) {
          if (options.height !== undefined && options.height !== height) {
              throw new Error('Image output config height doesn\'t match tensor height');
          }
          if (options.width !== undefined && options.width !== width) {
              throw new Error('Image output config width doesn\'t match tensor width');
          }
          if (options.format !== undefined && (channels === 4 && options.format !== 'RGBA') ||
              (channels === 3 && (options.format !== 'RGB' && options.format !== 'BGR'))) {
              throw new Error('Tensor format doesn\'t match input tensor dims');
          }
      }
      // Default pointer assignments
      const step = 4;
      let rImagePointer = 0, gImagePointer = 1, bImagePointer = 2, aImagePointer = 3;
      let rTensorPointer = 0, gTensorPointer = stride, bTensorPointer = stride * 2, aTensorPointer = -1;
      // Updating the pointer assignments based on the input image format
      if (inputformat === 'RGBA') {
          rTensorPointer = 0;
          gTensorPointer = stride;
          bTensorPointer = stride * 2;
          aTensorPointer = stride * 3;
      }
      else if (inputformat === 'RGB') {
          rTensorPointer = 0;
          gTensorPointer = stride;
          bTensorPointer = stride * 2;
      }
      else if (inputformat === 'RBG') {
          rTensorPointer = 0;
          bTensorPointer = stride;
          gTensorPointer = stride * 2;
      }
      image = pixels2DContext.createImageData(width, height);
      for (let i = 0; i < height * width; rImagePointer += step, gImagePointer += step, bImagePointer += step, aImagePointer += step, i++) {
          image.data[rImagePointer] = (this.data[rTensorPointer++] - normBias[0]) * normMean[0]; // R value
          image.data[gImagePointer] = (this.data[gTensorPointer++] - normBias[1]) * normMean[1]; // G value
          image.data[bImagePointer] = (this.data[bTensorPointer++] - normBias[2]) * normMean[2]; // B value
          image.data[aImagePointer] = aTensorPointer === -1 ?
              255 :
              (this.data[aTensorPointer++] - normBias[3]) * normMean[3]; // A value
      }
  }
  else {
      throw new Error('Can not access image data');
  }
  return image;
}
Tensor.prototype.round_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.round(this.data[i])
  }
  return this
}
/**
 * @param {number|number[]} reps 
 */
Tensor.prototype.tile = function (reps) {
  return this.clone().tile_(reps)
}
/**
 * @param {number|number[]} reps 
 */
Tensor.prototype.tile_ = function (reps) {
  if (typeof reps === 'number') {
    reps = [reps]
  }
  if (reps.length < this.dims.length) {
    throw new Error('Invalid number of repetitions')
  }
  const newDims = []
  const newStrides = []
  for (let i = 0; i < this.dims.length; ++i) {
    newDims.push(this.dims[i] * reps[i])
    newStrides.push(this.strides[i])
  }
  const newData = new this.data.constructor(newDims.reduce((a, b) => a * b))
  for (let i = 0; i < newData.length; ++i) {
    let index = 0
    for (let j = 0; j < this.dims.length; ++j) {
      index += Math.floor(i / newDims[j]) * this.strides[j]
    }
    newData[i] = this.data[index]
  }
  return new Tensor(this.type, newData, newDims)
}
/**
 * @param {number} min 
 * @param {number} max 
 */
Tensor.prototype.clipByValue = function (min, max) {
  return this.clone().clipByValue_(min, max)
}
/**
 * @param {number} min 
 * @param {number} max 
 */
Tensor.prototype.clipByValue_ = function (min, max) {
  if (max < min) {
    throw new Error('Invalid arguments')
  }
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.min(Math.max(this.data[i], min), max)
  }
  return this
}
Tensor.prototype.exp = function () {
  return this.clone().exp_()
}
Tensor.prototype.exp_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.exp(this.data[i])
  }
  return this
}
Tensor.prototype.sin = function () {
  return this.clone().sin_()
}
Tensor.prototype.sin_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.sin(this.data[i])
  }
  return this
}
Tensor.prototype.cos = function () {
  return this.clone().cos_()
}
Tensor.prototype.cos_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.cos(this.data[i])
  }
  return this
}
Tensor.prototype.location = 'cpu';
/**
 * @param {number} start 
 * @param {number} end 
 * @param {number} step 
 * @param {string} type 
 */
export function range(start, end, step = 1, type = 'float32') {
  const data = []
  for (let i = start; i < end; i += step) {
    data.push(i)
  }
  return new Tensor(type, data, [data.length])
}
/**
 * @param {number} start 
 * @param {number} end 
 * @param {number} step 
 * @param {string} type 
 */
export function linspace(start, end, num, type = 'float32') {
  const arr = []
  const step = (end - start) / (num - 1)
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i)
  }
  return new Tensor(type, arr, [num])
}
/**
 * 
 * @param {seedrandom.PRNG} rng 
 * @returns 
 */
function randomNormal(rng) {
  let u = 0; let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  return num
}
/**
 * @param {number} num 
 * @param {string} type 
 */
export function scalarTensor (num, type = 'float32') {
  return new Tensor(type, new Float32Array([num]), [1])
}
/**
 * @param {number[]} shape 
 * @param {number} mean 
 * @param {number} std 
 * @param {string} type 
 * @param {() => number} rng
 */
export function randomNormalTensor (shape, mean = 0, std = 1, type = 'float32', rng) {
  const data = [];
  for (let i = 0; i < shape.reduce((a, b) => a * b); i++) {
    data.push(randomNormal(rng) * std + mean);
  }
  return new Tensor(type, data, shape);
}
/**
 * Concatenates an array of tensors along the 0th dimension.
 *
 * @param {Tensor[]} tensors The array of tensors to concatenate.
 * @param {number} [axis]
 * @returns {Tensor} The concatenated tensor.
 */
export function cat(tensors, axis = 0) {
  if (tensors.length === 0) {
    throw new Error('No tensors provided.')
  }
  // Handle negative axis by converting it to its positive counterpart
  if (axis < 0) {
    axis = tensors[0].dims.length + axis;
  }
  const tensorType = tensors[0].type
  const tensorShape = [...tensors[0].dims]
  // Ensure all tensors have the same shape except for the concatenation axis
  for (const t of tensors) {
    for (let i = 0; i < tensorShape.length; i++) {
      if (i !== axis && tensorShape[i] !== t.dims[i]) {
        throw new Error('Tensor dimensions must match for concatenation, except along the specified axis.');
      }
    }
  }
  // Calculate the size of the concatenated tensor along the specified axis
  tensorShape[axis] = tensors.reduce((sum, t) => sum + t.dims[axis], 0);
  // Calculate total size to allocate
  const total = tensorShape.reduce((product, size) => product * size, 1);
  // Create output tensor of same type as the first tensor
  const data = new tensors[0].data.constructor(total);
  let offset = 0;
  for (const t of tensors) {
    const n = t.dims[axis];
    // Size of each slice along the axis
    const copySize = t.data.length / n;
    for (let i=0; i<n; i++) {
      const sourceStart = i * copySize
      const sourceEnd = sourceStart + copySize
      data.set(t.data.slice(sourceStart, sourceEnd), offset)
      offset += copySize
    }
  }
  return new Tensor(tensorType, data, tensorShape);
}
/**
 * Convert ONNX Tensors with our custom Tensor class to support additional functions.
 *
 * @param {Record<string, import('onnxruntime-common').Tensor>} modelRunResult 
 * @returns {Record<string, Tensor>}
 */
export function replaceTensors(modelRunResult) {
  /** @type {Record<string, Tensor>} */
  const result = {};
  for (const prop in modelRunResult) {
    const {type, data, dims} = modelRunResult[prop];
    if (dims) {
      result[prop] = new Tensor(type, data, dims);
    }
  }
  return result;
}

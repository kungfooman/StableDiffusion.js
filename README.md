# diffusers.js library for running diffusion models on GPU/WebGPU

See demo here https://islamov.ai/stable-diffusion-webgpu/

## Installation

```bash
git clone https://github.com/kungfooman/StableDiffusion.js/
cd StableDiffusion.js
npm i && cd examples/react && npm i
cd ../..
# You only need this for Linux, on Windows use Chrome Canary:
./start-chrome.sh
```

## Usage

Browser (see examples/react)
```js
import { DiffusionPipeline } from '@aislamov/diffusers.js'

const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx')
const images = pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 30,
})

const canvas = document.getElementById('canvas')
const data = await images[0].toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
canvas.getContext('2d').putImageData(data, 0, 0);
```

Node.js (see examples/node)
```js
import { DiffusionPipeline } from '@aislamov/diffusers.js'
import { PNG } from 'pngjs'

const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx')
const images = pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 30,
})

const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
p.data = Buffer.from(data.data)
p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
  console.log('Image saved as output.png');
})
```

'aislamov/stable-diffusion-2-1-base-onnx' model is optimized for GPU and will fail to load without CUDA/DML/WebGPU support. Please use 'cpu' revision on a machine without GPU.
```js
const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx', { revision: 'cpu' })
```

## Running examples
Browser/React
```bash
npm install && npm run build
cd examples/react && npm install
npm run start
```

Node
```bash
npm install && npm run build
cd examples/node && npm install
node src/txt2img.mjs --prompt "a dog" --rev cpu
```

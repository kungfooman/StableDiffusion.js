import React, { useEffect, useRef, useState } from 'react'
import './App.css';
import {
  DiffusionPipeline,
  ProgressCallback,
  ProgressCallbackPayload,
  setModelCacheDir,
  StableDiffusionPipeline,
  StableDiffusionXLPipeline
} from '@aislamov/diffusers.js'
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { Checkbox, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import { FormControlLabel } from '@mui/material';
import { BrowserFeatures, hasFp16 } from './components/BrowserFeatures'
import { FAQ } from './components/FAQ'
import { Tensor } from '@xenova/transformers'

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

/**
 * @typedef {object} SelectedPipeline
 * @property {string} name
 * @property {string} repo
 * @property {string} revision
 * @property {boolean} fp16
 * @property {number} steps
 * @property {boolean} hasImg2Img
 */

const pipelines = [
  {
    name: 'LCM Dreamshaper FP16 (2.2GB)',
    repo: 'aislamov/lcm-dreamshaper-v7-onnx',
    revision: 'main',
    fp16: true,
    width: 768,
    height: 768,
    steps: 8,
    hasImg2Img: false,
  },
  // {
  //   name: 'LCM Dreamshaper FP32 (4.2GB)',
  //   repo: 'aislamov/lcm-dreamshaper-v7-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 768,
  //   height: 768,
  //   steps: 8,
  // },
  {
    name: 'StableDiffusion 2.1 Base FP16 (2.6GB)',
    repo: 'aislamov/stable-diffusion-2-1-base-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 20,
    hasImg2Img: true,
  },
  // {
  //   name: 'StableDiffusion 2.1 Base FP32 (5.1GB)',
  //   repo: 'aislamov/stable-diffusion-2-1-base-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 512,
  //   height: 512,
  //   steps: 20,
  // },
]
console.log("wheeeew 123");
function App() {
  const [hasF16, setHasF16] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState(pipelines[0]);
  /** @type {ReturnType<typeof useState<'none'|'loading'|'ready'|'inferencing'>>} */
  const [modelState, setModelState] = useState('none');
  const [prompt, setPrompt] = useState('An astronaut riding a horse');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState('');
  const [status, setStatus] = useState('Ready');
  /** @type {ReturnType<typeof useState<StableDiffusionXLPipeline|StableDiffusionPipeline|null>>} */
  const pipeline = useRef(null);
  const [img2img, setImg2Img] = useState(false);
  /** @type {ReturnType<typeof useState<Float32Array|undefined>>} */
  const [inputImage, setInputImage] = useState();
  const [strength, setStrength] = useState(0.8);
  const [runVaeOnEachStep, setRunVaeOnEachStep] = useState(false);
  useEffect(() => {
    setModelCacheDir('models')
    hasFp16().then(v => {
      setHasF16(v)
      if (v === false) {
        setSelectedPipeline(pipelines.find(p => p.fp16 === false))
      }
    })
  }, [])

  useEffect(() => {
    setInferenceSteps(selectedPipeline?.steps || 20)
  }, [selectedPipeline])

  /**
   * @param {Tensor} image 
   */
  const drawImage = async (image) => {
    const canvas = document.getElementById('canvas');
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error("No canvas");
    }
    // @ts-ignore
    const data = await image.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
    canvas.getContext('2d').putImageData(data, 0, 0);
  }

  /**
   * @param {ProgressCallbackPayload} info 
   */
  const progressCallback = async (info) => {
    if (info.statusText) {
      setStatus(info.statusText)
    }

    if (info.images) {
      // @ts-ignore
      await drawImage(info.images[0])
    }
  }

  const loadModel = async () => {
    if (!selectedPipeline) {
      return
    }
    setModelState('loading')
    try {
      if (pipeline.current) {
        // @ts-ignore
        pipeline.current.release()
      }
      pipeline.current = await DiffusionPipeline.fromPretrained(
        selectedPipeline.repo,
        {
          revision: selectedPipeline?.revision,
          progressCallback
        }
      )
      setModelState('ready')
    } catch (e) {
      alert(e)
      console.error(e)
    }
  }

  /**
   * @param {Uint8ClampedArray} d 
   * @returns {any}
   */
  function getRgbData(d) {
    /** @type {any} */
    let rgbData = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for(let i = 0; i < d.length; i += 4) {
        let x = (i/4) % 512;
        let y = Math.floor((i/4) / 512)
        if(!rgbData[0][y]) rgbData[0][y] = [];
        if(!rgbData[1][y]) rgbData[1][y] = [];
        if(!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = (d[i+0]/255) * 2 - 1;
        rgbData[1][y][x] = (d[i+1]/255) * 2 - 1;
        rgbData[2][y][x] = (d[i+2]/255) * 2 - 1;
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }

  function uploadImage(e) {
    if(!e.target.files[0]) {
      // No image uploaded
      return;
    }

    const uploadedImage = new Image(512, 512); // resize image to 512, 512
    const reader = new FileReader();
    // On file read loadend
    reader.addEventListener('loadend', function(file) {
      // On image load
      uploadedImage.addEventListener('load', function() {
        const imageCanvas = document.createElement('canvas');
        imageCanvas.width = uploadedImage.width;
        imageCanvas.height = uploadedImage.height;
        const imgCtx = imageCanvas.getContext('2d');
        // todo test if CanvasRenderingContext2D
        imgCtx.drawImage(uploadedImage, 0, 0, uploadedImage.width, uploadedImage.height);
        const imageData = imgCtx.getImageData(0, 0, uploadedImage.width, uploadedImage.height).data;

        const rgb_array = getRgbData(imageData);
        setInputImage(rgb_array);
      });
      uploadedImage.src = file.target.result;
    });
    reader.readAsDataURL(e.target.files[0]);
  }

  const runInference = async () => {
    if (!pipeline.current) {
      return
    }
    setModelState('inferencing')

    const images = await pipeline.current.run({
      prompt: prompt,
      negativePrompt: negativePrompt,
      numInferenceSteps: inferenceSteps,
      guidanceScale: guidanceScale,
      seed: seed,
      width: 512,
      height: 512,
      runVaeOnEachStep,
      progressCallback,
      img2imgFlag: img2img,
      inputImage: inputImage,
      strength: strength
    })
    await drawImage(images[0])
    setModelState('ready')
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <BrowserFeatures />
        <Stack alignItems={'center'}>
          <p>Built with <a href={"https://github.com/dakenf/diffusers.js"} target={"_blank"}>diffusers.js</a></p>
        </Stack>
        <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setPrompt(e.target.value)}
                  value={prompt}
                />
                <TextField
                  label="Negative Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  value={negativePrompt}
                />
                <TextField
                  label="Number of inference steps (Because of PNDM Scheduler, it will be i+1)"
                  variant="standard"
                  type='number'
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                <TextField
                  label="Guidance Scale. Controls how similar the generated image will be to the prompt."
                  variant="standard"
                  type='number'
                  InputProps={{ inputProps: { min: 1, max: 20, step: 0.5 } }}
                  disabled={modelState != 'ready'}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                  value={guidanceScale}
                />
                <TextField
                  label="Seed (Creates initial random noise)"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setSeed(e.target.value)}
                  value={seed}
                />
                {selectedPipeline?.hasImg2Img &&
                  (
                    <>
                      <FormControlLabel
                        label="Check if you want to use the Img2Img pipeline"
                        control={<Checkbox
                          disabled={modelState != 'ready'}
                          onChange={(e) => setImg2Img(e.target.checked)}
                          checked={img2img}
                        />}
                      />
                      <label htmlFor="upload_image">Upload Image for Img2Img Pipeline:</label>
                      <TextField
                        id="upload_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={!img2img}
                        onChange={(e) => uploadImage(e)}
                      />
                      <TextField
                        label="Strength (Noise to add to input image). Value ranges from 0 to 1"
                        variant="standard"
                        type='number'
                        InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
                        disabled={!img2img}
                        onChange={(e) => setStrength(parseFloat(e.target.value))}
                        value={strength}
                      />
                    </>
                )}
                <FormControlLabel
                  label="Check if you want to run VAE after each step"
                  control={<Checkbox
                    disabled={modelState != 'ready'}
                    onChange={(e) => setRunVaeOnEachStep(e.target.checked)}
                    checked={runVaeOnEachStep}
                  />}
                />
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                <p>Press the button below to download model. It will be stored in your browser cache.</p>
                <p>All settings above will become editable once model is downloaded.</p>
                <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'}>Load model</Button>
                <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'}>Run</Button>
                <p>{status}</p>
                <p><a href={'https://github.com/dakenf'}>Follow me on GitHub</a></p>
              </Stack>

            </Grid>
            <Grid item xs={6}>
              <canvas id={'canvas'} width={512} height={512} style={{ border: '1px dashed #ccc'}} />
            </Grid>
          </Grid>
        </Box>
        <Divider/>
        <FAQ />
      </Container>
    </ThemeProvider>
  );
}

export default App;

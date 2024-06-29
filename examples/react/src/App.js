import {Fragment, Component} from 'react';
import {
  DiffusionPipeline,
  //ProgressCallback,
  //ProgressCallbackPayload,
  setModelCacheDir,
  //StableDiffusionPipeline,
  //StableDiffusionXLPipeline
} from '@aislamov/diffusers.js'
import {CssBaseline             } from './mini-ui.js';
import {Box                     } from './mini-ui.js';
import {Container               } from './mini-ui.js';
import {ThemeProvider           } from './mini-ui.js';
import {createTheme             } from './mini-ui.js';
import {Stack                   } from './mini-ui.js';
import {Grid                    } from './mini-ui.js';
import {TextField               } from './mini-ui.js';
import {Divider                 } from './mini-ui.js';
import {Checkbox                } from './mini-ui.js';
import {FormControl             } from './mini-ui.js';
import {InputLabel              } from './mini-ui.js';
import {MenuItem                } from './mini-ui.js';
import {Select                  } from './mini-ui.js';
import {Row                     } from './mini-ui.js';
import {FormControlLabel        } from './mini-ui.js';
import {BrowserFeatures, hasFp16} from './components/BrowserFeatures.js';
import {jsx                     } from './jsx.js';
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
 * @property {number} inferenceSteps
 * @property {boolean} hasImg2Img
 * @property {boolean} sdV1
 */
/** @type {SelectedPipeline[]} */
const pipelines = [
  {
    name: 'LCM Dreamshaper FP16 (2.2GB)',
    repo: 'aislamov/lcm-dreamshaper-v7-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 768,
    inferenceSteps: 8,
    hasImg2Img: false,
    guidanceScale: 7.5,
  },
  // {
  //   name: 'LCM Dreamshaper FP32 (4.2GB)',
  //   repo: 'aislamov/lcm-dreamshaper-v7-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 768,
  //   height: 768,
  //   inferenceSteps: 8,
  //    guidanceScale: 7.5,
  // },
  {
    name: 'StableDiffusion 2.1 Base FP16 (2.6GB)',
    repo: 'aislamov/stable-diffusion-2-1-base-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    inferenceSteps: 20,
    hasImg2Img: true,
    guidanceScale: 7.5,
  },
  // {
  //   name: 'StableDiffusion 2.1 Base FP32 (5.1GB)',
  //   repo: 'aislamov/stable-diffusion-2-1-base-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 512,
  //   height: 512,
  //   inferenceSteps: 20,
  //    guidanceScale: 7.5,
  // },
  /*
  wasm-utils.ts:61 Uncaught (in promise) 
  Error: failed to call OrtRun(). ERROR_CODE: 2, ERROR_MESSAGE: Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(int64))
      at t.checkLastError (wasm-utils.ts:61:11)
      at t.run (wasm-core-impl.ts:313:9)
      at async t.OnnxruntimeWebAssemblySessionHandler.run (session-handler.ts:120:11)
      at async a.run (inference-session-impl.js:94:15)
      at async _Session.run (index.esm.js:583:20)
      at async _StableDiffusionPipeline.run (index.esm.js:1492:21)
      at async runInference (App.js:222:20)
  */
      {
        name: 'TheyCallMeHex/Deliberate-V3-ONNX',
        repo: 'TheyCallMeHex/Deliberate-V3-ONNX',
        revision: 'main',
        fp16: true,
        width: 768,
        height: 768,
        inferenceSteps: 8,
        hasImg2Img: false,
        sdV1: true,
        guidanceScale: 7.5,
      },
]
/**
 * @typedef {object} Props
 */
/**
 * @typedef {object} State
 * @property {boolean} hasF16 - Using half-floats. F16 = a floating point system
 * @property {typeof pipelines[0]} selectedPipeline
 * @property {'none' | 'loading' | 'ready' | 'inferencing'} modelState - Inferencing == "calculating image output"
 * @property {string} prompt - The prompt.
 * @property {string} negativePrompt - The negative prompt.
 * @property {number} inferenceSteps - Number of inference steps.
 * @property {number} width - The width.
 * @property {number} height - The height.
 * @property {number} guidanceScale - The guidance scale.
 * @property {string} seed - The seed.
 * @property {string} modelStatus - Status for downloading model, displayed near <select>
 * @property {string} inferenceStatus - Status for inferences, displayed near <canvas>
 * @property {boolean} img2img - Is this using an image conversion model; true = using an image conversion model
 * @property {Float32Array|undefined} inputImage 
 * @property {number} strength
 * @property {boolean} runVaeOnEachStep
 * @property {number} now
 */
/** @type {typeof Component<Props, State>} */
const TypedComponent = Component;
class App extends TypedComponent {
  /** @type {SelectedPipeline} */
  static pipeline;
  /** @type {State} */
  state = {
    hasF16: false,
    selectedPipeline: pipelines[0],
    modelState: 'none',
    prompt: 'An astronaut riding a horse',
    negativePrompt: '',
    inferenceSteps: pipelines[0].inferenceSteps,
    width: pipelines[0].width,
    height: pipelines[0].height,
    guidanceScale: pipelines[0].guidanceScale,
    seed: '',
    generatedSeed: '',
    modelStatus: '',
    inferenceStatus: '',
    img2img: false,
    inputImage: undefined,
    strength: 0.8,
    runVaeOnEachStep: false,
    now: Date.now(),
  }
  constructor(props) {
    super(props);
    this.canvas = React.createRef();
  }
  setSelectedPipeline(selectedPipeline) {
    this.mergeState({selectedPipeline});
  }
  componentDidMount() {
    App.instance = this;
    // this.onLayoutChange = this.onLayoutChange.bind(this);
    // window.addEventListener("resize", this.onLayoutChange);
    // window.addEventListener("orientationchange", this.onLayoutChange);
    setModelCacheDir('models');
    /*
    hasFp16().then(v => {
      setHasF16(v)
      if (v === false) {
        setSelectedPipeline(pipelines.find(p => p.fp16 === false))
      }
    });
    */
  }
  /*
  componentWillUnmount() {
      window.removeEventListener("resize", this.onLayoutChange);
      window.removeEventListener("orientationchange", this.onLayoutChange);
  }
  */
  /**
   * @param {Partial<State>} state - The partial state to update.
   */
  mergeState(state) {
    // New state is always calculated from the current state,
    // avoiding any potential issues with asynchronous updates.
    this.setState(prevState => ({ ...prevState, ...state }));
  }
  setInferenceStatus(inferenceStatus) {
    this.mergeState({inferenceStatus});
  }
  /**
   * @param {Tensor} image 
   */
  async drawImage(image) {
    const canvas = this.canvas.current;
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error('No canvas');
    }
    App.lastImage = image;
    App.lastCanvas = canvas;
    const data = await image.toImageData({tensorLayout: 'NCWH', format: 'RGB'});
    canvas.getContext('2d').putImageData(data, 0, 0);
  }
  /**
   * @param {ProgressCallbackPayload} info 
   */
  async modelProgressCallback(info) {
    if (info.statusText) {
      this.mergeState({modelStatus: info.statusText});
    } else {
      console.log("modelProgressCallback", info);
    }
  }
  /**
   * @param {import('../../../src/pipelines/common.js').ProgressCallbackPayload} info 
   */
  async inferenceProgressCallback(info) {
    if (info.statusText) {
      this.setInferenceStatus(info.statusText);
    } else if (info.status === 'Seed') {
      this.mergeState({generatedSeed: info.seed});
    } else {
      console.log("inferenceProgressCallback", info);
    }
    if (info.images) {
      await this.drawImage(info.images[0]);
    }
  }
  setModelState(modelState) {
    this.mergeState({modelState});
  }
  setPrompt(prompt) {
    this.mergeState({prompt});
  }
  setNegativePrompt(negativePrompt) {
    this.mergeState({negativePrompt});
  }
  async loadModel() {
    const {selectedPipeline} = this.state;
    if (!selectedPipeline) {
      return;
    }
    this.setModelState('loading');
    try {
      App.pipeline?.release();
      App.pipeline = await DiffusionPipeline.fromPretrained(
        selectedPipeline.repo,
        {
          revision: selectedPipeline?.revision,
          progressCallback: this.modelProgressCallback.bind(this),
        }
      )
      this.setModelState('ready');
    } catch (e) {
      alert(e)
      console.error(e)
    }
  }
  /**
   * @param {Uint8ClampedArray} d 
   * @returns {any}
   */
  getRgbData(d) {
    console.log("getRgbData", d);
    /** @type {any} */
    let rgbData = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for (let i = 0; i < d.length; i += 4) {
      let x = (i / 4) % 512;
      let y = Math.floor((i / 4) / 512)
      if (!rgbData[0][y]) rgbData[0][y] = [];
      if (!rgbData[1][y]) rgbData[1][y] = [];
      if (!rgbData[2][y]) rgbData[2][y] = [];
      rgbData[0][y][x] = (d[i + 0] / 255) * 2 - 1;
      rgbData[1][y][x] = (d[i + 1] / 255) * 2 - 1;
      rgbData[2][y][x] = (d[i + 2] / 255) * 2 - 1;
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }
  uploadImage(e) {
    if (!e.target.files[0]) {
      // No image uploaded
      return;
    }
    const uploadedImage = new Image(512, 512); // resize image to 512, 512
    const reader = new FileReader();
    // On file read loadend
    reader.addEventListener('loadend', function (file) {
      // On image load
      uploadedImage.addEventListener('load', function () {
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
  async runInference() {
    if (!App.pipeline) {
      return;
    }
    this.setModelState('inferencing');
    const {
      prompt,
      negativePrompt,
      inferenceSteps,
      guidanceScale,
      seed,
      width,
      height,
      runVaeOnEachStep,
      img2img,
      inputImage,
      strength,
      selectedPipeline,
    } = this.state;
    const {sdV1} = selectedPipeline;
    try {
      const images = await App.pipeline.run({
        prompt,
        negativePrompt,
        numInferenceSteps: inferenceSteps,
        guidanceScale,
        seed,
        width,
        height,
        runVaeOnEachStep,
        progressCallback: this.inferenceProgressCallback.bind(this),
        img2imgFlag: img2img,
        inputImage,
        strength,
        sdV1,
      });
      await this.drawImage(images[0]);
    } catch (e) {
      console.error('Oops', e);
      const inferenceStatus = `Oops: ${e} (see console for more details)`;
      this.mergeState({inferenceStatus});
    }
    this.setModelState('ready');
  }
  renderDownloadButton() {
    const {prompt, generatedSeed} = this.state;
    return jsx(
      'button',
      {
        onClick: () => {
          const canvas = this.canvas.current;
          if (!canvas) {
            console.warn('Missing canvas ref');
            return;
          }
          const link = document.createElement('a');
          // @todo Allow right click -> save as somehow.
          const escapedPrompt = escape(prompt);
          link.download = `prompt ${escapedPrompt} seed ${generatedSeed}.png`;
          link.href = canvas.toDataURL('image/png');
          //link.target = '_blank';
          link.click();
        }
      },
      `Download ${generatedSeed}`,
    );
  }
  render() {
    const {
      hasF16,
      selectedPipeline,
      modelState,
      prompt,
      negativePrompt,
      inferenceSteps,
      width,
      height,
      guidanceScale,
      seed,
      generatedSeed,
      modelStatus,
      inferenceStatus,
      img2img,
      inputImage,
      strength,
      runVaeOnEachStep,
    } = this.state;
    const disabled = modelState !== 'ready';
    return jsx(
      ThemeProvider,
      {
        theme: darkTheme
      },
      jsx(CssBaseline, { enableColorScheme: true }),
      jsx(
        Container,
        null,
        jsx(BrowserFeatures, null),
        jsx(Stack, { alignItems: 'center' },
        jsx(Box, { sx: { bgcolor: '#282c34' }, pt: 4, pl: 3, pr: 3, pb: 4 },
          jsx(Grid, { container: true, spacing: 2 },
            jsx(Grid, { item: true, xs: 6 },
              jsx(Stack, { spacing: 2 },
                jsx(
                  TextField,
                  {
                    label: "Prompt",
                    disabled,
                    onChange: (e) => this.setPrompt(e.target.value),
                    value: prompt,
                  }
                ),
                jsx(
                  TextField,
                  {
                    label: "Negative Prompt",
                    disabled,
                    onChange: (e) => this.setNegativePrompt(e.target.value),
                    value: negativePrompt,
                  }
                ),
                jsx(
                  TextField,
                  {
                    label: "Number of inference steps (Because of PNDM Scheduler, it will be i+1)",
                    type: 'number',
                    disabled,
                    onChange: (e) => {
                      let inferenceSteps = parseInt(e.target.value);
                      if (isNaN(inferenceSteps)) {
                        inferenceSteps = selectedPipeline.inferenceSteps;
                      }
                      this.mergeState({inferenceSteps});
                    },
                    value: inferenceSteps,
                  }
                ),
                jsx(Row, null,
                  'Width',
                  jsx(
                    'input', {
                      type: 'number',
                      disabled,
                      onChange: (e) => {
                        let width = parseInt(e.target.value);
                        if (isNaN(width)) {
                          width = selectedPipeline.width;
                        }
                        this.mergeState({width});
                      },
                      value: width,
                    }
                  ),
                  'Height',
                  jsx(
                    'input',
                    {
                      type: 'number',
                      disabled,
                      onChange: (e) => {
                        let height = parseInt(e.target.value);
                        if (isNaN(height)) {
                          height = selectedPipeline.height;
                        }
                        this.mergeState({height});
                      },
                      value: height,
                    }
                  ),
                  jsx(
                    'button',
                    {
                      disabled,
                      onClick: (e) => {
                        const {width, height} = this.state;
                        this.mergeState({height: width, width: height});
                      },
                    },
                    'Flip'
                  ),
                  jsx(
                    'button',
                    {
                      disabled,
                      onClick: (e) => {
                        this.mergeState({width: 512, height: 512});
                      },
                    },
                    '512x512'
                  ),
                  jsx(
                    'button',
                    {
                      disabled,
                      onClick: (e) => {
                        this.mergeState({width: 768, height: 512});
                      },
                    },
                    '768x512'
                  ),
                  jsx(
                    'button',
                    {
                      disabled,
                      onClick: (e) => {
                        this.mergeState({width: 512, height: 768});
                      },
                    },
                    '512x768'
                  ),
                ),
                jsx(
                  TextField,
                  {
                    label: "Guidance Scale. Controls how similar the generated image will be to the prompt.",
                    type: 'number',
                    InputProps: {
                      inputProps: {min: 1, max: 20, step: 0.5},
                    },
                    disabled,
                    onChange: (e) => {
                      let guidanceScale = parseFloat(e.target.value);
                      if (isNaN(guidanceScale)) {
                        guidanceScale = selectedPipeline.guidanceScale;
                      }
                      this.mergeState({guidanceScale});
                    },
                    value: guidanceScale,
                  }
                ),
                jsx(
                  TextField,
                  {
                    label: "Seed (Creates initial random noise)",
                    disabled,
                    onChange: (e) => this.mergeState({seed: e.target.value}),
                    value: seed,
                  }
                ),
                (selectedPipeline === null || selectedPipeline === void 0 ? void 0 : selectedPipeline.hasImg2Img) &&
                // @todo own method
                jsx(
                  Fragment,
                  null,
                  jsx(
                    FormControlLabel,
                    {
                      label: "Check if you want to use the Img2Img pipeline",
                      control: jsx(
                        Checkbox,
                        {
                          disabled,
                          onChange: (e) => setImg2Img(e.target.checked), checked: img2img
                        }
                      ),
                    }
                  ),
                  jsx(
                    "label",
                    {
                      htmlFor: "upload_image"
                    },
                    "Upload Image for Img2Img Pipeline:"
                  ),
                  jsx(
                    TextField,
                    {
                      id: "upload_image",
                      inputProps: { accept: "image/*" },
                      type: "file",
                      disabled: !img2img,
                      onChange: (e) => uploadImage(e)
                    }
                  ),
                  jsx(
                    TextField,
                    {
                      label: "Strength (Noise to add to input image). Value ranges from 0 to 1",
                      type: 'number',
                      InputProps: {
                        inputProps: { min: 0, max: 1, step: 0.1 }
                      },
                      disabled: !img2img,
                      onChange: (e) => setStrength(parseFloat(e.target.value)),
                      value: strength
                    }
                  )
                ),
                jsx(
                  FormControlLabel,
                  {
                    label: "Run VAE after each step",
                    control: jsx(
                      Checkbox,
                      {
                        disabled,
                        onChange: (e) => setRunVaeOnEachStep(e.target.checked),
                        checked: runVaeOnEachStep
                      }
                    )
                  }
                ),
                jsx(Row, null,
                  'Model:',
                  jsx(
                    Select,
                    {
                      value: selectedPipeline.name,
                      onChange: e => {
                        this.setSelectedPipeline(pipelines.find(p => e.target.value === p.name));
                        this.setModelState('none');
                      }
                    },
                    pipelines.map((p, key) => jsx(
                      MenuItem,
                      {
                        value: p.name,
                        /*disabled: !hasF16 && p.fp16,*/
                        key
                      },
                      p.name,
                    ))
                  ),
                  jsx(
                    'button',
                    {
                      onClick: () => this.loadModel(),
                      disabled: modelState != 'none'
                    },
                    "Load model"
                  ),
                  modelStatus,
                ),
              ),
            jsx(
              Stack,
              null,
              inferenceStatus,
              jsx(
                Row,
                null,
                jsx(
                  'button',
                  {
                    onClick: () => this.mergeState({now: Date.now()}),
                  },
                  "Refresh UI"
                ),
                jsx(
                  'button',
                  {
                    onClick: () => this.runInference(),
                    disabled
                  },
                  "Run"
                ),
                this.renderDownloadButton(),
              ),
              jsx(
                "canvas",
                {
                  ref: this.canvas,
                  // @todo flipped
                  width: height,
                  height: width,
                  style: {
                    border: '1px dashed #ccc'
                  }
                }
              ),
            )
        )
        ),
        jsx(Divider, null),
        jsx(
          "a",
          {
            href: "https://github.com/kungfooman/StableDiffusion.js/",
            target: "_blank"
          },
          "https://github.com/kungfooman/StableDiffusion.js/")
        ),
      )
    )
    )
  }
}
export {darkTheme, pipelines, App};

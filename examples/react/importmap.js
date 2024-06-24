window.process = {
    env: {
        NODE_ENV: 'development'
    }
};
function importFile(content) {
  return "data:text/javascript;base64," + btoa(content);
}
const playcanvasEngineThis = document.currentScript.src + '/../../../';
const playcanvasEngine = playcanvasEngineThis;
// const playcanvasEngine = '/playcanvas-engine-jsdoc/';
const nodeModules = './node_modules/';
// const nodeModules = playcanvasEngineThis + 'examples/node_modules/';
const react = {
  'prop-types'               : nodeModules + 'react-es6/prop-types/index.js',
  'tiny-warning'             : nodeModules + 'react-es6/tiny-warning.mjs',
  'tiny-invariant'           : nodeModules + 'react-es6/tiny-invariant.mjs',
  'mini-create-react-context': nodeModules + 'react-router/node_modules/mini-create-react-context/dist/esm/index.js',
  'path-to-regexp'           : nodeModules + 'react-es6/path-to-regexp.mjs',
  'react-is'                 : nodeModules + 'react-es6/react-is.mjs',
  'hoist-non-react-statics'  : nodeModules + 'react-es6/hoist-non-react-statics.mjs',
  'resolve-pathname'         : nodeModules + 'react-es6/resolve-pathname.mjs',
  'isarray'                  : nodeModules + 'react-es6/isarray.mjs'
};
const react_min = {
  ...react,
  'react': nodeModules + 'react-es6/build/react.min.mjs',
  'react-dom': nodeModules + 'react-es6/build/react-dom.min.mjs',
  'react-dom/client': nodeModules + 'react-es6/build/react-dom-client.min.mjs'
};
const { host } = document.location;
const local = host === 'localhost' || host === '127.0.0.1';
const playcanvas = local ? {
  'playcanvas': playcanvasEngine + 'src/index.js',
  'playcanvas/': playcanvasEngine
  // 'playcanvas'            : playcanvasEngine + 'build/playcanvas.dbg.mjs/index.js',
} : {
  'playcanvas': playcanvasEngine + 'build/playcanvas.whole.mjs',
  'playcanvas/': playcanvasEngine
};
const imports = {
  ...playcanvas,
  '@playcanvas/pcui'                                       : nodeModules + '@playcanvas/pcui/dist/module/src/index.mjs',
  '@playcanvas/pcui/react'                                 : nodeModules + '@playcanvas/pcui/react/dist/module/src/index.mjs',
  '@playcanvas/observer'                                   : nodeModules + '@playcanvas/observer/dist/index.mjs',
  '@monaco-editor/react'                                   : nodeModules + '@monaco-editor/react/dist/index.mjs',
  '@monaco-editor/loader'                                  : nodeModules + '@monaco-editor/loader/lib/es/index.js',
  'state-local'                                            : nodeModules + 'state-local/lib/es/state-local.js',
  'react-router-dom'                                       : nodeModules + 'react-router-dom/esm/react-router-dom.js',
  'react-router'                                           : nodeModules + 'react-router/esm/react-router.js',
  '@babel/runtime/helpers/esm/inheritsLoose'               : nodeModules + '@babel/runtime/helpers/esm/inheritsLoose.js',
  '@babel/runtime/helpers/esm/extends'                     : nodeModules + '@babel/runtime/helpers/esm/extends.js',
  '@babel/runtime/helpers/esm/objectWithoutPropertiesLoose': nodeModules + '@babel/runtime/helpers/esm/objectWithoutPropertiesLoose.js',
  'history'                                                : nodeModules + 'history/esm/history.js',
  'value-equal'                                            : nodeModules + 'value-equal/esm/value-equal.js',
  '@playcanvas/pcui/styles'                                : nodeModules + '@playcanvas/pcui/styles/dist/index.mjs',
  'playcanvas-extras'                                      : playcanvasEngine + 'extras/index.js',
  'fflate'                                                 : nodeModules + 'fflate/esm/browser.js',
  '@tweenjs/tween.js'                                      : nodeModules + '@tweenjs/tween.js/dist/tween.esm.js',
  '@mui/material' : './mui.mjs',
  'wasm-feature-detect'                                    : nodeModules + 'wasm-feature-detect/dist/esm/index.js',
  'web-vitals'                                             : nodeModules + 'web-vitals/dist/modules/index.js',
  // rip:
  '@mui/material/CssBaseline'                              : nodeModules + '@mui/material/CssBaseline/CssBaseline.js',
  '@mui/material/Box'                                      : nodeModules + '@mui/material/Box/Box.js',
  '@mui/material/Container'                                : nodeModules + '@mui/material/Container/Container.js',
  '@mui/material/styles/ThemeProvider'                     : nodeModules + '@mui/material/styles/ThemeProvider.js',
  '@mui/material/styles/createTheme'                       : nodeModules + '@mui/material/styles/createTheme.js',
  '@mui/material/Stack'                                    : nodeModules + '@mui/material/Stack/Stack.js',
  '@mui/material/Grid'                                     : nodeModules + '@mui/material/Grid/Grid.js',
  '@mui/material/TextField'                                : nodeModules + '@mui/material/TextField/TextField.js',
  '@mui/material/Button'                                   : nodeModules + '@mui/material/Button/Button.js',
  '@mui/material/Divider'                                  : nodeModules + '@mui/material/Divider/Divider.js',
  '@mui/material/Checkbox'                                 : nodeModules + '@mui/material/Checkbox/Checkbox.js',
  '@mui/material/FormControl'                              : nodeModules + '@mui/material/FormControl/FormControl.js',
  '@mui/material/InputLabel'                               : nodeModules + '@mui/material/InputLabel/InputLabel.js',
  '@mui/material/MenuItem'                                 : nodeModules + '@mui/material/MenuItem/MenuItem.js',
  '@mui/material/Select'                                   : nodeModules + '@mui/material/Select/Select.js',
  '@mui/material/FormControlLabel'                         : nodeModules + '@mui/material/FormControlLabel/FormControlLabel.js',
  '@mui/material'                                          : './mui.mjs',
  '@xenova/transformers'                                   : nodeModules + '@xenova/transformers/dist/transformers.js',
  '@aislamov/diffusers.js'                                 : '../../dist/index.esm.js',
  '@huggingface/hub'                                       : '../../node_modules/@huggingface/hub/dist/browser/index.mjs',
  'idb'                                                    : '../../node_modules/idb/build/index.js',
  'seedrandom'                                             : '../../node_modules/seedrandom/seedrandom.mjs',
  '@aislamov/onnxruntime-web64'                            : importFile("console.log('todo @aislamov/onnxruntime-web64');\nexport default {};"),
  'react/jsx-runtime'                            : importFile("console.log('todo react/jsx-runtime');\nexport default {};"),
  ...react_min
};
const importmap = document.createElement('script');
importmap.type = 'importmap';
importmap.textContent = JSON.stringify({ imports });
let node = document.body;
if (!node) {
    node = document.head;
}
if (!node) {
    console.error('importmap.js> make sure to either have a <HEAD> or <BODY> before you include this file.');
} else {
    node.appendChild(importmap);
}

//import React from 'react';
//import {createRoot} from 'react-dom/client';
import {react as React, reactDom} from '@mui/material';
const {createRoot} = reactDom;
//import './index.css';
import App from './App.js';
import reportWebVitals from './reportWebVitals.js';
import {jsx} from './jsx.js';
// @ts-ignore
window.assert = () => {}
const root = createRoot(
  document.getElementById('root')
);
root.render(
  jsx(React.StrictMode, null, [
    jsx(App, null)
  ])
);
// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

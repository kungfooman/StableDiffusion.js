import {StrictMode     } from 'react';
import {createRoot     } from 'react-dom/client';
import {App            } from './App.js';
import {reportWebVitals} from './reportWebVitals.js';
import {jsx            } from './jsx.js';
import * as React          from 'react';
import * as ReactDomClient from 'react-dom/client';
import * as ModuleApp from './App.js';
Object.assign(window, {createRoot, App, reportWebVitals, jsx, React, ReactDomClient, ...ModuleApp});
const root = createRoot(document.getElementById('root'));
root.render(
  jsx(StrictMode, null, jsx(App, null))
);
// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

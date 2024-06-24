import {react as React} from '@mui/material';
const { createElement, Fragment } = React;
export const jsx = createElement;
export const fragment = (/** @type {any} */ ...args) => jsx(Fragment, null, ...args);

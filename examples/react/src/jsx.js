import React from 'react';
const { createElement, Fragment } = React;
export const jsx = createElement;
export const fragment = (/** @type {any} */ ...args) => jsx(Fragment, null, ...args);

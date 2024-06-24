import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import React from 'react';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import {fragment, jsx} from '../jsx.js';
/**
 * @param {{ question: string, answer:string }} props 
 * @returns 
 */
function FaqItem(props) {
  return fragment(
    jsx(
      ListItem,
      null,
      jsx(ListItemText, {primary: 'Q: ' + props.question})
    ),
    jsx(
      ListItem,
      null,
      jsx(ListItemText, {primary: 'A: ' + props.answer})
    ),
    jsx(Divider, null)
  );
}
export function FAQ() {
  return jsx(
    Box, {
      sx: {
        width: '100%',
        bgcolor: 'background.paper'
      }
    },
    jsx(
      List,
      null,
      jsx(
        ListItem,
        null,
        jsx("h2", null, "FAQ")
      ),
      jsx(FaqItem, {
        question: 'What if I get protobuf parsing failed error?',
        answer: 'Open DevTools, go to Application -> Storage and press "Clear site data".'
      }),
      jsx(FaqItem, {
        question: 'What if I get sbox_fatal_memory_exceeded?',
        answer: "You don't have enough RAM to run SD. You can try reloading the tab or browser."
      }),
      jsx(FaqItem, {
        question: 'How did you make it possible?',
        answer: 'In order to run it, I had to port StableDiffusionPipeline from python to JS. Then patch onnxruntime and emscripten+binaryen (WebAssembly compiler toolchain) to support allocating and using >4GB memory. Then WebAssembly spec and V8 engine.'
      })
    )
  );
}

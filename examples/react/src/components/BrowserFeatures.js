import React, { useEffect, useState } from 'react'
import { memory64, jspi } from 'wasm-feature-detect'
import Stack from '@mui/material/Stack'
import Alert from '@mui/material/Alert'
import { jsx } from '../jsx.js';
export async function hasFp16 () {
  try {
    // @ts-ignore
    const adapter = await navigator.gpu.requestAdapter()
    return adapter.features.has('shader-f16')
  } catch (e) {
    return false
  }
}
export const BrowserFeatures = () => {
  const [hasMemory64, setHasMemory64] = useState(false);
  const [hasSharedMemory64, setHasSharedMemory64] = useState(false);
  const [hasJspi, setHasJspi] = useState(false);
  const [hasF16, setHasF16] = useState(false);
  const [hasGpu, setHasGpu] = useState(false);
  useEffect(() => {
    memory64().then(value => setHasMemory64(value))
    // @ts-ignore
    jspi().then(value => setHasJspi(value))
    try {
      // @ts-ignore
      const mem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true, index: 'i64' })
      // @ts-ignore
      setHasSharedMemory64(mem.type().index === 'i64')
    } catch (e) {
      //
    }
    hasFp16().then(v => {
      setHasF16(v)
      setHasGpu(true)
    })
  }, [])
  return (jsx(Stack, null,
    !hasMemory64 && jsx(Alert, { severity: "error" }, "You need latest Chrome with \"Experimental WebAssembly\" flag enabled!"),
    !hasJspi && jsx(Alert, { severity: "error" }, "You need \"Experimental WebAssembly JavaScript Promise Integration (JSPI)\" flag enabled!"),
    !hasSharedMemory64 && jsx(Alert, { severity: "error" }, "You need Chrome Canary 119 or newer!"),
    !hasF16 && jsx(Alert, { severity: "error" }, "You need Chrome Canary 121 or higher for FP16 support!"),
    !hasGpu && jsx(Alert, { severity: "error" }, "You need a browser with WebGPU support!")));
}

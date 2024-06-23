const fs = require("fs");
const path = require("path");
// copy onnxruntime-web WebAssembly files to public/ folder
let srcFolder = path.join(__dirname, "node_modules", "@aislamov/onnxruntime-web64", "dist");
if (!fs.existsSync(srcFolder)) {
  srcFolder = path.resolve('../../node_modules/@aislamov/onnxruntime-web64/dist');
}
console.log('using', srcFolder);
const destFolder = path.join(__dirname, "public", "static", "js");
if (fs.existsSync('./node_modules/.cache')) {
  fs.rmdirSync(path.resolve('./node_modules/.cache'), { recursive: true })
}
if (fs.existsSync(destFolder)) {
  fs.rmSync(destFolder, { recursive: true, force: true });
}
fs.mkdirSync(destFolder, { recursive: true });
fs.copyFileSync(path.join(srcFolder, "ort-wasm.wasm"), path.join(destFolder, "ort-wasm.wasm"));
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-simd.wasm"),
  path.join(destFolder, "ort-wasm-simd.wasm")
);
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-simd.jsep.wasm"),
  path.join(destFolder, "ort-wasm-simd.jsep.wasm")
);
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-threaded.wasm"),
  path.join(destFolder, "ort-wasm-threaded.wasm")
);
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-simd-threaded.wasm"),
  path.join(destFolder, "ort-wasm-simd-threaded.wasm")
);
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-simd-threaded.jsep.wasm"),
  path.join(destFolder, "ort-wasm-simd-threaded.jsep.wasm")
);
fs.copyFileSync(
  path.join(srcFolder, "ort-wasm-simd-threaded.worker.js"),
  path.join(destFolder, "ort-wasm-simd-threaded.worker.js")
);

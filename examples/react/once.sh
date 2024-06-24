# React and MaterialUI still don't deliver browser-compatible ESM
# files and if you point it out you are accused of hate-speech.
# npx esbuild --bundle node_modules/@mui/material/index.js --outfile=mui.mjs --format=esm
npx esbuild --bundle ui.js --outfile=mui.mjs --format=esm

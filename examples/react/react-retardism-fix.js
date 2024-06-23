const {readFileSync, writeFileSync} = require('fs');
let content = readFileSync('build/index.html', 'utf-8');
content = content.replaceAll('>', '>\n');
content = content.replaceAll('/diffusers.js/', '');
writeFileSync('build/index.html', content);

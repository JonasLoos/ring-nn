const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.csv': 'text/csv',
    '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
    // Parse URL
    let url = req.url;
    
    // Default to index.html
    if (url === '/') {
        url = '/index.html';
    }
    
    // Determine file path
    let filePath;
    if (url.startsWith('/training_logs_')) {
        // For log files, look in the parent directory
        filePath = path.join(__dirname, '..', url);
    } else {
        // For webapp files, look in the current directory
        filePath = path.join(__dirname, url);
    }
    
    // Get file extension
    const extname = path.extname(filePath);
    const contentType = MIME_TYPES[extname] || 'application/octet-stream';
    
    // Read file
    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                // File not found
                res.writeHead(404);
                res.end('File not found');
            } else {
                // Server error
                res.writeHead(500);
                res.end(`Server Error: ${err.code}`);
            }
        } else {
            // Success
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
}); 
#!/usr/bin/env python3
"""Simple HTTP server for testing therapeutic visualization"""
import http.server
import socketserver
import os

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Serving therapeutic visualization at http://localhost:{PORT}/therapeutics_viz.html")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    httpd.serve_forever()
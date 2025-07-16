#!/usr/bin/env python3
"""
Simple HTTP server for the TE-AI Architecture Dashboard
Serves the dashboard HTML and the state JSON file
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve te_ai_state.json from current directory
        if self.path.startswith('/te_ai_state.json'):
            self.path = '/te_ai_state.json'
        
        # Serve dashboard HTML
        elif self.path == '/' or self.path == '/dashboard':
            self.path = '/architecture-dashboard.html'
        
        return super().do_GET()
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    # Change to scripts directory
    os.chdir(Path(__file__).parent)
    
    print(f"\n🌐 Starting TE-AI Architecture Dashboard Server")
    print(f"📊 Dashboard URL: http://localhost:{PORT}/")
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"\n💡 Make sure to run your TE-AI simulation in another terminal!")
    print(f"   The dashboard will display real-time data from te_ai_state.json\n")
    print(f"Press Ctrl+C to stop the server\n")
    
    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Make sure port {PORT} is not already in use")
        sys.exit(1)

if __name__ == "__main__":
    main()
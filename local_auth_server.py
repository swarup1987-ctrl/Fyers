import threading
import http.server
import socketserver
import urllib.parse

class AuthCodeRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        auth_code = params.get("auth_code", [None])[0]

        if auth_code:
            if hasattr(self.server, "auth_code_callback") and callable(self.server.auth_code_callback):
                self.server.auth_code_callback(auth_code)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authentication successful! You may close this window and return to the app.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing 'auth_code' in URL.")

    def log_message(self, format, *args):
        return

class AuthCodeHTTPServer(socketserver.TCPServer):
    allow_reuse_address = True
    def __init__(self, server_address, RequestHandlerClass, auth_code_callback):
        super().__init__(server_address, RequestHandlerClass)
        self.auth_code_callback = auth_code_callback

def start_auth_server(port, auth_code_callback):
    server = AuthCodeHTTPServer(("", port), AuthCodeRequestHandler, auth_code_callback)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread

def stop_auth_server(server, thread):
    if server:
        server.shutdown()
        server.server_close()
    if thread:
        thread.join(timeout=2)
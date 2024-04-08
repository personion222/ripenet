from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import sys

ip = sys.argv[1]
port = "8080"

print(f"openning server at\nhttps://{ip}:{port}/index.html")

httpd = HTTPServer(server_address=(ip, 8080), RequestHandlerClass=SimpleHTTPRequestHandler)

ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_ctx.load_cert_chain("certificate.pem", "key.pem")
ssl_ctx.set_ciphers("@SECLEVEL=1:ALL")

httpd.socket = ssl_ctx.wrap_socket(
    sock=httpd.socket,
    server_side=True
)

httpd.serve_forever()

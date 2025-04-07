#!/bin/bash

SSL_DIR="./nginx_base/ssl"

mkdir -p "$SSL_DIR"

# Generate private key
openssl genrsa -out "$SSL_DIR/nginx.key" 2048

# Generate CSR
openssl req -new -key "$SSL_DIR/nginx.key" -out "$SSL_DIR/nginx.csr" -subj "//CN=localhost/O=OpenAI Proxy/C=US"

# Generate certificate
openssl x509 -req -days 365 -in "$SSL_DIR/nginx.csr" -signkey "$SSL_DIR/nginx.key" -out "$SSL_DIR/nginx.crt"
# Permissions
chmod 600 "$SSL_DIR/nginx.key"
chmod 644 "$SSL_DIR/nginx.crt"

echo "SSL certificates generated successfully!"
echo "Private key: $SSL_DIR/nginx.key"
echo "Certificate: $SSL_DIR/nginx.crt"


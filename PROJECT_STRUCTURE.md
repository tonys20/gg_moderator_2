#  Project Folder Structure

```
├── Dockerfile.client
├── Dockerfile.guardian
├── PROJECT_STRUCTURE.md
├── README.md
├── TEMPLATE.env
├── app
│   ├── .env
│   ├── misc.py
│   ├── openai_client.py
│   └── requirements.txt
├── docker-compose.yml
├── generate_ssl_cert.sh
├── granite_model_files
│   ├── README.md
│   ├── added_tokens.json
│   ├── config.json
│   ├── generation_config.json
│   ├── gitattributes
│   ├── merges.txt
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── roc.png
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── ibm_granite
│   ├── __init__.py
│   └── run_granite.py
├── mitmproxy_module
│   └── openai_proxy.py
├── nginx_base
│   ├── nginx-1.26.3
│   └── ssl
│       ├── nginx.crt
│       ├── nginx.csr
│       └── nginx.key
├── project_requirement.pdf
├── requirements.txt
└── screenshots
    └── sample_usage.png
```
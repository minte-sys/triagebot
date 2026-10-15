# Configuration

## The config file
Settings live in config.toml in the folder you run repobrief from. Copy
config.example.toml to start. Use --config to point at a file somewhere else.

## Empty or missing config file
If config.toml is empty or a section is missing, repobrief stops with KeyError naming
the missing section, for example KeyError: 'llm'. Copy config.example.toml again.

## API keys
Keys are read from a .env file in the same folder as config.toml, never from config.toml
itself. Set GITHUB_TOKEN for GitHub and LLM_API_KEY for the model provider.

## Model settings
The [llm] section sets backend, model and timeout_s. timeout_s is how long to wait for
the model, 60 seconds by default.

## Web viewer port
The [web] section sets port, 8000 by default. Change it if another program already uses
that port.

## Proxies
repobrief uses the HTTPS_PROXY environment variable if it is set. Behind a company proxy
that inspects TLS, also set SSL_CERT_FILE to your company's certificate bundle.

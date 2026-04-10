# llmctl

A CLI tool for managing multiple local [llama.cpp](https://github.com/ggerganov/llama.cpp) model servers with an OpenAI-compatible reverse proxy. Load several GGUF models simultaneously, then point any OpenAI client at a single endpoint that routes requests by model name.

## Prerequisites

- **llama.cpp server** — one of `llama-server`, `llama-cpp-server`, or `server` must be in your `PATH` (or set via config)
- **huggingface-cli** (optional) — needed only for `llmctl pull`. Install with `pip install huggingface_hub`

## Installation

### From source

```sh
git clone https://github.com/makkalot/llmctl.git
cd llmctl
make install    # builds all platforms, installs current-platform binary to /usr/local/bin/llmctl
```

### Build only

```sh
make build      # outputs to ./bin/ for darwin/linux × amd64/arm64
```

## Quick start

```sh
# 1. Tell llmctl where your models and server binary are
llmctl set models_dir /path/to/your/models
llmctl set server_bin /path/to/llama-server

# 2. Load a model (starts a llama-server backend)
llmctl load mistral

# 3. Start the OpenAI-compatible proxy
llmctl proxy

# 4. Use it from any OpenAI client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Commands

### Model management

| Command | Aliases | Description |
|---------|---------|-------------|
| `llmctl list` | `ls` | List `.gguf` models found in your models directory |
| `llmctl pull <user/repo>` | `download` | Download a GGUF model from Hugging Face |
| `llmctl rm <model>` | `remove`, `delete` | Delete a model file from disk |
| `llmctl alias <name> <model>` | | Create a short alias for a model file |

### Instance management

| Command | Aliases | Description |
|---------|---------|-------------|
| `llmctl load <model>` | `run` | Load a model (starts a llama-server backend) |
| `llmctl unload <name>` | | Stop a running model instance |
| `llmctl stop` | `kill` | Stop all instances and the proxy |
| `llmctl default <name>` | | Set the default model for unmatched requests |
| `llmctl ps` | | List all loaded instances with status |
| `llmctl logs <name>` | `log` | Show last 50 lines of an instance's log |

### Proxy

| Command | Aliases | Description |
|---------|---------|-------------|
| `llmctl proxy` | `serve` | Start the OpenAI-compatible reverse proxy |
| `llmctl status` | | Show overview of running state |

### Configuration

| Command | Aliases | Description |
|---------|---------|-------------|
| `llmctl config` | `cfg` | Show current configuration |
| `llmctl set <key> <value>` | | Update a config value |

## Loading models

### From local files

`llmctl load` finds models by fuzzy matching against `.gguf` files in your configured models directory:

```sh
llmctl load mistral              # matches e.g. Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
llmctl load llama3 --name chat   # load with a custom instance name
```

### From Hugging Face (direct)

Use the `-hf` flag to load directly from a Hugging Face repo without downloading first:

```sh
llmctl load "" -hf unsloth/Qwen3.5-27B-GGUF
```

### From Hugging Face (download first)

```sh
llmctl pull user/repo            # downloads to HF cache inside models_dir
llmctl load user/repo            # loads from the cached download
```

The `pull` command uses `huggingface-cli` (or `python3 -m huggingface_hub`) and stores models in the HuggingFace cache layout under your models directory.

## Multi-model workflow

```sh
# Load multiple models — each gets its own backend port (starting at 9100)
llmctl load mistral
llmctl load codellama --name code
llmctl load llama3 --name chat

# The first loaded model becomes the default
# Change the default if needed
llmctl default chat

# Start the proxy (single endpoint on port 8080)
llmctl proxy

# Route by model name in the request body
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "code", "messages": [...]}'

# Or omit the model to hit the default
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages": [...]}'

# List available models (OpenAI-compatible)
curl http://localhost:8080/v1/models
```

The proxy matches the `model` field in requests using case-insensitive substring matching. If no model matches, the request goes to the default instance.

## Configuration

Config is stored at `~/.llmctl.json`. Set values with `llmctl set`:

```sh
llmctl set models_dir /path/to/models
llmctl set server_bin /path/to/llama-server
llmctl set host 0.0.0.0
llmctl set port 8080
llmctl set gpu_layers -1          # -1 = offload all layers to GPU
llmctl set ctx_size 4096
```

| Key | Default | Description |
|-----|---------|-------------|
| `models_dir` | `~/models` | Directory containing `.gguf` files |
| `server_bin` | auto-detected | Path to `llama-server` binary |
| `host` | `0.0.0.0` | Proxy listen address |
| `port` | `8080` | Proxy listen port |
| `gpu_layers` | `-1` (all) | Number of layers to offload to GPU |
| `ctx_size` | `4096` | Context window size |

### Extra server arguments

Add extra `llama-server` arguments in the config file directly (`~/.llmctl.json`):

```json
{
  "extra_args": ["--flash-attn", "on", "--temp", "0.6"]
}
```

### Per-model overrides

Override settings for specific models by adding a `models` section to `~/.llmctl.json`:

```json
{
  "models": {
    "codellama": {
      "gpu_layers": 20,
      "ctx_size": 8192,
      "extra_args": ["--temp", "0.2"]
    }
  }
}
```

### Aliases

Create short names for model files:

```sh
llmctl alias qwen Qwen3.5-27B-GGUF/UD-Q4_K_XL.gguf
llmctl load qwen
```

## API endpoints

When the proxy is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (routes by `model` field) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |

Requests can also specify a model via the `X-Model` HTTP header.

## Runtime files

| Path | Description |
|------|-------------|
| `~/.llmctl.json` | Configuration |
| `~/.llmctl.registry.json` | Registry of running instances and proxy PID |
| `~/.llmctl-logs/<name>.log` | Per-instance server logs |

## License

See [LICENSE](LICENSE) for details.

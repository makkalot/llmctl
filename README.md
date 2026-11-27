# llmctl

A CLI tool for managing multiple local [llama.cpp](https://github.com/ggerganov/llama.cpp) model servers with an OpenAI-compatible reverse proxy. Load several GGUF models simultaneously, then point any OpenAI client at a single endpoint that routes requests by model name.

## Prerequisites

- **Docker** — every model backend runs `llama-server` in its own container. The user running llmctl must be able to run `docker` (e.g. be in the `docker` group).
- **GPU support for containers**:
  - NVIDIA desktop/server: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (containers get `--gpus all`)
  - Jetson: the `nvidia` container runtime that ships with JetPack (containers get `--runtime nvidia`), plus a Jetson-compatible llama.cpp image (the official images are not built for Jetson)
- **huggingface-cli** (optional) — needed only for `llmctl pull`. Install with `pip install huggingface_hub`

## Installation

### Systemd service

Install the proxy as a systemd system service (auto-starts on boot, restarts on crash). The service runs as the invoking user (`$SUDO_USER`), so model state stays in that user's home:

```sh
sudo make systemd-install     # builds, installs binary, sets up service
sudo systemctl status llmctl  # check status
journalctl -u llmctl -f       # follow logs
```

The unit file is installed to `/etc/systemd/system/llmctl.service`.
Run `sudo make systemd-uninstall` to remove it.

#### From source

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
# 1. Tell llmctl where your models are (and, optionally, which image to run)
llmctl set models_dir /path/to/your/models
llmctl set docker.image ghcr.io/ggml-org/llama.cpp:server-cuda

# 2. Load a model (starts a llama-server container)
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
| `llmctl load <model>` | `run` | Load a model (starts a llama-server container) |
| `llmctl unload <name>` | | Stop a running model instance |
| `llmctl stop` | `kill` | Stop all instances and the proxy |
| `llmctl default <name>` | | Set the default model for unmatched requests |
| `llmctl ps` | | List all loaded instances with status and context size |
| `llmctl info <name>` | | Show detailed info for an instance (params, aliases, etc.) |
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

`llmctl load` finds models by fuzzy matching against `.gguf` files in your configured models directory. Both `/` and `:` are supported as separators for HuggingFace-style names:

```sh
llmctl load mistral              # matches e.g. Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
llmctl load llama3 --name chat   # load with a custom instance name
llmctl load Qwen3.5-27B-GGUF:UD-Q4_K_XL   # HF-style name with : separator
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
llmctl set docker.image ghcr.io/ggml-org/llama.cpp:server-cuda
llmctl set docker.gpu nvidia      # nvidia | jetson | none
llmctl set docker.network none    # none | internal | bridge
llmctl set host 0.0.0.0
llmctl set port 8080
llmctl set gpu_layers -1          # -1 = offload all layers to GPU
llmctl set ctx_size 4096
```

| Key | Default | Description |
|-----|---------|-------------|
| `models_dir` | `~/models` | Directory containing `.gguf` files |
| `docker.image` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp server image |
| `docker.gpu` | `nvidia` | `nvidia` (`--gpus all`), `jetson` (`--runtime nvidia`) or `none` |
| `docker.entrypoint` | image default | Override the image entrypoint, e.g. `llama-server` for images without one |
| `docker.network` | `none` | Backend network: `none` (no network, unix socket), `internal` (no egress) or `bridge` (full egress) — see below |
| `docker.read_only` | `true` | Read-only container root filesystem (`/tmp` is a tmpfs) |
| `docker.pids_limit` | `512` | `--pids-limit` for each backend; `0` disables |
| `docker.hf_cache_dir` | `$LLAMA_CACHE` or `~/.cache/llama.cpp` | Host directory for in-container `-hf` downloads (`bridge` network only) |
| `docker.bin` | `docker` | Container CLI (any docker-compatible CLI, e.g. `podman`) |
| `host` | `0.0.0.0` | Proxy listen address |
| `port` | `8080` | Proxy listen port |
| `gpu_layers` | `-1` (all) | Number of layers to offload to GPU |
| `ctx_size` | `4096` | Context window size |

### Docker backends

Each loaded model runs as a hardened container named `llmctl-<instance>` with the label `llmctl.instance=<instance>`. The goal is to contain a compromised inference engine: it should not be able to reach the network, write to your models, or gain privileges.

```sh
docker run -d --name llmctl-<instance> --label llmctl.instance=<instance> \
  --user <your uid>:<your gid> --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp --pids-limit 512 \
  --network none [--gpus all | --runtime nvidia] \
  -v <models_dir>:<models_dir>:ro -v ~/.llmctl-run/llmctl-<instance>:~/.llmctl-run/llmctl-<instance> \
  [docker.args...] <image> --model <path> --host ~/.llmctl-run/llmctl-<instance>/llama.sock ...
```

#### Network isolation (`docker.network`)

| Mode | How the proxy reaches the backend | What the container can reach |
|------|------------------------------------|------------------------------|
| `none` (default) | llama-server listens on a unix socket in a per-instance directory (`~/.llmctl-run/<container>/llama.sock`) | Nothing — no network interfaces besides loopback |
| `internal` | Container IP on the `llmctl-internal` Docker network (`--internal`, inter-container traffic disabled) | No internet or DNS, no other backends — **but** host services listening on all interfaces (e.g. the llmctl proxy, sshd) via the network gateway |
| `bridge` | `127.0.0.1:<port>` published port | Everything (internet included). Needed only for in-container `-hf` downloads |

`none` requires a llama.cpp build with unix-socket support (`--host <path>.sock`, upstream PR #12613). If your image is older (some Jetson images), use `internal` for that model:

```json
{ "models": { "jetson-model": { "docker_network": "internal" } } }
```

**`-hf` loads with isolated networks.** In `none`/`internal` mode the container can't download, so `llmctl load "" -hf user/repo[:quant]` downloads the files on the host first (same `huggingface_hub` tooling and `models_dir` cache as `llmctl pull`), picking the file like llama.cpp does: the `.gguf` matching the quant tag (default `Q4_K_M`), all parts of split models, and an `mmproj` if the repo has one. The container then loads the local file. In `bridge` mode `-hf` is passed through to llama-server, with `docker.hf_cache_dir` mounted read-write.

#### Mounts

Host directories are mounted at the same path in the container, so model paths in the registry, `llmctl info` and `extra_args` work unchanged:

- `models_dir` — read-only.
- Directories of a model or mmproj outside `models_dir`, including symlink targets (HF cache blobs) — read-only.
- Absolute paths in `extra_args` (e.g. `--chat-template-file`, `--lora`) — read-only, except the arguments of `--slot-save-path` and `--log-file`, which llama-server writes to.
- The per-instance socket directory (`none`) or HF cache (`bridge`) — read-write.

#### Lifecycle

- If the image is missing, llmctl pulls it before starting the container, so the first pull doesn't count against the health-check timeout.
- Containers are not started with `--rm` and have no restart policy. On `unload`, `stop`, or when llmctl finds a backend that has exited, it saves the container logs to `~/.llmctl-logs/<name>.log` and removes the container (and its socket directory). `llmctl stop` also removes any `llmctl.instance`-labelled containers the registry has lost track of.

Extra `docker run` flags go in `docker.args` (global) or `docker_args` (per model, appended to the global ones). Image and entrypoint can also be set per model:

```json
{
  "docker": {
    "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
    "gpu": "nvidia",
    "args": ["--shm-size", "1g"]
  },
  "models": {
    "vision": {
      "docker_image": "my-registry/llama.cpp:server-cuda-custom",
      "docker_args": ["-e", "CUDA_VISIBLE_DEVICES=1"]
    }
  }
}
```

On **Jetson**, use an L4T-based llama.cpp image built for your JetPack version. If the image has no `llama-server` entrypoint, set it:

```json
{
  "docker": {
    "image": "<your-jetson-llama.cpp-image>",
    "gpu": "jetson",
    "entrypoint": "llama-server"
  }
}
```

### Custom llama.cpp builds

To run a fork, a branch, a pull request or your own checkout instead of a prebuilt image, build it and load a model with it.

**Trying a fork** takes two commands — paste the URL from your browser:

```sh
llmctl build https://github.com/someone/llama.cpp/tree/their-branch   # adds build "someone" and builds it
llmctl load qwen --build someone                                       # runs as "qwen-someone", next to your normal qwen
```

`llmctl build` accepts a GitHub repo, branch (`/tree/<branch>`), pull request (`/pull/<N>`) or commit (`/commit/<sha>`) URL, `owner/repo`, any other git URL, or a local path. It names the build after the GitHub owner (`<owner>-pr<N>` for pull requests) and saves it in the `builds` config, so the same command later only rebuilds if the branch moved. Settings can be given as flags: `--ref`, `--backend jetson`, `--cuda-arch 89`, `--cmake-arg -DGGML_CUDA_FA_ALL_QUANTS=ON` (repeatable), `--dockerfile`.

`load --build <name>[@commit]` uses the build for that one load; the instance is named `<model>-<build>` (unless `--name` is given) and keeps the model's other settings (ctx size, extra args). If the fork predates unix-socket support, add `--network internal`.

```sh
llmctl build add ik https://github.com/ikawrakow/ik_llama.cpp   # add under a chosen name without building
llmctl build rm someone                                         # remove the build, its images and fetched source
```

For builds you keep, you can also edit the config directly and point models at them: `llmctl build` fetches the source on the host and builds a llama-server image with an embedded two-stage Dockerfile ([`docker/llama-server.Dockerfile`](docker/llama-server.Dockerfile)).

```json
{
  "builds": {
    "ik":       { "repo": "https://github.com/ikawrakow/ik_llama.cpp", "ref": "main" },
    "upstream": { "repo": "https://github.com/ggml-org/llama.cpp", "ref": "#12345", "cmake_args": ["-DGGML_CUDA_FA_ALL_QUANTS=ON"] },
    "jetson":   { "repo": "https://github.com/ggml-org/llama.cpp", "ref": "b6000", "backend": "jetson" },
    "mine":     { "path": "~/src/llama.cpp" }
  },
  "models": {
    "qwen":      { "docker_build": "ik" },
    "qwen-pin":  { "docker_build": "ik@3f2a9c1" }
  }
}
```

```sh
llmctl build ik                 # build the current commit of ik's ref (skips if already built)
llmctl build ik --ref b6000     # one-off ref override (branch, tag, full SHA or #PR)
llmctl build ik --ref 3f2a9c1   # roll back: re-point :latest at an existing image, no rebuild
llmctl build --all              # build/refresh every build
llmctl builds                   # list builds, latest commit, age, which models use them
```

| Field | Default | Description |
|-------|---------|-------------|
| `repo` / `path` | — | Git URL, or a local git checkout (uncommitted changes included, `.gitignore`d files excluded). Exactly one. |
| `ref` | `HEAD` | Branch, tag, full commit SHA, or `#<number>` for a GitHub pull request (`refs/pull/N/head`) |
| `backend` | `cuda` | `cuda` (desktop/server) or `jetson` (L4T; build on the Jetson itself) |
| `cuda_arch` | host GPU (`nvidia-smi`) / `87` on Jetson | `CMAKE_CUDA_ARCHITECTURES`, e.g. `"89"` or `"86;89;120"` |
| `base_image` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` / L4T JetPack matching `/etc/nv_tegra_release` | Build-stage image |
| `runtime_image` | `nvidia/cuda:12.8.1-runtime-ubuntu24.04` / same as `base_image` | Runtime-stage image |
| `cmake_args` | — | Extra cmake flags (space-separated in the build; values can't contain spaces) |
| `target` | `llama-server` | CMake target to build |
| `dockerfile` | embedded | Use your own Dockerfile (relative to the source tree, e.g. a fork's `.devops/cuda.Dockerfile`) |
| `keep` | `3` | Images kept per build; older ones are removed after a successful build |

How it works:

- **Commit-tagged images.** The ref is resolved with `git ls-remote` on the host (using your git credentials), and the image is tagged `llmctl/<build>:<sha12>` plus `:latest`. If that commit was already built, `llmctl build` just re-points `:latest`. Local checkouts with uncommitted changes are tagged `<sha12>-dirty-<hash>`.
- **Source on the host.** Only the needed commit is shallow-fetched into `~/.cache/llmctl/src/<build>` and used as the build context, so PRs and private forks work, and the image needs no git.
- **Fast rebuilds.** ccache lives in a BuildKit cache mount, so a nightly with a few changed files rebuilds in minutes. Requires BuildKit (default since Docker 23).
- **Static binary.** llama-server is linked statically (`-DBUILD_SHARED_LIBS=OFF`) into a slim runtime image with entrypoint `/app/llama-server`.
- **Unix-socket check.** llmctl checks the fork's server source for unix-socket support and records it on the image. A model whose build lacks it can't use `docker.network: none`; `load` says so and suggests `"docker_network": "internal"`.
- **Retention.** The newest `keep` images per build are kept; `:latest` and images used by a container are never removed.
- **Loading.** A model with `docker_build` runs `llmctl/<build>:latest` (or the pinned commit with `name@commit`). Loads never build: if the image is missing, `load` tells you to run `llmctl build`. `ps` and `info` show which build and commit an instance runs. `docker.build` sets a default build for all models.
- **Network.** Builds need network (base images, apt); only running backends are isolated.

### Autoswitching

Autoswitching lets the proxy start configured models on demand. When the requested model does not fit in the detected GPU memory, the proxy unloads the least-recently-used running models that have `auto_unload: true` until the requested model fits.

Enable it in `~/.llmctl.json`:

```json
{
  "autoswitch": {
    "enabled": true,
    "total_vram_mb": 24576,
    "startup_timeout_sec": 60
  },
  "aliases": {
    "qwen": "Qwen3.5-27B-GGUF:UD-Q4_K_XL.gguf",
    "coder": "Qwen3.5-Coder-GGUF:Q4_K_M.gguf"
  },
  "models": {
    "qwen": {
      "auto_load": true,
      "auto_unload": true
    },
    "coder": {
      "auto_load": true,
      "auto_unload": false,
      "force_auto_load": true,
      "ctx_size": 32768
    }
  }
}
```

`auto_load` allows the proxy to start a model when a request names it. `auto_unload` allows that running instance to be evicted later for another autoswitch request. Models without `auto_unload` are pinned. `force_auto_load` tells the proxy to start the model even when the memory estimate says it will not fit, which is useful when llama.cpp will offload part of the model to CPU.

By default, llmctl estimates memory as GGUF file size plus a small `ctx_size` overhead. If the estimate is wrong for a model, set `vram_mb` on that model to override it. GPU memory is detected automatically on Linux with `nvidia-smi` and on macOS with `system_profiler`; set `autoswitch.total_vram_mb` when detection is unavailable or you want an explicit capacity.

You can also set top-level autoswitch values with:

```sh
llmctl set autoswitch.enabled true
llmctl set autoswitch.total_vram_mb 24576
llmctl set autoswitch.startup_timeout_sec 90
```

### Extra server arguments

Add extra `llama-server` arguments in the config file directly (`~/.llmctl.json`):

```json
{
  "extra_args": ["--flash-attn", "on", "--temp", "0.6"]
}
```

### Per-model overrides

Override settings for specific models by adding a `models` section to `~/.llmctl.json`. Per-model `extra_args` are merged with global `extra_args` — matching flags are replaced, new flags are appended:

```json
{
  "extra_args": ["--flash-attn", "on", "--temp", "0.6"],
  "models": {
    "codellama": {
      "gpu_layers": 20,
      "ctx_size": 8192,
      "extra_args": ["--temp", "0.2"]
    }
  }
}
```

In this example, `codellama` would use `--flash-attn on --temp 0.2` (global `--temp 0.6` is replaced by the per-model override).

### Aliases

Create short names for model files. When loading via an alias, the alias becomes the instance name exposed in the API:

```sh
llmctl alias qwen Qwen3.5-27B-GGUF/UD-Q4_K_XL.gguf
llmctl load qwen
# Instance name is "qwen" — use it in API requests: {"model": "qwen", ...}
```

Aliases also support the `:` separator:

```sh
llmctl alias qwen Qwen3.5-27B-GGUF:UD-Q4_K_XL
```

You can load the same model multiple times with different aliases and per-model overrides:

```json
{
  "aliases": {
    "qwen27b": "Qwen3.5-27B-GGUF:UD-Q4_K_XL",
    "qwen27b_code": "Qwen3.5-27B-GGUF:UD-Q4_K_XL"
  },
  "models": {
    "qwen27b_code": {
      "ctx_size": 32768,
      "extra_args": ["--chat-template-kwargs", "{\"enable_thinking\":true}"]
    }
  }
}
```

```sh
llmctl load qwen27b       # instance "qwen27b", default params
llmctl load qwen27b_code  # instance "qwen27b_code", code-specific params
```

## API endpoints

When the proxy is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (routes by `model` field) |
| `/v1/models` | GET | List loaded models and configured autoloadable models |
| `/health` | GET | Health check |

Requests can also specify a model via the `X-Model` HTTP header.

## Runtime files

| Path | Description |
|------|-------------|
| `~/.llmctl.json` | Configuration |
| `~/.llmctl.registry.json` | Registry of running instances (container names, ports) and proxy PID |
| `~/.llmctl-logs/<name>.log` | Server logs of stopped instances (running ones are read with `docker logs`) |
| `~/.llmctl-run/<container>/llama.sock` | Per-instance unix socket (`docker.network: none`) |

## License

See [LICENSE](LICENSE) for details.

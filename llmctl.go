package main

// Module map (all files remain `package main`, standard library only):
//
//   config.go     — constants, Config/ModelConfig, load/save, merge & validation
//   registry.go   — Instance/Registry persistence and registry query helpers
//   process.go    — process lifecycle primitives (stop, health wait, log tail)
//   models.go     — .gguf/HF discovery, resolution, naming, modelRef helpers
//   autoswitch.go — VRAM accounting and autoswitch eviction plan
//   proxy.go      — OpenAI-compatible reverse proxy and UI handlers
//   load.go       — load options/spec resolution and startInstance lifecycle
//   commands.go   — one function per CLI command
//   helpers.go    — small generic host/flag/format helpers
//   main.go       — usage text and CLI dispatch

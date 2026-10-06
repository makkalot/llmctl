package main

import (
	"fmt"
	"os"
	"runtime"
)

func printUsage() {
	fmt.Printf(`llmctl v%s — multi-model llama.cpp manager with OpenAI proxy

Usage:
  llmctl <command> [args]

Model Management:
  list                        List .gguf models on disk
  pull <user/repo>            Download from Hugging Face
  rm <model>                  Delete a model file
  alias <name> <model>        Create a short alias

Instance Management:
  load <model> [-hf <hf_repo>] [--name NAME] [--mmproj <path>]  Load a model (starts a backend)
  unload <name>                             Stop a model instance
  stop                        Stop everything
  default <name>              Set default model for unmatched requests
  ps                          List loaded instances
  info <name>                 Show detailed info for an instance
  logs <name>                 Show instance logs

Proxy:
  proxy                       Start the OpenAI-compatible proxy
  status                      Overview

Config:
  config                      Show config
  set <key> <value>           Update config

Workflow:
  llmctl load mistral                              # backend on :9100
  llmctl load llama3 --name chat                   # backend on :9101
  llmctl load "" -hf unsloth/Qwen3.5-27B-GGUF     # direct from HF
  llmctl default chat
  llmctl proxy                                     # proxy on :8080

  # Any OpenAI client just works:
  curl http://server:8080/v1/chat/completions \
    -d '{"model":"chat", "messages":[...]}'

  curl http://server:8080/v1/models

Platform: %s/%s
`, appVersion, runtime.GOOS, runtime.GOARCH)
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	cfg := loadConfig()

	switch os.Args[1] {
	case "list", "ls":
		cmdList(cfg)
	case "load", "run":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl load <model> [-hf <huggingface_repo>] [--name NAME] [--mmproj <path>]")
			os.Exit(1)
		}
		name, args := extractFlag(os.Args[3:], "--name")
		hf, _ := extractFlag(args, "-hf")
		mmproj, _ := extractFlag(args, "--mmproj")
		cmdLoad(cfg, os.Args[2], name, hf, mmproj)
	case "unload":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl unload <name>")
			os.Exit(1)
		}
		cmdUnload(cfg, os.Args[2])
	case "stop", "kill":
		cmdStopAll()
	case "default":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl default <name>")
			os.Exit(1)
		}
		cmdDefault(os.Args[2])
	case "ps":
		cmdPS()
	case "info":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl info <name>")
			os.Exit(1)
		}
		cmdInfo(os.Args[2])
	case "proxy", "serve":
		startProxy(cfg)
	case "status":
		cmdStatus(cfg)
	case "logs", "log":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl logs <name>")
			os.Exit(1)
		}
		cmdLogs(os.Args[2])
	case "pull", "download":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl pull <user/repo>")
			os.Exit(1)
		}
		cmdPull(cfg, os.Args[2])
	case "rm", "remove", "delete":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl rm <model>")
			os.Exit(1)
		}
		cmdRM(cfg, os.Args[2])
	case "alias":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl alias <name> <model>")
			os.Exit(1)
		}
		cmdAlias(cfg, os.Args[2], os.Args[3])
	case "config", "cfg":
		cmdConfig(cfg)
	case "set":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl set <key> <value>")
			os.Exit(1)
		}
		cmdSet(cfg, os.Args[2], os.Args[3])
	case "version", "-v", "--version":
		fmt.Printf("llmctl v%s (%s/%s)\n", appVersion, runtime.GOOS, runtime.GOARCH)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\nRun 'llmctl help' for usage.\n", os.Args[1])
		os.Exit(1)
	}
}

package llmctl

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// AppVersion is the fallback version; main overrides it via ldflags.
	AppVersion    = "0.2.0"
	BuildTime     = "" // build timestamp, injected via ldflags (main only)
	defaultPort   = 8080
	defaultHost   = "0.0.0.0"
	registryFile  = ".llmctl.registry.json"
	configFile    = ".llmctl.json"
	defaultModels = "models"
	startPort     = 9100 // internal backend ports start here
)

// ═══════════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════════

type ModelConfig struct {
	ServerBin  *string  `json:"server_bin,omitempty"`
	GpuLayers  *int     `json:"gpu_layers,omitempty"`
	CtxSize    *int     `json:"ctx_size,omitempty"`
	Mmproj     string   `json:"mmproj,omitempty"`
	ExtraArgs  []string `json:"extra_args,omitempty"`
	AutoLoad   bool     `json:"auto_load,omitempty"`
	AutoUnload bool     `json:"auto_unload,omitempty"`
	VramMB     int      `json:"vram_mb,omitempty"`
}

type AutoswitchConfig struct {
	Enabled           bool `json:"enabled,omitempty"`
	TotalVramMB       int  `json:"total_vram_mb,omitempty"`
	StartupTimeoutSec int  `json:"startup_timeout_sec,omitempty"`
}

func validateAutoswitchConfig(cfg Config) error {
	if !cfg.Autoswitch.Enabled {
		return nil
	}
	var errors []string
	for name, mc := range cfg.Models {
		if mc.AutoLoad || mc.AutoUnload {
			if mc.VramMB <= 0 {
				errors = append(errors, fmt.Sprintf("  model %q has auto_load/auto_unload but vram_mb is not set", name))
			}
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("autoswitch config errors:\n%s", strings.Join(errors, "\n"))
	}
	return nil
}

type Config struct {
	ModelsDir  string                 `json:"models_dir"`
	ServerBin  string                 `json:"server_bin"`
	Host       string                 `json:"host"`
	Port       int                    `json:"port"`
	GpuLayers  int                    `json:"gpu_layers"`
	CtxSize    int                    `json:"ctx_size"`
	Mmproj     string                 `json:"mmproj,omitempty"`
	ExtraArgs  []string               `json:"extra_args,omitempty"`
	Aliases    map[string]string      `json:"aliases,omitempty"`
	Models     map[string]ModelConfig `json:"models,omitempty"`
	Autoswitch AutoswitchConfig       `json:"autoswitch,omitempty"`
}

func defaultConfig() Config {
	return Config{
		ModelsDir: modelsDir(),
		ServerBin: findServerBin(),
		Host:      defaultHost,
		Port:      defaultPort,
		GpuLayers: -1,
		CtxSize:   4096,
		Aliases:   map[string]string{},
	}
}

func homeDir() string {
	h, _ := os.UserHomeDir()
	return h
}

func configPath() string   { return filepath.Join(homeDir(), configFile) }
func modelsDir() string    { return filepath.Join(homeDir(), defaultModels) }
func registryPath() string { return filepath.Join(homeDir(), registryFile) }

func LoadConfig() Config {
	cfg := defaultConfig()
	data, err := os.ReadFile(configPath())
	if err != nil {
		return cfg
	}
	_ = json.Unmarshal(data, &cfg)
	if cfg.Aliases == nil {
		cfg.Aliases = map[string]string{}
	}
	if cfg.Models == nil {
		cfg.Models = map[string]ModelConfig{}
	}
	return cfg
}

func saveConfig(cfg Config) error {
	data, _ := json.MarshalIndent(cfg, "", "  ")
	return os.WriteFile(configPath(), data, 0644)
}

func mergeExtraArgs(global, perModel []string) []string {
	type arg struct {
		flag string
		val  string
	}
	parse := func(args []string) []arg {
		var pairs []arg
		for i := 0; i < len(args); i++ {
			if strings.HasPrefix(args[i], "--") {
				a := arg{flag: args[i]}
				if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
					a.val = args[i+1]
					i++
				}
				pairs = append(pairs, a)
			} else {
				pairs = append(pairs, arg{val: args[i]})
			}
		}
		return pairs
	}
	globalPairs := parse(global)
	overridePairs := parse(perModel)

	overrideMap := make(map[string]string)
	var overrideOrder []string
	for _, p := range overridePairs {
		if p.flag != "" {
			if _, exists := overrideMap[p.flag]; !exists {
				overrideOrder = append(overrideOrder, p.flag)
			}
			overrideMap[p.flag] = p.val
		}
	}

	var result []string
	seen := make(map[string]bool)
	for _, p := range globalPairs {
		if p.flag != "" {
			if ov, ok := overrideMap[p.flag]; ok {
				seen[p.flag] = true
				result = append(result, p.flag)
				if ov != "" {
					result = append(result, ov)
				}
				continue
			}
			result = append(result, p.flag)
			if p.val != "" {
				result = append(result, p.val)
			}
			continue
		}
		result = append(result, p.val)
	}

	for _, flag := range overrideOrder {
		if !seen[flag] {
			result = append(result, flag)
			if overrideMap[flag] != "" {
				result = append(result, overrideMap[flag])
			}
		}
	}
	return result
}

func validateExtraArgs(args []string) error {
	suggestions := map[string]string{
		"presence_penalty":     "--presence-penalty",
		"frequency_penalty":    "--frequency-penalty",
		"repetition_penalty":   "--repeat-penalty",
		"repeat_penalty":       "--repeat-penalty",
		"--presence_penalty":   "--presence-penalty",
		"--frequency_penalty":  "--frequency-penalty",
		"--repetition_penalty": "--repeat-penalty",
		"--repeat_penalty":     "--repeat-penalty",
		"--repetition-penalty": "--repeat-penalty",
	}
	for _, a := range args {
		if suggestion, ok := suggestions[a]; ok {
			return fmt.Errorf("invalid extra_args entry %q; llama.cpp flags use dashes, try %q", a, suggestion)
		}
	}
	return nil
}

// configForModel returns a copy of cfg with per-model overrides applied.
// The modelKey is matched exactly against keys in cfg.Models.
func configForModel(cfg Config, modelKey string) Config {
	mc, ok := cfg.Models[modelKey]
	if !ok {
		return cfg
	}
	if mc.ServerBin != nil {
		cfg.ServerBin = *mc.ServerBin
	}
	if mc.GpuLayers != nil {
		cfg.GpuLayers = *mc.GpuLayers
	}
	if mc.CtxSize != nil {
		cfg.CtxSize = *mc.CtxSize
	}
	if mc.Mmproj != "" {
		cfg.Mmproj = mc.Mmproj
	}
	if mc.ExtraArgs != nil {
		cfg.ExtraArgs = mergeExtraArgs(cfg.ExtraArgs, mc.ExtraArgs)
	}
	return cfg
}

func modelConfigKey(cfg Config, requestedName, instanceName, modelPath string) string {
	canonicalRef := modelRef(cfg.ModelsDir, modelPath)
	base := filepath.Base(modelPath)
	baseNoExt := strings.TrimSuffix(strings.TrimSuffix(base, ".gguf"), ".GGUF")

	candidates := []string{
		requestedName,
		instanceName,
		canonicalRef,
		strings.TrimSuffix(strings.TrimSuffix(canonicalRef, ".gguf"), ".GGUF"),
		base,
		baseNoExt,
	}
	for _, c := range candidates {
		if _, ok := cfg.Models[c]; ok {
			return c
		}
	}

	var aliasMatches []string
	for alias, target := range cfg.Aliases {
		if _, ok := cfg.Models[alias]; !ok {
			continue
		}
		if aliasTargetMatchesModel(cfg, target, modelPath) {
			aliasMatches = append(aliasMatches, alias)
		}
	}
	sort.Strings(aliasMatches)
	if len(aliasMatches) > 0 {
		return aliasMatches[0]
	}
	return ""
}

func aliasTargetMatchesModel(cfg Config, target, modelPath string) bool {
	if resolved, err := resolveModel(cfg, target); err == nil {
		return filepath.Clean(resolved) == filepath.Clean(modelPath)
	}

	normalizedTarget := strings.ToLower(strings.TrimSuffix(strings.TrimSuffix(target, ".gguf"), ".GGUF"))
	refs := []string{
		modelRef(cfg.ModelsDir, modelPath),
		filepath.Base(modelPath),
		strings.TrimSuffix(strings.TrimSuffix(filepath.Base(modelPath), ".gguf"), ".GGUF"),
	}
	for _, ref := range refs {
		normalizedRef := strings.ToLower(strings.TrimSuffix(strings.TrimSuffix(ref, ".gguf"), ".GGUF"))
		if normalizedTarget == normalizedRef {
			return true
		}
	}
	return false
}

func findServerBin() string {
	for _, name := range []string{"llama-server", "llama-cpp-server", "server"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return "llama-server"
}

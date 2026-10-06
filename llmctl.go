package main

import (
	"embed"
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	appVersion    = "0.2.0"
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

func loadConfig() Config {
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

// ═══════════════════════════════════════════════════════════════════════════
// Instance Registry — tracks multiple running backends
// ═══════════════════════════════════════════════════════════════════════════

type Instance struct {
	Name            string   `json:"name"`
	Model           string   `json:"model"` // full path to .gguf
	Mmproj          string   `json:"mmproj,omitempty"`
	PID             int      `json:"pid"`
	Port            int      `json:"port"`
	IsDefault       bool     `json:"is_default"` // default model for unmatched requests
	CtxSize         int      `json:"ctx_size"`
	GpuLayers       int      `json:"gpu_layers"`
	ServerBin       string   `json:"server_bin"`
	ExtraArgs       []string `json:"extra_args,omitempty"`
	Aliases         []string `json:"aliases,omitempty"`
	AutoUnload      bool     `json:"auto_unload,omitempty"`
	EstimatedVramMB int      `json:"estimated_vram_mb,omitempty"`
	LastUsedAt      int64    `json:"last_used_at,omitempty"`
}

type Registry struct {
	Instances []Instance `json:"instances"`
	ProxyPID  int        `json:"proxy_pid,omitempty"`
}

func loadRegistry() Registry {
	var reg Registry
	data, err := os.ReadFile(registryPath())
	if err != nil {
		return reg
	}
	_ = json.Unmarshal(data, &reg)
	return reg
}

func saveRegistry(reg Registry) error {
	data, _ := json.MarshalIndent(reg, "", "  ")
	return os.WriteFile(registryPath(), data, 0644)
}

func (r *Registry) FindByName(name string) *Instance {
	for i := range r.Instances {
		if r.Instances[i].Name == name {
			return &r.Instances[i]
		}
	}
	return nil
}

func (r *Registry) Default() *Instance {
	for i := range r.Instances {
		if r.Instances[i].IsDefault {
			return &r.Instances[i]
		}
	}
	for i := range r.Instances {
		if isRunning(r.Instances[i].PID) {
			return &r.Instances[i]
		}
	}
	return nil
}

func (r *Registry) Remove(name string) {
	var filtered []Instance
	for _, inst := range r.Instances {
		if inst.Name != name {
			filtered = append(filtered, inst)
		}
	}
	r.Instances = filtered
}

func (r *Registry) CleanDead() {
	var alive []Instance
	for _, inst := range r.Instances {
		if isRunning(inst.PID) {
			alive = append(alive, inst)
		}
	}
	r.Instances = alive
}

func (r *Registry) NextPort() int {
	used := map[int]bool{}
	for _, inst := range r.Instances {
		used[inst.Port] = true
	}
	for p := startPort; ; p++ {
		if !used[p] {
			return p
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Process helpers
// ═══════════════════════════════════════════════════════════════════════════

func isRunning(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func stopProcess(pid int) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return
	}
	proc.Signal(syscall.SIGTERM)
	for i := 0; i < 50; i++ {
		if !isRunning(pid) {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	proc.Signal(syscall.SIGKILL)
	time.Sleep(200 * time.Millisecond)
}

func waitForHealth(port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	for time.Now().Before(deadline) {
		resp, err := http.Get(addr)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false
}

func printLogTail(path string, maxLines int) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return
	}
	start := 0
	if len(lines) > maxLines {
		start = len(lines) - maxLines
	}
	fmt.Fprintln(os.Stderr, "Last log lines:")
	for _, line := range lines[start:] {
		fmt.Fprintln(os.Stderr, "  "+line)
	}
}

func backendExited(exitCh <-chan error) (error, bool) {
	select {
	case err := <-exitCh:
		return err, true
	default:
		return nil, false
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Model helpers
// ═══════════════════════════════════════════════════════════════════════════

func listModelFiles(dir string) []string {
	var models []string
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			models = append(models, e.Name())
		} else if e.IsDir() && strings.HasPrefix(e.Name(), "models--") {
			subPath := filepath.Join(dir, e.Name())
			models = append(models, listModelFilesRecursive(subPath, e.Name())...)
		}
	}
	// Also scan the hub/ subdirectory used by huggingface_hub's cache
	hubDir := filepath.Join(dir, "hub")
	hubEntries, _ := os.ReadDir(hubDir)
	for _, e := range hubEntries {
		if e.IsDir() && strings.HasPrefix(e.Name(), "models--") {
			subPath := filepath.Join(hubDir, e.Name())
			models = append(models, listModelFilesRecursive(subPath, filepath.Join("hub", e.Name()))...)
		}
	}
	sort.Strings(models)
	return models
}

func listModelFilesRecursive(dir, prefix string) []string {
	var models []string
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		full := filepath.Join(dir, e.Name())
		if e.IsDir() && e.Name() == "snapshots" {
			snapshots, _ := os.ReadDir(full)
			for _, s := range snapshots {
				snapDir := filepath.Join(full, s.Name())
				files, _ := os.ReadDir(snapDir)
				for _, f := range files {
					if strings.HasSuffix(strings.ToLower(f.Name()), ".gguf") {
						models = append(models, filepath.Join(prefix, "snapshots", s.Name(), f.Name()))
					}
				}
			}
		} else if e.IsDir() {
			models = append(models, listModelFilesRecursive(full, filepath.Join(prefix, e.Name()))...)
		}
	}
	return models
}

type hfModelRef struct {
	User string
	Repo string
	File string
}

func resolveModel(cfg Config, name string) (string, error) {
	if resolved, ok := cfg.Aliases[name]; ok {
		name = resolved
	}
	if filepath.IsAbs(name) {
		if _, err := os.Stat(name); err == nil {
			return name, nil
		}
		return "", fmt.Errorf("model not found: %s", name)
	}

	origName := name

	if ref, ok := parseHFModelRef(name); ok {
		matches := findHFSnapshots(cfg.ModelsDir, ref)
		if len(matches) == 1 {
			return matches[0], nil
		}
		if len(matches) > 1 {
			return "", ambiguousModelError(origName, cfg.ModelsDir, matches)
		}
	}

	if strings.HasPrefix(name, "models--") || strings.HasPrefix(name, "hub"+string(filepath.Separator)+"models--") || strings.HasPrefix(name, "hub/models--") {
		if path := findInHFCache(cfg.ModelsDir, filepath.ToSlash(name)); path != "" {
			return path, nil
		}
	}

	if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
		name = name + ".gguf"
	}
	full := filepath.Join(cfg.ModelsDir, name)
	if _, err := os.Stat(full); err == nil {
		return full, nil
	}

	// Fuzzy substring match: also try matching just the filename part
	// (after last / or :) against model files.
	lower := strings.ToLower(strings.TrimSuffix(name, ".gguf"))
	basename := lower
	for _, sep := range []string{"/", ":"} {
		if idx := strings.LastIndex(lower, sep); idx >= 0 {
			candidate := lower[idx+1:]
			if len(candidate) < len(basename) {
				basename = candidate
			}
		}
	}
	var matches []string
	for _, m := range listModelFiles(cfg.ModelsDir) {
		ml := strings.ToLower(m)
		if strings.Contains(ml, lower) || strings.Contains(ml, basename) {
			matches = append(matches, filepath.Join(cfg.ModelsDir, m))
		}
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) > 1 {
		return "", ambiguousModelError(origName, cfg.ModelsDir, matches)
	}
	return "", fmt.Errorf("model not found: %s (looked in %s)", origName, cfg.ModelsDir)
}

func parseHFModelRef(name string) (hfModelRef, bool) {
	name = strings.TrimPrefix(name, "https://huggingface.co/")
	name = strings.TrimSuffix(name, "/")
	slash := strings.Index(name, "/")
	if slash <= 0 || slash == len(name)-1 {
		return hfModelRef{}, false
	}
	user := name[:slash]
	rest := name[slash+1:]
	repo := rest
	file := ""
	if colon := strings.Index(rest, ":"); colon >= 0 {
		repo = rest[:colon]
		file = rest[colon+1:]
	} else if slash := strings.Index(rest, "/"); slash >= 0 {
		repo = rest[:slash]
		file = rest[slash+1:]
	}
	if user == "" || repo == "" {
		return hfModelRef{}, false
	}
	return hfModelRef{User: user, Repo: repo, File: file}, true
}

func hfCachePrefix(ref hfModelRef) string {
	return "models--" + ref.User + "--" + ref.Repo
}

func findHFSnapshots(modelsDir string, ref hfModelRef) []string {
	cachePrefix := hfCachePrefix(ref)
	searchBases := []string{
		modelsDir,
		filepath.Join(modelsDir, "hub"),
	}
	var matches []string
	seenRefs := map[string]bool{}
	for _, base := range searchBases {
		snapshotsDir := filepath.Join(base, cachePrefix, "snapshots")
		snapshots, err := os.ReadDir(snapshotsDir)
		if err != nil {
			continue
		}
		for _, s := range snapshots {
			snapDir := filepath.Join(snapshotsDir, s.Name())
			files, _ := os.ReadDir(snapDir)
			for _, f := range files {
				if !strings.HasSuffix(strings.ToLower(f.Name()), ".gguf") {
					continue
				}
				if ref.File == "" || f.Name() == ref.File {
					path := filepath.Join(snapDir, f.Name())
					canonicalRef := modelRef(modelsDir, path)
					if !seenRefs[canonicalRef] {
						seenRefs[canonicalRef] = true
						matches = append(matches, path)
					}
				}
			}
		}
	}
	sort.Strings(matches)
	return matches
}

func ambiguousModelError(name, modelsDir string, matches []string) error {
	refs := make([]string, 0, len(matches))
	for _, m := range matches {
		refs = append(refs, modelRef(modelsDir, m))
	}
	sort.Strings(refs)
	return fmt.Errorf("ambiguous model %q; matches: %s", name, strings.Join(refs, ", "))
}

func findInHFCache(modelsDir, name string) string {
	name = strings.TrimPrefix(filepath.ToSlash(name), "hub/")
	parts := strings.SplitN(name, "/", 2)
	if len(parts) < 2 {
		return ""
	}
	// Search both <modelsDir>/<cacheDir> and <modelsDir>/hub/<cacheDir>
	// since huggingface_hub stores caches under a hub/ subdirectory.
	candidates := []string{
		filepath.Join(modelsDir, parts[0]),
		filepath.Join(modelsDir, "hub", parts[0]),
	}
	for _, cacheDir := range candidates {
		snapshotsDir := filepath.Join(cacheDir, "snapshots")
		entries, err := os.ReadDir(snapshotsDir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			snapPath := filepath.Join(snapshotsDir, e.Name())
			files, _ := os.ReadDir(snapPath)
			for _, f := range files {
				if f.Name() == parts[1] {
					return filepath.Join(snapPath, f.Name())
				}
			}
		}
	}
	return ""
}

func modelRef(modelsDir, path string) string {
	rel := path
	if filepath.IsAbs(path) {
		if r, err := filepath.Rel(modelsDir, path); err == nil && !strings.HasPrefix(r, "..") {
			rel = r
		}
	}
	rel = filepath.ToSlash(rel)
	parts := strings.Split(rel, "/")
	offset := 0
	if len(parts) > 0 && parts[0] == "hub" {
		offset = 1
	}
	if len(parts) >= offset+4 && strings.HasPrefix(parts[offset], "models--") && parts[offset+1] == "snapshots" {
		cacheParts := strings.SplitN(strings.TrimPrefix(parts[offset], "models--"), "--", 2)
		if len(cacheParts) == 2 {
			return cacheParts[0] + "/" + cacheParts[1] + ":" + strings.Join(parts[offset+3:], "/")
		}
	}
	return filepath.Base(path)
}

func shortName(path string) string { return filepath.Base(path) }

func fileSizeStr(path string) string {
	info, err := os.Stat(path)
	if err != nil {
		return "?"
	}
	mb := float64(info.Size()) / 1024 / 1024
	if mb > 1024 {
		return fmt.Sprintf("%.1f GB", mb/1024)
	}
	return fmt.Sprintf("%.0f MB", mb)
}

// Derive a clean instance name from the model filename or canonical HF ref.
func deriveInstanceName(modelRef string) string {
	base := filepath.Base(modelRef)
	if idx := strings.LastIndex(base, ":"); idx >= 0 {
		base = base[idx+1:]
	}
	base = strings.TrimSuffix(base, ".gguf")
	base = strings.TrimSuffix(base, ".GGUF")
	// Strip quantization suffix like .Q4_K_M
	parts := strings.Split(base, ".")
	if len(parts) > 1 {
		last := strings.ToUpper(parts[len(parts)-1])
		if len(last) > 0 && last[0] == 'Q' {
			base = strings.Join(parts[:len(parts)-1], ".")
		}
	}
	base = strings.Map(func(r rune) rune {
		if r == ' ' || r == '/' || r == '\\' || r == ':' {
			return '-'
		}
		return r
	}, base)
	return strings.ToLower(base)
}

// ═══════════════════════════════════════════════════════════════════════════
// Autoswitch helpers
// ═══════════════════════════════════════════════════════════════════════════

func autoswitchStartupTimeout(cfg Config) time.Duration {
	if cfg.Autoswitch.StartupTimeoutSec > 0 {
		return time.Duration(cfg.Autoswitch.StartupTimeoutSec) * time.Second
	}
	return 60 * time.Second
}

// gpuTotalVramMB returns total VRAM of GPU 0 from nvidia-smi, or the config override.
func gpuTotalVramMB(cfg Config) (int, error) {
	if cfg.Autoswitch.TotalVramMB > 0 {
		return cfg.Autoswitch.TotalVramMB, nil
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, fmt.Errorf("cannot detect GPU VRAM (nvidia-smi failed: %v); set autoswitch.total_vram_mb in config", err)
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return 0, fmt.Errorf("nvidia-smi returned empty output; set autoswitch.total_vram_mb in config")
	}
	val, err := strconv.Atoi(strings.TrimSpace(lines[0]))
	if err != nil {
		return 0, fmt.Errorf("cannot parse nvidia-smi output %q; set autoswitch.total_vram_mb in config", lines[0])
	}
	if val <= 0 {
		return 0, fmt.Errorf("nvidia-smi reported %d MB for GPU 0, refusing; set autoswitch.total_vram_mb in config", val)
	}
	return val, nil
}

// gpuUsedVramMB returns currently used VRAM on GPU 0 from nvidia-smi.
func gpuUsedVramMB() (int, error) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, fmt.Errorf("cannot read GPU usage (nvidia-smi failed: %v)", err)
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return 0, fmt.Errorf("nvidia-smi returned empty output for memory.used")
	}
	return strconv.Atoi(strings.TrimSpace(lines[0]))
}

// autoswitchAvailableMB returns available VRAM on GPU 0.
func autoswitchAvailableMB(cfg Config) (int, error) {
	total, err := gpuTotalVramMB(cfg)
	if err != nil {
		return 0, err
	}
	used, err := gpuUsedVramMB()
	if err != nil {
		return 0, err
	}
	avail := total - used
	if avail < 0 {
		avail = 0
	}
	return avail, nil
}

func formatRegistryTime(ts int64) string {
	if ts > 1_000_000_000_000 {
		return time.Unix(0, ts).Format(time.RFC3339)
	}
	return time.Unix(ts, 0).Format(time.RFC3339)
}

// autoswitchEvictionPlan checks if requiredMB fits in availableMB.
// If not, returns a list of auto_unload instances to evict (sorted oldest first).
func autoswitchEvictionPlan(reg Registry, targetName string, requiredMB, availableMB int) ([]Instance, bool) {
	if availableMB >= requiredMB {
		return nil, true
	}

	candidates := make([]Instance, 0)
	for _, inst := range reg.Instances {
		if inst.Name != targetName && inst.AutoUnload {
			candidates = append(candidates, inst)
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].LastUsedAt == candidates[j].LastUsedAt {
			return candidates[i].Name < candidates[j].Name
		}
		return candidates[i].LastUsedAt < candidates[j].LastUsedAt
	})

	return candidates, len(candidates) > 0
}

func shouldFallbackToDefault(targetName string, autoswitchErr error) bool {
	return targetName == "" || autoswitchErr == nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Reverse Proxy — routes by "model" field in OpenAI requests
// ═══════════════════════════════════════════════════════════════════════════

func hasAutoLoadModels(cfg Config) bool {
	if !cfg.Autoswitch.Enabled {
		return false
	}
	for _, mc := range cfg.Models {
		if mc.AutoLoad {
			return true
		}
	}
	return false
}

func writeOpenAIError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"message": msg,
			"type":    "invalid_request_error",
		},
	})
}

func markInstanceUsed(name string) {
	reg := loadRegistry()
	for i := range reg.Instances {
		if reg.Instances[i].Name == name {
			reg.Instances[i].LastUsedAt = time.Now().UnixNano()
			saveRegistry(reg)
			return
		}
	}
}

func autoswitchModel(cfg Config, targetName string) (*url.URL, string, error) {
	if !cfg.Autoswitch.Enabled {
		return nil, "", fmt.Errorf("model %q is not loaded", targetName)
	}

	spec, err := resolveLoadSpec(cfg, loadOptions{ModelName: targetName})
	if err != nil {
		return nil, "", err
	}
	if spec.ConfigKey == "" {
		return nil, "", fmt.Errorf("model %q is not configured for autoswitch", targetName)
	}
	mc := cfg.Models[spec.ConfigKey]
	if !mc.AutoLoad {
		return nil, "", fmt.Errorf("model %q is not configured with auto_load", targetName)
	}
	requiredMB := mc.VramMB

	reg := loadRegistry()
	reg.CleanDead()

	availableMB, err := autoswitchAvailableMB(cfg)
	if err != nil {
		return nil, "", err
	}

	if availableMB < requiredMB {
		candidates, hasCandidates := autoswitchEvictionPlan(reg, spec.InstanceName, requiredMB, availableMB)
		if !hasCandidates {
			return nil, "", fmt.Errorf("model %q needs %d MB but only %d MB available on GPU 0, no auto_unload models to evict", targetName, requiredMB, availableMB)
		}
		evicted := 0
		for _, inst := range candidates {
			if err := unloadInstance(inst.Name, false); err != nil {
				return nil, "", fmt.Errorf("failed to evict %q: %w", inst.Name, err)
			}
			evicted++
			newAvail, checkErr := autoswitchAvailableMB(cfg)
			if checkErr == nil && newAvail >= requiredMB {
				break
			}
		}
		availableMB, err = autoswitchAvailableMB(cfg)
		if err != nil {
			return nil, "", err
		}
		if availableMB < requiredMB {
			return nil, "", fmt.Errorf("model %q needs %d MB but only %d MB available after evicting %d auto_unload models", targetName, requiredMB, availableMB, evicted)
		}
	}

	inst, healthy, err := loadInstance(cfg, loadOptions{
		ModelName:       targetName,
		WaitTimeout:     autoswitchStartupTimeout(cfg),
		Verbose:         false,
		AutoUnload:      mc.AutoUnload,
		EstimatedVramMB: requiredMB,
	})
	if err != nil {
		return nil, "", err
	}
	if !healthy {
		stopProcess(inst.PID)
		reg := loadRegistry()
		reg.Remove(inst.Name)
		saveRegistry(reg)
		return nil, "", fmt.Errorf("model %q did not become healthy before timeout", inst.Name)
	}
	u, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", inst.Port))
	return u, inst.Name, nil
}

//go:embed web/index.html
var webFS embed.FS

var proxyEvents []struct{ TS, Msg string }
var eventsMu sync.Mutex

func addEvent(msg string) {
	eventsMu.Lock()
	defer eventsMu.Unlock()
	proxyEvents = append(proxyEvents, struct{ TS, Msg string }{time.Now().Format("15:04:05"), msg})
	if len(proxyEvents) > 200 {
		proxyEvents = proxyEvents[len(proxyEvents)-200:]
	}
}

func jsonResp(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func handleUIVRAM(w http.ResponseWriter, cfg Config) {
	total, _ := gpuTotalVramMB(cfg)
	used, _ := gpuUsedVramMB()
	avail := total - used
	if avail < 0 {
		avail = 0
	}
	jsonResp(w, map[string]int{"total": total, "used": used, "avail": avail})
}

func handleUIModels(w http.ResponseWriter, cfg Config) {
	reg := loadRegistry()
	reg.CleanDead()
	type row struct {
		Name    string `json:"name"`
		Running bool   `json:"running"`
		Vram    int    `json:"vram"`
		Port    int    `json:"port"`
	}

	// The universe of models shown in the UI is exactly what the user
	// configured: every alias plus every key in the models map. We do NOT
	// scan the disk, so arbitrary .gguf files (or unknown subdirectory
	// layouts) never leak into the dashboard.
	names := make([]string, 0, len(cfg.Aliases)+len(cfg.Models))
	seen := make(map[string]bool, len(cfg.Aliases)+len(cfg.Models))
	for name := range cfg.Aliases {
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	for name := range cfg.Models {
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	sort.Strings(names)

	// Registry instances keyed by name so loaded models can be enriched
	// with port/running/vram from the live process.
	instByName := make(map[string]*Instance, len(reg.Instances))
	for i := range reg.Instances {
		instByName[reg.Instances[i].Name] = &reg.Instances[i]
	}

	out := make([]row, 0, len(names))
	for _, name := range names {
		if inst := instByName[name]; inst != nil {
			vram := inst.EstimatedVramMB
			if vram == 0 {
				vram = cfg.Models[name].VramMB
			}
			out = append(out, row{name, isRunning(inst.PID), vram, inst.Port})
			continue
		}
		// Configured but not currently loaded.
		out = append(out, row{Name: name, Running: false, Vram: cfg.Models[name].VramMB})
	}

	jsonResp(w, out)
}

func handleUILoad(w http.ResponseWriter, r *http.Request, cfg Config) {
	name := strings.TrimPrefix(r.URL.Path, "/api/ui/load/")
	addEvent("Loading model " + name)
	_, _, err := autoswitchModel(cfg, name)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		jsonResp(w, map[string]string{"error": err.Error()})
		return
	}
	addEvent("Loaded model " + name)
	jsonResp(w, map[string]string{"status": "loaded"})
}

func handleUIUnload(w http.ResponseWriter, r *http.Request, cfg Config) {
	name := strings.TrimPrefix(r.URL.Path, "/api/ui/unload/")
	addEvent("Unloading model " + name)
	reg := loadRegistry()
	inst := reg.FindByName(name)
	if inst == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		jsonResp(w, map[string]string{"error": "model not found"})
		return
	}
	stopProcess(inst.PID)
	reg.Remove(name)
	saveRegistry(reg)
	addEvent("Unloaded model " + name)
	jsonResp(w, map[string]string{"status": "unloaded"})
}

func handleUILogs(w http.ResponseWriter, cfg Config) {
	eventsMu.Lock()
	defer eventsMu.Unlock()
	type evt struct{ TS, Msg string }
	out := make([]evt, len(proxyEvents))
	for i, e := range proxyEvents {
		out[i] = evt{e.TS, e.Msg}
	}
	jsonResp(w, out)
}

func startProxy(cfg Config) {
	reg := loadRegistry()
	reg.CleanDead()

	if err := validateAutoswitchConfig(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	if len(reg.Instances) == 0 && !hasAutoLoadModels(cfg) {
		fmt.Fprintln(os.Stderr, "Error: no models loaded. Use `llmctl load <model>` first.")
		os.Exit(1)
	}

	var mu sync.RWMutex
	backends := map[string]*url.URL{}

	rebuildBackends := func() {
		mu.Lock()
		defer mu.Unlock()
		r := loadRegistry()
		r.CleanDead()
		for k := range backends {
			delete(backends, k)
		}
		for _, inst := range r.Instances {
			u, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", inst.Port))
			backends[inst.Name] = u
		}
	}
	rebuildBackends()

	// Refresh routing table periodically
	go func() {
		for {
			time.Sleep(5 * time.Second)
			rebuildBackends()
		}
	}()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// GET /v1/models — OpenAI compatible model listing
		if r.URL.Path == "/v1/models" && r.Method == "GET" {
			handleListModels(w, cfg)
			return
		}

		// GET /health
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"ok"}`))
			return
		}

		// Web UI
		if r.URL.Path == "/ui" || r.URL.Path == "/ui/" {
			data, _ := webFS.ReadFile("web/index.html")
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(data)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/ui/vram") {
			handleUIVRAM(w, cfg)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/ui/models") {
			handleUIModels(w, cfg)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/ui/load/") && r.Method == "POST" {
			handleUILoad(w, r, cfg)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/ui/unload/") && r.Method == "POST" {
			handleUIUnload(w, r, cfg)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/ui/logs") {
			handleUILogs(w, cfg)
			return
		}

		// Extract "model" from POST body
		var targetName string
		var bodyBytes []byte

		if r.Method == "POST" {
			var err error
			bodyBytes, err = io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, "failed to read body", 500)
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

			var parsed struct {
				Model string `json:"model"`
			}
			if json.Unmarshal(bodyBytes, &parsed) == nil && parsed.Model != "" {
				targetName = parsed.Model
			}
		}

		// Fallback: X-Model header
		if targetName == "" {
			targetName = r.Header.Get("X-Model")
		}

		mu.RLock()
		target, ok := backends[targetName]
		routedName := targetName

		// Fuzzy match
		if !ok && targetName != "" {
			lower := strings.ToLower(targetName)
			for name, u := range backends {
				if strings.Contains(strings.ToLower(name), lower) {
					target = u
					ok = true
					routedName = name
					break
				}
			}
		}
		mu.RUnlock()

		if ok {
			markInstanceUsed(routedName)
		}

		var autoswitchErr error
		if !ok && targetName != "" {
			u, name, err := autoswitchModel(cfg, targetName)
			if err == nil {
				target = u
				ok = true
				routedName = name
				rebuildBackends()
				markInstanceUsed(routedName)
			} else {
				autoswitchErr = err
			}
		}

		// Fallback to default
		if !ok && shouldFallbackToDefault(targetName, autoswitchErr) {
			reg := loadRegistry()
			if def := reg.Default(); def != nil {
				target, _ = url.Parse(fmt.Sprintf("http://127.0.0.1:%d", def.Port))
				ok = true
				routedName = def.Name
				markInstanceUsed(routedName)
			}
		}

		if !ok {
			if autoswitchErr != nil {
				writeOpenAIError(w, 503, autoswitchErr.Error())
				return
			}
			writeOpenAIError(w, 404, fmt.Sprintf("model '%s' not loaded. Run: llmctl load <model>", targetName))
			return
		}

		// Proxy the request
		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				req.URL.Scheme = target.Scheme
				req.URL.Host = target.Host
				req.Host = target.Host
			},
		}
		proxy.ServeHTTP(w, r)
	})

	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	fmt.Printf("Proxy listening on %s\n", addr)
	fmt.Printf("  API: http://%s:%d/v1\n", displayHost(cfg.Host), cfg.Port)
	fmt.Printf("  UI:  http://%s:%d/ui\n", displayHost(cfg.Host), cfg.Port)
	addEvent("Proxy started")
	fmt.Println("  Routes:")

	mu.RLock()
	routeCount := len(backends)
	for name, u := range backends {
		reg := loadRegistry()
		def := ""
		if inst := reg.FindByName(name); inst != nil && inst.IsDefault {
			def = " (default)"
		}
		fmt.Printf("    %-28s → %s%s\n", name, u.String(), def)
	}
	mu.RUnlock()
	if routeCount == 0 && hasAutoLoadModels(cfg) {
		fmt.Println("    (no loaded models; autoswitch will load configured models on demand)")
	}

	fmt.Println("\n  Ctrl+C to stop.")

	// Save proxy PID
	reg = loadRegistry()
	reg.ProxyPID = os.Getpid()
	saveRegistry(reg)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\n✓ Proxy stopped.")
		reg := loadRegistry()
		reg.ProxyPID = 0
		saveRegistry(reg)
		os.Exit(0)
	}()

	if err := http.ListenAndServe(addr, handler); err != nil {
		fmt.Fprintf(os.Stderr, "Proxy error: %v\n", err)
		os.Exit(1)
	}
}

func handleListModels(w http.ResponseWriter, cfg Config) {
	reg := loadRegistry()
	reg.CleanDead()

	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var models []modelObj
	seen := map[string]bool{}
	for _, inst := range reg.Instances {
		models = append(models, modelObj{
			ID:      inst.Name,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "llmctl",
		})
		seen[inst.Name] = true
	}
	for name, mc := range cfg.Models {
		if !mc.AutoLoad || seen[name] {
			continue
		}
		models = append(models, modelObj{
			ID:      name,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "llmctl",
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI Commands
// ═══════════════════════════════════════════════════════════════════════════

func findMmprojForModel(modelPath string) string {
	dir := filepath.Dir(modelPath)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		lower := strings.ToLower(e.Name())
		if strings.HasPrefix(lower, "mmproj-") && strings.HasSuffix(lower, ".gguf") {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

func resolveMmproj(cfg Config, modelPath, mmprojName string) (string, error) {
	if mmprojName == "" {
		return findMmprojForModel(modelPath), nil
	}
	if filepath.IsAbs(mmprojName) {
		if _, err := os.Stat(mmprojName); err == nil {
			return mmprojName, nil
		}
		return "", fmt.Errorf("mmproj not found: %s", mmprojName)
	}

	nearModel := filepath.Join(filepath.Dir(modelPath), mmprojName)
	if _, err := os.Stat(nearModel); err == nil {
		return nearModel, nil
	}

	inModelsDir := filepath.Join(cfg.ModelsDir, mmprojName)
	if _, err := os.Stat(inModelsDir); err == nil {
		return inModelsDir, nil
	}

	path, err := resolveModel(cfg, mmprojName)
	if err != nil {
		return "", fmt.Errorf("mmproj %q not resolved near %s: %w", mmprojName, modelRef(cfg.ModelsDir, modelPath), err)
	}
	return path, nil
}

type loadOptions struct {
	ModelName       string
	InstanceName    string
	HFRepo          string
	MmprojArg       string
	WaitTimeout     time.Duration
	Verbose         bool
	AutoUnload      bool
	EstimatedVramMB int
}

type loadSpec struct {
	Config       Config
	ModelName    string
	InstanceName string
	ModelPath    string
	HFRepo       string
	MmprojPath   string
	AliasUsed    string
	ConfigKey    string
}

func resolveLoadSpec(cfg Config, opts loadOptions) (loadSpec, error) {
	var modelPath string
	var err error

	if opts.HFRepo != "" {
		modelPath = opts.HFRepo
	} else {
		modelPath, err = resolveModel(cfg, opts.ModelName)
		if err != nil {
			return loadSpec{}, err
		}
	}

	aliasUsed := ""
	if _, ok := cfg.Aliases[opts.ModelName]; ok {
		aliasUsed = opts.ModelName
	}

	instanceName := opts.InstanceName
	if instanceName == "" {
		if aliasUsed != "" {
			instanceName = aliasUsed
		} else {
			instanceName = deriveInstanceName(modelRef(cfg.ModelsDir, modelPath))
		}
	}

	configKey := modelConfigKey(cfg, opts.ModelName, instanceName, modelPath)
	if configKey != "" {
		cfg = configForModel(cfg, configKey)
	}

	var mmprojPath string
	if opts.MmprojArg != "" {
		mmprojPath, err = resolveMmproj(cfg, modelPath, opts.MmprojArg)
		if err != nil {
			return loadSpec{}, err
		}
	} else if cfg.Mmproj != "" {
		mmprojPath, err = resolveMmproj(cfg, modelPath, cfg.Mmproj)
		if err != nil {
			return loadSpec{}, err
		}
	} else {
		mmprojPath, _ = resolveMmproj(cfg, modelPath, "")
	}

	return loadSpec{
		Config:       cfg,
		ModelName:    opts.ModelName,
		InstanceName: instanceName,
		ModelPath:    modelPath,
		HFRepo:       opts.HFRepo,
		MmprojPath:   mmprojPath,
		AliasUsed:    aliasUsed,
		ConfigKey:    configKey,
	}, nil
}

func loadInstance(cfg Config, opts loadOptions) (Instance, bool, error) {
	spec, err := resolveLoadSpec(cfg, opts)
	if err != nil {
		return Instance{}, false, err
	}
	cfg = spec.Config
	autoUnload := opts.AutoUnload
	estimatedVramMB := opts.EstimatedVramMB
	if spec.ConfigKey != "" {
		if mc, ok := cfg.Models[spec.ConfigKey]; ok {
			if !autoUnload {
				autoUnload = mc.AutoUnload
			}
			if estimatedVramMB == 0 && mc.VramMB > 0 {
				estimatedVramMB = mc.VramMB
			}
		}
	}

	reg := loadRegistry()
	reg.CleanDead()

	if existing := reg.FindByName(spec.InstanceName); existing != nil {
		if isRunning(existing.PID) {
			if opts.Verbose {
				fmt.Printf("Replacing instance '%s'...\n", spec.InstanceName)
			}
			stopProcess(existing.PID)
		}
		reg.Remove(spec.InstanceName)
	}

	backendPort := reg.NextPort()
	displayName := spec.ModelPath
	if spec.HFRepo != "" {
		displayName = spec.HFRepo
	}
	if opts.Verbose {
		fmt.Printf("Loading %s as '%s' (backend :%d)...\n", shortName(displayName), spec.InstanceName, backendPort)
	}
	if err := validateExtraArgs(cfg.ExtraArgs); err != nil {
		return Instance{}, false, err
	}

	args := []string{}
	if spec.HFRepo != "" {
		args = append(args, "-hf", spec.HFRepo)
	} else {
		args = append(args, "--model", spec.ModelPath)
	}
	args = append(args,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(backendPort),
		"--ctx-size", strconv.Itoa(cfg.CtxSize),
	)
	if cfg.GpuLayers != 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(cfg.GpuLayers))
	}
	if spec.MmprojPath != "" {
		args = append(args, "--mmproj", spec.MmprojPath)
		if opts.Verbose {
			fmt.Printf("  mmproj: %s\n", shortName(spec.MmprojPath))
		}
	}
	args = append(args, cfg.ExtraArgs...)

	cmd := exec.Command(cfg.ServerBin, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}

	logDir := filepath.Join(homeDir(), ".llmctl-logs")
	os.MkdirAll(logDir, 0755)
	logFile, _ := os.Create(filepath.Join(logDir, spec.InstanceName+".log"))
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		return Instance{}, false, fmt.Errorf("error starting server: %w; is %q installed and in PATH?", err, cfg.ServerBin)
	}
	exitCh := make(chan error, 1)
	go func() {
		exitCh <- cmd.Wait()
	}()

	isDefault := len(reg.Instances) == 0

	inst := Instance{
		Name:            spec.InstanceName,
		Model:           spec.ModelPath,
		Mmproj:          spec.MmprojPath,
		PID:             cmd.Process.Pid,
		Port:            backendPort,
		IsDefault:       isDefault,
		CtxSize:         cfg.CtxSize,
		GpuLayers:       cfg.GpuLayers,
		ServerBin:       cfg.ServerBin,
		ExtraArgs:       cfg.ExtraArgs,
		AutoUnload:      autoUnload,
		EstimatedVramMB: estimatedVramMB,
		LastUsedAt:      time.Now().UnixNano(),
	}
	if spec.AliasUsed != "" {
		inst.Aliases = []string{spec.AliasUsed}
	}
	reg.Instances = append(reg.Instances, inst)
	saveRegistry(reg)

	timeout := opts.WaitTimeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	if waitForHealth(backendPort, timeout) {
		if opts.Verbose {
			fmt.Printf("✓ '%s' ready (PID %d, backend :%d)\n", spec.InstanceName, inst.PID, backendPort)
		}
	} else {
		if exitErr, exited := backendExited(exitCh); exited {
			reg := loadRegistry()
			reg.Remove(spec.InstanceName)
			if isDefault && len(reg.Instances) > 0 {
				reg.Instances[0].IsDefault = true
			}
			saveRegistry(reg)
			if opts.Verbose {
				fmt.Fprintf(os.Stderr, "Error: '%s' exited before becoming healthy. Check: llmctl logs %s\n",
					spec.InstanceName, spec.InstanceName)
				if exitErr != nil {
					fmt.Fprintf(os.Stderr, "Backend exit: %v\n", exitErr)
				}
				printLogTail(filepath.Join(logDir, spec.InstanceName+".log"), 20)
			}
			return Instance{}, false, fmt.Errorf("%q exited before becoming healthy: %v", spec.InstanceName, exitErr)
		}
		if opts.Verbose {
			fmt.Printf("⚠ '%s' started (PID %d) but not yet healthy. Check: llmctl logs %s\n",
				spec.InstanceName, inst.PID, spec.InstanceName)
		}
		return inst, false, nil
	}

	if opts.Verbose && isDefault {
		fmt.Printf("  ★ Set as default model\n")
	}
	if opts.Verbose {
		fmt.Printf("\n  Proxy endpoint: http://%s:%d/v1\n", displayHost(cfg.Host), cfg.Port)
		fmt.Printf("  Model name:     %s\n", spec.InstanceName)
		fmt.Printf("  Direct backend: http://127.0.0.1:%d\n", backendPort)
	}
	return inst, true, nil
}

func cmdLoad(cfg Config, modelName string, instanceName string, hfRepo string, mmprojArg string) {
	_, _, err := loadInstance(cfg, loadOptions{
		ModelName:    modelName,
		InstanceName: instanceName,
		HFRepo:       hfRepo,
		MmprojArg:    mmprojArg,
		WaitTimeout:  60 * time.Second,
		Verbose:      true,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func findInstanceByNameFuzzy(reg Registry, name string) *Instance {
	inst := reg.FindByName(name)
	if inst != nil {
		return inst
	}
	lower := strings.ToLower(name)
	for i := range reg.Instances {
		if strings.Contains(strings.ToLower(reg.Instances[i].Name), lower) {
			return &reg.Instances[i]
		}
	}
	return nil
}

func unloadInstance(name string, verbose bool) error {
	reg := loadRegistry()
	reg.CleanDead()

	inst := findInstanceByNameFuzzy(reg, name)
	if inst == nil {
		return fmt.Errorf("no instance %q. Use `llmctl ps` to see loaded models", name)
	}

	if verbose {
		fmt.Printf("Stopping '%s' (PID %d)...\n", inst.Name, inst.PID)
	}
	stopProcess(inst.PID)
	wasDefault := inst.IsDefault
	reg.Remove(inst.Name)

	if wasDefault && len(reg.Instances) > 0 {
		reg.Instances[0].IsDefault = true
		if verbose {
			fmt.Printf("  New default: %s\n", reg.Instances[0].Name)
		}
	}
	saveRegistry(reg)
	if verbose {
		fmt.Println("✓ Stopped.")
	}
	return nil
}

func cmdUnload(_ Config, name string) {
	if err := unloadInstance(name, true); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func cmdStopAll() {
	reg := loadRegistry()

	if reg.ProxyPID > 0 && isRunning(reg.ProxyPID) {
		fmt.Printf("Stopping proxy (PID %d)...\n", reg.ProxyPID)
		stopProcess(reg.ProxyPID)
	}
	for _, inst := range reg.Instances {
		if isRunning(inst.PID) {
			fmt.Printf("Stopping '%s' (PID %d)...\n", inst.Name, inst.PID)
			stopProcess(inst.PID)
		}
	}
	reg.Instances = nil
	reg.ProxyPID = 0
	saveRegistry(reg)
	fmt.Println("✓ All stopped.")
}

func cmdDefault(name string) {
	reg := loadRegistry()
	reg.CleanDead()

	found := false
	for i := range reg.Instances {
		if strings.Contains(strings.ToLower(reg.Instances[i].Name), strings.ToLower(name)) {
			reg.Instances[i].IsDefault = true
			name = reg.Instances[i].Name
			found = true
		} else {
			reg.Instances[i].IsDefault = false
		}
	}
	if !found {
		fmt.Fprintf(os.Stderr, "Error: no instance '%s'\n", name)
		os.Exit(1)
	}
	saveRegistry(reg)
	fmt.Printf("✓ Default model: '%s'\n", name)
}

func cmdPS() {
	reg := loadRegistry()
	reg.CleanDead()
	saveRegistry(reg)

	if len(reg.Instances) == 0 {
		fmt.Println("No models loaded. Use `llmctl load <model>` to start one.")
		return
	}

	fmt.Println("Loaded models:")
	fmt.Println()
	fmt.Printf("  %-3s %-20s %-26s %-8s %-8s %-8s %s\n", "", "NAME", "MODEL", "PID", "PORT", "CTX", "STATUS")
	fmt.Printf("  %-3s %-20s %-26s %-8s %-8s %-8s %s\n", "", "────", "─────", "───", "────", "───", "──────")

	for _, inst := range reg.Instances {
		marker := "  "
		if inst.IsDefault {
			marker = "★ "
		}
		status := "stopped"
		if isRunning(inst.PID) {
			resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", inst.Port))
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					status = "healthy"
				} else {
					status = "loading"
				}
			} else {
				status = "running"
			}
		}
		model := shortName(inst.Model)
		if len(model) > 26 {
			model = model[:23] + "..."
		}
		ctxStr := "-"
		if inst.CtxSize > 0 {
			ctxStr = strconv.Itoa(inst.CtxSize)
		}
		fmt.Printf("  %s%-20s %-26s %-8d %-8d %-8s %s\n",
			marker, inst.Name, model, inst.PID, inst.Port, ctxStr, status)
		if inst.AutoUnload || inst.EstimatedVramMB > 0 {
			fmt.Printf("    autoswitch: auto_unload=%t estimated_vram_mb=%d\n", inst.AutoUnload, inst.EstimatedVramMB)
		}
		if len(inst.Aliases) > 0 {
			fmt.Printf("    aliases: %s\n", strings.Join(inst.Aliases, ", "))
		}
	}

	fmt.Println()
	if reg.ProxyPID > 0 && isRunning(reg.ProxyPID) {
		fmt.Printf("  Proxy: running (PID %d)\n", reg.ProxyPID)
	} else {
		fmt.Println("  Proxy: not running — start with `llmctl proxy`")
	}
}

func cmdInfo(name string) {
	reg := loadRegistry()
	reg.CleanDead()
	saveRegistry(reg)

	inst := reg.FindByName(name)
	if inst == nil {
		lower := strings.ToLower(name)
		for i := range reg.Instances {
			if strings.Contains(strings.ToLower(reg.Instances[i].Name), lower) {
				inst = &reg.Instances[i]
				break
			}
		}
	}
	if inst == nil {
		fmt.Fprintf(os.Stderr, "Error: no instance '%s'. Use `llmctl ps` to see loaded models.\n", name)
		os.Exit(1)
	}

	status := "stopped"
	if isRunning(inst.PID) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", inst.Port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				status = "healthy"
			} else {
				status = "loading"
			}
		} else {
			status = "running"
		}
	}

	fmt.Printf("Instance:   %s\n", inst.Name)
	if len(inst.Aliases) > 0 {
		fmt.Printf("Aliases:    %s\n", strings.Join(inst.Aliases, ", "))
	}
	if inst.IsDefault {
		fmt.Println("Default:    ★ yes")
	}
	fmt.Printf("Status:     %s\n", status)
	fmt.Printf("PID:        %d\n", inst.PID)
	fmt.Printf("Model:      %s\n", inst.Model)
	if inst.Mmproj != "" {
		fmt.Printf("Mmproj:     %s\n", inst.Mmproj)
	}
	fmt.Printf("Backend:    http://127.0.0.1:%d\n", inst.Port)
	fmt.Printf("Server:     %s\n", inst.ServerBin)
	fmt.Printf("Ctx size:   %d\n", inst.CtxSize)
	fmt.Printf("GPU layers: %d\n", inst.GpuLayers)
	if inst.AutoUnload || inst.EstimatedVramMB > 0 || inst.LastUsedAt > 0 {
		fmt.Printf("Auto unload: %t\n", inst.AutoUnload)
		if inst.EstimatedVramMB > 0 {
			fmt.Printf("Est. VRAM:  %d MB\n", inst.EstimatedVramMB)
		}
		if inst.LastUsedAt > 0 {
			fmt.Printf("Last used:  %s\n", formatRegistryTime(inst.LastUsedAt))
		}
	}
	if len(inst.ExtraArgs) > 0 {
		fmt.Println("Extra args:")
		for i := 0; i < len(inst.ExtraArgs); i++ {
			if strings.HasPrefix(inst.ExtraArgs[i], "--") {
				if i+1 < len(inst.ExtraArgs) && !strings.HasPrefix(inst.ExtraArgs[i+1], "--") {
					fmt.Printf("  %s %s\n", inst.ExtraArgs[i], inst.ExtraArgs[i+1])
					i++
				} else {
					fmt.Printf("  %s\n", inst.ExtraArgs[i])
				}
			} else {
				fmt.Printf("  %s\n", inst.ExtraArgs[i])
			}
		}
	}
}

func cmdStatus(cfg Config) {
	reg := loadRegistry()
	reg.CleanDead()

	fmt.Printf("Models loaded: %d\n", len(reg.Instances))
	if reg.ProxyPID > 0 && isRunning(reg.ProxyPID) {
		fmt.Printf("Proxy: running (PID %d) on :%d\n", reg.ProxyPID, cfg.Port)
		fmt.Printf("API:   http://%s:%d/v1\n", displayHost(cfg.Host), cfg.Port)
	} else {
		fmt.Println("Proxy: not running")
	}
	if len(reg.Instances) > 0 {
		fmt.Println()
		cmdPS()
	}
}

func isMmprojFile(name string) bool {
	return strings.HasPrefix(strings.ToLower(filepath.Base(name)), "mmproj-")
}

func loadedModelsByPath(reg Registry) map[string]string {
	loaded := map[string]string{}
	for _, inst := range reg.Instances {
		loaded[filepath.Clean(inst.Model)] = inst.Name
	}
	return loaded
}

func aliasesByResolvedPath(cfg Config) map[string][]string {
	revAlias := map[string][]string{}
	for a, t := range cfg.Aliases {
		if p, err := resolveModel(cfg, t); err == nil {
			revAlias[filepath.Clean(p)] = append(revAlias[filepath.Clean(p)], a)
		} else {
			revAlias[t] = append(revAlias[t], a)
		}
	}
	return revAlias
}

func cmdList(cfg Config) {
	models := listModelFiles(cfg.ModelsDir)
	filtered := make([]string, 0, len(models))
	for _, m := range models {
		if !isMmprojFile(m) {
			filtered = append(filtered, m)
		}
	}
	models = filtered
	if len(models) == 0 {
		fmt.Printf("No .gguf models in %s\n", cfg.ModelsDir)
		return
	}

	reg := loadRegistry()
	reg.CleanDead()
	loaded := loadedModelsByPath(reg)
	revAlias := aliasesByResolvedPath(cfg)

	fmt.Printf("Models in %s:\n\n", cfg.ModelsDir)
	for _, m := range models {
		full := filepath.Join(cfg.ModelsDir, m)
		displayName := modelRef(cfg.ModelsDir, full)
		marker := "  "
		extra := ""
		if name, ok := loaded[filepath.Clean(full)]; ok {
			marker = "▶ "
			extra = fmt.Sprintf(" [loaded as: %s]", name)
		}
		aliases := ""
		if a, ok := revAlias[filepath.Clean(full)]; ok {
			aliases = fmt.Sprintf(" (alias: %s)", strings.Join(a, ", "))
		}
		fmt.Printf("  %s%-60s %8s%s%s\n", marker, displayName, fileSizeStr(full), aliases, extra)
	}
	fmt.Println()
}

func cmdLogs(name string) {
	logDir := filepath.Join(homeDir(), ".llmctl-logs")
	logPath := filepath.Join(logDir, name+".log")

	if _, err := os.Stat(logPath); err != nil {
		entries, _ := os.ReadDir(logDir)
		lower := strings.ToLower(name)
		for _, e := range entries {
			if strings.Contains(strings.ToLower(e.Name()), lower) {
				logPath = filepath.Join(logDir, e.Name())
				break
			}
		}
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "No logs for '%s'\n", name)
		os.Exit(1)
	}
	lines := strings.Split(string(data), "\n")
	start := 0
	if len(lines) > 50 {
		start = len(lines) - 50
	}
	for _, line := range lines[start:] {
		fmt.Println(line)
	}
}

func cmdAlias(cfg Config, alias, model string) {
	modelPath, err := resolveModel(cfg, model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	model = modelRef(cfg.ModelsDir, modelPath)
	cfg.Aliases[alias] = model
	saveConfig(cfg)
	fmt.Printf("✓ Alias '%s' → %s\n", alias, model)
}

func parsePullTarget(target string) (user string, repoName string, repoID string, specificFile string, err error) {
	target = strings.TrimPrefix(target, "https://huggingface.co/")
	target = strings.TrimSuffix(target, "/")
	parts := strings.Split(target, "/")
	if len(parts) < 2 {
		return "", "", "", "", fmt.Errorf("invalid repo")
	}

	user, repoName = parts[0], parts[1]
	if user == "" || repoName == "" {
		return "", "", "", "", fmt.Errorf("invalid repo")
	}

	if colon := strings.Index(repoName, ":"); colon >= 0 {
		specificFile = repoName[colon+1:]
		repoName = repoName[:colon]
	}
	if len(parts) >= 3 {
		specificFile = strings.Join(parts[2:], "/")
	}
	if repoName == "" {
		return "", "", "", "", fmt.Errorf("invalid repo")
	}
	repoID = user + "/" + repoName
	return user, repoName, repoID, specificFile, nil
}

func cmdPull(cfg Config, repo string) {
	user, repoName, repoID, specificFile, err := parsePullTarget(repo)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Usage: llmctl pull <user/repo>[:<file>]")
		os.Exit(1)
	}

	if specificFile == "" {
		apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/%s", user, repoName)
		resp, err := http.Get(apiURL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var repoInfo struct {
			Siblings []struct {
				Filename string `json:"rfilename"`
			} `json:"siblings"`
		}
		json.Unmarshal(body, &repoInfo)

		var ggufFiles []string
		for _, s := range repoInfo.Siblings {
			if strings.HasSuffix(strings.ToLower(s.Filename), ".gguf") {
				ggufFiles = append(ggufFiles, s.Filename)
			}
		}

		if len(ggufFiles) == 0 {
			fmt.Println("No .gguf files in this repo.")
			return
		}

		if len(ggufFiles) == 1 {
			specificFile = ggufFiles[0]
		} else {
			fmt.Println("Available GGUF files:")
			for i, f := range ggufFiles {
				fmt.Printf("  [%d] %s\n", i+1, f)
			}
			fmt.Print("\nPick [1]: ")
			reader := bufio.NewReader(os.Stdin)
			input, _ := reader.ReadString('\n')
			input = strings.TrimSpace(input)
			idx := 0
			if input != "" {
				n, err := strconv.Atoi(input)
				if err != nil || n < 1 || n > len(ggufFiles) {
					fmt.Fprintln(os.Stderr, "Invalid selection.")
					os.Exit(1)
				}
				idx = n - 1
			}
			specificFile = ggufFiles[idx]
		}
	}

	cacheKey := "models--" + user + "--" + repoName + "/" + specificFile
	if hfPath := findInHFCache(cfg.ModelsDir, cacheKey); hfPath != "" {
		fmt.Printf("Already in cache: %s\n", hfPath)
		fmt.Printf("  Use: llmctl load %s:%s\n", repoID, specificFile)
		return
	}

	dlRepo := repoID + ":" + specificFile

	fmt.Printf("Downloading %s to HF cache at %s...\n", dlRepo, cfg.ModelsDir)
	fmt.Println("  (This uses huggingface_hub to populate the proper cache format)")
	fmt.Println("  Press Ctrl+C to cancel")

	var cmd *exec.Cmd
	env := os.Environ()
	env = append(env, "HF_HOME="+cfg.ModelsDir)

	if p, err := exec.LookPath("hf"); err == nil {
		cmd = exec.Command(p, "download", repoID, specificFile)
	} else if p, err := exec.LookPath("huggingface-cli"); err == nil {
		cmd = exec.Command(p, "download", repoID, specificFile)
	} else if p, err := exec.LookPath("python3"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repoID, specificFile)
	} else if p, err := exec.LookPath("python"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repoID, specificFile)
	} else {
		fmt.Fprintf(os.Stderr, "Error: huggingface_hub not installed.\n")
		fmt.Fprintf(os.Stderr, "  Run: pip install huggingface_hub\n")
		os.Exit(1)
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	cmd.Env = env

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	wasCancelled := false

	go func() {
		<-sigCh
		wasCancelled = true
		fmt.Println("\nCancelled.")
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	}()

	if err := cmd.Run(); err != nil {
		if wasCancelled {
			return
		}
		fmt.Fprintf(os.Stderr, "Download failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✓ Downloaded %s\n", dlRepo)
	fmt.Println("\nTo load, use the repo path:")
	fmt.Printf("  llmctl load %s:%s\n", repoID, specificFile)
}

func cmdRM(cfg Config, modelName string) {
	modelPath, err := resolveModel(cfg, modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	reg := loadRegistry()
	for _, inst := range reg.Instances {
		if inst.Model == modelPath && isRunning(inst.PID) {
			fmt.Fprintf(os.Stderr, "Error: model loaded as '%s'. Unload first.\n", inst.Name)
			os.Exit(1)
		}
	}
	fmt.Printf("Delete %s (%s)? [y/N]: ", shortName(modelPath), fileSizeStr(modelPath))
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	if strings.TrimSpace(strings.ToLower(input)) != "y" {
		return
	}
	os.Remove(modelPath)
	fmt.Printf("✓ Deleted %s\n", shortName(modelPath))
}

func cmdConfig(cfg Config) {
	fmt.Println("Configuration:")
	fmt.Printf("  Config:      %s\n", configPath())
	fmt.Printf("  Models dir:  %s\n", cfg.ModelsDir)
	fmt.Printf("  Server bin:  %s\n", cfg.ServerBin)
	fmt.Printf("  Proxy:       %s:%d\n", cfg.Host, cfg.Port)
	fmt.Printf("  GPU layers:  %d (-1 = all)\n", cfg.GpuLayers)
	fmt.Printf("  Ctx size:    %d\n", cfg.CtxSize)
	if len(cfg.ExtraArgs) > 0 {
		fmt.Printf("  Extra args:  %s\n", strings.Join(cfg.ExtraArgs, " "))
	}
	if cfg.Autoswitch.Enabled || cfg.Autoswitch.TotalVramMB > 0 || cfg.Autoswitch.StartupTimeoutSec > 0 {
		fmt.Println("  Autoswitch:")
		fmt.Printf("    enabled:               %t\n", cfg.Autoswitch.Enabled)
		if cfg.Autoswitch.TotalVramMB > 0 {
			fmt.Printf("    total_vram_mb:         %d\n", cfg.Autoswitch.TotalVramMB)
		}
		if cfg.Autoswitch.StartupTimeoutSec > 0 {
			fmt.Printf("    startup_timeout_sec:   %d\n", cfg.Autoswitch.StartupTimeoutSec)
		}
	}
	if len(cfg.Aliases) > 0 {
		fmt.Println("  Aliases:")
		for k, v := range cfg.Aliases {
			fmt.Printf("    %s → %s\n", k, v)
		}
	}
	if len(cfg.Models) > 0 {
		fmt.Println("  Model overrides:")
		for name, mc := range cfg.Models {
			fmt.Printf("    [%s]\n", name)
			if mc.ServerBin != nil {
				fmt.Printf("      server_bin:  %s\n", *mc.ServerBin)
			}
			if mc.GpuLayers != nil {
				fmt.Printf("      gpu_layers:  %d\n", *mc.GpuLayers)
			}
			if mc.CtxSize != nil {
				fmt.Printf("      ctx_size:    %d\n", *mc.CtxSize)
			}
			if mc.ExtraArgs != nil {
				fmt.Printf("      extra_args:  %s\n", strings.Join(mc.ExtraArgs, " "))
			}
			if mc.AutoLoad {
				fmt.Printf("      auto_load:   %t\n", mc.AutoLoad)
			}
			if mc.AutoUnload {
				fmt.Printf("      auto_unload: %t\n", mc.AutoUnload)
			}
			if mc.VramMB > 0 {
				fmt.Printf("      vram_mb:     %d\n", mc.VramMB)
			}
		}
	}
}

func cmdSet(cfg Config, key, value string) {
	switch key {
	case "models_dir":
		cfg.ModelsDir = value
	case "server_bin":
		cfg.ServerBin = value
	case "host":
		cfg.Host = value
	case "port":
		n, _ := strconv.Atoi(value)
		if n == 0 {
			fmt.Fprintln(os.Stderr, "Error: invalid port")
			os.Exit(1)
		}
		cfg.Port = n
	case "gpu_layers":
		n, err := strconv.Atoi(value)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error: must be a number")
			os.Exit(1)
		}
		cfg.GpuLayers = n
	case "ctx_size":
		n, err := strconv.Atoi(value)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error: must be a number")
			os.Exit(1)
		}
		cfg.CtxSize = n
	case "autoswitch.enabled":
		v, err := strconv.ParseBool(value)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error: must be true or false")
			os.Exit(1)
		}
		cfg.Autoswitch.Enabled = v
	case "autoswitch.total_vram_mb":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			fmt.Fprintln(os.Stderr, "Error: must be a non-negative number")
			os.Exit(1)
		}
		cfg.Autoswitch.TotalVramMB = n
	case "autoswitch.startup_timeout_sec":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			fmt.Fprintln(os.Stderr, "Error: must be a non-negative number")
			os.Exit(1)
		}
		cfg.Autoswitch.StartupTimeoutSec = n
	default:
		fmt.Fprintf(os.Stderr, "Unknown key: %s\n", key)
		os.Exit(1)
	}
	saveConfig(cfg)
	fmt.Printf("✓ %s = %s\n", key, value)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func displayHost(h string) string {
	if h == "0.0.0.0" {
		return getLocalIP()
	}
	return h
}

func getLocalIP() string {
	addrs, _ := net.InterfaceAddrs()
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			return ipnet.IP.String()
		}
	}
	return "127.0.0.1"
}

func extractFlag(args []string, flag string) (string, []string) {
	var remaining []string
	value := ""
	skip := false
	for i, a := range args {
		if skip {
			skip = false
			continue
		}
		if a == flag && i+1 < len(args) {
			value = args[i+1]
			skip = true
		} else if strings.HasPrefix(a, flag+"=") {
			value = strings.TrimPrefix(a, flag+"=")
		} else {
			remaining = append(remaining, a)
		}
	}
	return value, remaining
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

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

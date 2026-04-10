package main

import (
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
	ServerBin *string  `json:"server_bin,omitempty"`
	GpuLayers *int     `json:"gpu_layers,omitempty"`
	CtxSize   *int     `json:"ctx_size,omitempty"`
	ExtraArgs []string `json:"extra_args,omitempty"`
}

type Config struct {
	ModelsDir string                 `json:"models_dir"`
	ServerBin string                 `json:"server_bin"`
	Host      string                 `json:"host"`
	Port      int                    `json:"port"`
	GpuLayers int                    `json:"gpu_layers"`
	CtxSize   int                    `json:"ctx_size"`
	ExtraArgs []string               `json:"extra_args,omitempty"`
	Aliases   map[string]string      `json:"aliases,omitempty"`
	Models    map[string]ModelConfig `json:"models,omitempty"`
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
	if mc.ExtraArgs != nil {
		cfg.ExtraArgs = mc.ExtraArgs
	}
	return cfg
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
	Name      string `json:"name"`
	Model     string `json:"model"` // full path to .gguf
	PID       int    `json:"pid"`
	Port      int    `json:"port"`       // internal backend port
	IsDefault bool   `json:"is_default"` // default model for unmatched requests
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

	// Check if name looks like a HuggingFace repo (user/repo) and try to
	// find a .gguf file in the HF cache before falling back to other lookups.
	origName := name
	if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
		repoParts := strings.SplitN(name, "/", 2)
		if len(repoParts) == 2 {
			cachePrefix := "models--" + repoParts[0] + "--" + repoParts[1]
			// Search multiple cache layouts:
			//   1. <modelsDir>/<cachePrefix>                  (top-level models-- dir)
			//   2. <modelsDir>/hub/<cachePrefix>              (hf download format)
			searchBases := []string{
				cfg.ModelsDir,
				filepath.Join(cfg.ModelsDir, "hub"),
			}
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
						if strings.HasSuffix(strings.ToLower(f.Name()), ".gguf") {
							return filepath.Join(snapDir, f.Name()), nil
						}
					}
				}
			}
		}
		name = name + ".gguf"
	}
	full := filepath.Join(cfg.ModelsDir, name)
	if _, err := os.Stat(full); err == nil {
		return full, nil
	}
	lower := strings.ToLower(strings.TrimSuffix(name, ".gguf"))
	for _, m := range listModelFiles(cfg.ModelsDir) {
		if strings.Contains(strings.ToLower(m), lower) {
			return filepath.Join(cfg.ModelsDir, m), nil
		}
	}
	if strings.HasPrefix(name, "models--") {
		if path := findInHFCache(cfg.ModelsDir, name); path != "" {
			return path, nil
		}
	}
	return "", fmt.Errorf("model not found: %s (looked in %s)", origName, cfg.ModelsDir)
}

func findInHFCache(modelsDir, name string) string {
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

// Derive a clean instance name from the model filename
func deriveInstanceName(modelPath string) string {
	base := filepath.Base(modelPath)
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
		if r == ' ' || r == '/' || r == '\\' {
			return '-'
		}
		return r
	}, base)
	return strings.ToLower(base)
}

// ═══════════════════════════════════════════════════════════════════════════
// Reverse Proxy — routes by "model" field in OpenAI requests
// ═══════════════════════════════════════════════════════════════════════════

func startProxy(cfg Config) {
	reg := loadRegistry()
	reg.CleanDead()

	if len(reg.Instances) == 0 {
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
			handleListModels(w)
			return
		}

		// GET /health
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"ok"}`))
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

		// Fuzzy match
		if !ok && targetName != "" {
			lower := strings.ToLower(targetName)
			for name, u := range backends {
				if strings.Contains(strings.ToLower(name), lower) {
					target = u
					ok = true
					break
				}
			}
		}

		// Fallback to default
		if !ok {
			reg := loadRegistry()
			if def := reg.Default(); def != nil {
				target, _ = url.Parse(fmt.Sprintf("http://127.0.0.1:%d", def.Port))
				ok = true
			}
		}
		mu.RUnlock()

		if !ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(404)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]string{
					"message": fmt.Sprintf("model '%s' not loaded. Run: llmctl load <model>", targetName),
					"type":    "invalid_request_error",
				},
			})
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
	fmt.Println("  Routes:")

	mu.RLock()
	for name, u := range backends {
		reg := loadRegistry()
		def := ""
		if inst := reg.FindByName(name); inst != nil && inst.IsDefault {
			def = " (default)"
		}
		fmt.Printf("    %-28s → %s%s\n", name, u.String(), def)
	}
	mu.RUnlock()

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

func handleListModels(w http.ResponseWriter) {
	reg := loadRegistry()
	reg.CleanDead()

	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var models []modelObj
	for _, inst := range reg.Instances {
		models = append(models, modelObj{
			ID:      inst.Name,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "llmctl",
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI Commands
// ═══════════════════════════════════════════════════════════════════════════

func cmdLoad(cfg Config, modelName string, instanceName string, hfRepo string) {
	var modelPath string
	var err error

	if hfRepo != "" {
		modelPath = hfRepo
	} else {
		modelPath, err = resolveModel(cfg, modelName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}

	// Apply per-model config overrides (exact match on modelName).
	cfg = configForModel(cfg, modelName)

	if instanceName == "" {
		instanceName = deriveInstanceName(modelPath)
	}

	reg := loadRegistry()
	reg.CleanDead()

	if existing := reg.FindByName(instanceName); existing != nil {
		if isRunning(existing.PID) {
			fmt.Printf("Replacing instance '%s'...\n", instanceName)
			stopProcess(existing.PID)
		}
		reg.Remove(instanceName)
	}

	backendPort := reg.NextPort()
	displayName := modelPath
	if hfRepo != "" {
		displayName = hfRepo
	}
	fmt.Printf("Loading %s as '%s' (backend :%d)...\n", shortName(displayName), instanceName, backendPort)

	args := []string{}
	if hfRepo != "" {
		args = append(args, "-hf", hfRepo)
	} else {
		args = append(args, "--model", modelPath)
	}
	args = append(args,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(backendPort),
		"--ctx-size", strconv.Itoa(cfg.CtxSize),
	)
	if cfg.GpuLayers != 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(cfg.GpuLayers))
	}
	args = append(args, cfg.ExtraArgs...)

	cmd := exec.Command(cfg.ServerBin, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}

	logDir := filepath.Join(homeDir(), ".llmctl-logs")
	os.MkdirAll(logDir, 0755)
	logFile, _ := os.Create(filepath.Join(logDir, instanceName+".log"))
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		fmt.Fprintf(os.Stderr, "Is '%s' installed and in PATH?\n", cfg.ServerBin)
		os.Exit(1)
	}

	isDefault := len(reg.Instances) == 0

	inst := Instance{
		Name:      instanceName,
		Model:     modelPath,
		PID:       cmd.Process.Pid,
		Port:      backendPort,
		IsDefault: isDefault,
	}
	reg.Instances = append(reg.Instances, inst)
	saveRegistry(reg)

	if waitForHealth(backendPort, 60*time.Second) {
		fmt.Printf("✓ '%s' ready (PID %d, backend :%d)\n", instanceName, inst.PID, backendPort)
	} else {
		fmt.Printf("⚠ '%s' started (PID %d) but not yet healthy. Check: llmctl logs %s\n",
			instanceName, inst.PID, instanceName)
	}

	if isDefault {
		fmt.Printf("  ★ Set as default model\n")
	}
	fmt.Printf("\n  Proxy endpoint: http://%s:%d/v1\n", displayHost(cfg.Host), cfg.Port)
	fmt.Printf("  Model name:     %s\n", instanceName)
	fmt.Printf("  Direct backend: http://127.0.0.1:%d\n", backendPort)
}

func cmdUnload(_ Config, name string) {
	reg := loadRegistry()
	reg.CleanDead()

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

	fmt.Printf("Stopping '%s' (PID %d)...\n", inst.Name, inst.PID)
	stopProcess(inst.PID)
	wasDefault := inst.IsDefault
	reg.Remove(inst.Name)

	if wasDefault && len(reg.Instances) > 0 {
		reg.Instances[0].IsDefault = true
		fmt.Printf("  New default: %s\n", reg.Instances[0].Name)
	}
	saveRegistry(reg)
	fmt.Println("✓ Stopped.")
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

	fmt.Println("Loaded models:\n")
	fmt.Printf("  %-3s %-28s %-30s %-8s %-8s %s\n", "", "NAME", "MODEL", "PID", "PORT", "STATUS")
	fmt.Printf("  %-3s %-28s %-30s %-8s %-8s %s\n", "", "────", "─────", "───", "────", "──────")

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
		if len(model) > 30 {
			model = model[:27] + "..."
		}
		fmt.Printf("  %s%-28s %-30s %-8d %-8d %s\n",
			marker, inst.Name, model, inst.PID, inst.Port, status)
	}

	fmt.Println()
	if reg.ProxyPID > 0 && isRunning(reg.ProxyPID) {
		fmt.Printf("  Proxy: running (PID %d)\n", reg.ProxyPID)
	} else {
		fmt.Println("  Proxy: not running — start with `llmctl proxy`")
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

func cmdList(cfg Config) {
	models := listModelFiles(cfg.ModelsDir)
	if len(models) == 0 {
		fmt.Printf("No .gguf models in %s\n", cfg.ModelsDir)
		return
	}

	reg := loadRegistry()
	reg.CleanDead()
	loaded := map[string]string{}
	for _, inst := range reg.Instances {
		loaded[shortName(inst.Model)] = inst.Name
	}

	revAlias := map[string][]string{}
	for a, t := range cfg.Aliases {
		revAlias[t] = append(revAlias[t], a)
	}

	fmt.Printf("Models in %s:\n\n", cfg.ModelsDir)
	for _, m := range models {
		full := filepath.Join(cfg.ModelsDir, m)
		displayName := filepath.Base(m)
		marker := "  "
		extra := ""
		if name, ok := loaded[displayName]; ok {
			marker = "▶ "
			extra = fmt.Sprintf(" [loaded as: %s]", name)
		}
		aliases := ""
		if a, ok := revAlias[displayName]; ok {
			aliases = fmt.Sprintf(" (alias: %s)", strings.Join(a, ", "))
		}
		fmt.Printf("  %s%-42s %8s%s%s\n", marker, displayName, fileSizeStr(full), aliases, extra)
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
	if !strings.HasSuffix(strings.ToLower(model), ".gguf") {
		model = model + ".gguf"
	}
	full := filepath.Join(cfg.ModelsDir, model)
	if _, err := os.Stat(full); err != nil {
		lower := strings.ToLower(strings.TrimSuffix(model, ".gguf"))
		found := false
		for _, m := range listModelFiles(cfg.ModelsDir) {
			if strings.Contains(strings.ToLower(m), lower) {
				model = m
				found = true
				break
			}
		}
		if !found {
			fmt.Fprintf(os.Stderr, "Error: model '%s' not found\n", model)
			os.Exit(1)
		}
	}
	cfg.Aliases[alias] = model
	saveConfig(cfg)
	fmt.Printf("✓ Alias '%s' → %s\n", alias, model)
}

func cmdPull(cfg Config, repo string) {
	repo = strings.TrimPrefix(repo, "https://huggingface.co/")
	repo = strings.TrimSuffix(repo, "/")

	parts := strings.Split(repo, "/")
	if len(parts) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: llmctl pull <user/repo>[:<file>]")
		os.Exit(1)
	}

	user, repoName := parts[0], parts[1]
	specificFile := ""
	if len(parts) >= 3 {
		specificFile = strings.Join(parts[2:], "/")
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
		fmt.Printf("  Use: llmctl load %s\n", repo)
		return
	}

	dlRepo := repo + ":" + specificFile

	fmt.Printf("Downloading %s to HF cache at %s...\n", dlRepo, cfg.ModelsDir)
	fmt.Println("  (This uses huggingface_hub to populate the proper cache format)")
	fmt.Println("  Press Ctrl+C to cancel")

	var cmd *exec.Cmd
	env := os.Environ()
	env = append(env, "HF_HOME="+cfg.ModelsDir)

	if p, err := exec.LookPath("hf"); err == nil {
		cmd = exec.Command(p, "download", repo, specificFile)
	} else if p, err := exec.LookPath("huggingface-cli"); err == nil {
		cmd = exec.Command(p, "download", repo, specificFile)
	} else if p, err := exec.LookPath("python3"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repo, specificFile)
	} else if p, err := exec.LookPath("python"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repo, specificFile)
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
	fmt.Printf("  llmctl load %s\n", repo)
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
  load <model> [-hf <hf_repo>] [--name NAME]  Load a model (starts a backend)
  unload <name>                             Stop a model instance
  stop                        Stop everything
  default <name>              Set default model for unmatched requests
  ps                          List loaded instances
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
			fmt.Fprintln(os.Stderr, "Usage: llmctl load <model> [-hf <huggingface_repo>] [--name NAME]")
			os.Exit(1)
		}
		name, args := extractFlag(os.Args[3:], "--name")
		hf, _ := extractFlag(args, "-hf")
		cmdLoad(cfg, os.Args[2], name, hf)
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

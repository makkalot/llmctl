package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

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

package llmctl

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"syscall"
	"time"
)

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

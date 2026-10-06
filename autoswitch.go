package main

import (
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

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

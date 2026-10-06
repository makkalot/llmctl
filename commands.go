package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

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

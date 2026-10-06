package llmctl

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"time"
)

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

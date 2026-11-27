package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
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
	"path"
	"path/filepath"
	"regexp"
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

	defaultDockerImage = "ghcr.io/ggml-org/llama.cpp:server-cuda"
	containerPrefix    = "llmctl-"
	containerLabel     = "llmctl.instance"
	defaultPidsLimit   = 512

	// Backend network modes (docker.network).
	netNone     = "none"     // no network; llama-server listens on a unix socket
	netInternal = "internal" // docker --internal network, no egress
	netBridge   = "bridge"   // default docker bridge, full egress (needed for in-container -hf)

	internalNetwork = "llmctl-internal"
)

// ═══════════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════════

type ModelConfig struct {
	DockerImage      *string  `json:"docker_image,omitempty"`
	DockerEntrypoint *string  `json:"docker_entrypoint,omitempty"`
	DockerArgs       []string `json:"docker_args,omitempty"`
	DockerNetwork    *string  `json:"docker_network,omitempty"`
	DockerBuild      *string  `json:"docker_build,omitempty"` // name of an entry in `builds` (optionally name@commit)
	GpuLayers        *int     `json:"gpu_layers,omitempty"`
	CtxSize          *int     `json:"ctx_size,omitempty"`
	Mmproj           string   `json:"mmproj,omitempty"`
	ExtraArgs        []string `json:"extra_args,omitempty"`
	AutoLoad         bool     `json:"auto_load,omitempty"`
	AutoUnload       bool     `json:"auto_unload,omitempty"`
	VramMB           int      `json:"vram_mb,omitempty"`
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
	Docker     DockerConfig           `json:"docker"`
	Host       string                 `json:"host"`
	Port       int                    `json:"port"`
	GpuLayers  int                    `json:"gpu_layers"`
	CtxSize    int                    `json:"ctx_size"`
	Mmproj     string                 `json:"mmproj,omitempty"`
	ExtraArgs  []string               `json:"extra_args,omitempty"`
	Aliases    map[string]string      `json:"aliases,omitempty"`
	Models     map[string]ModelConfig `json:"models,omitempty"`
	Autoswitch AutoswitchConfig       `json:"autoswitch,omitempty"`
	Builds     map[string]BuildConfig `json:"builds,omitempty"`
}

// BuildConfig describes a llama.cpp (or fork) build made by `llmctl build`.
// Exactly one of Repo or Path is set.
type BuildConfig struct {
	Repo         string   `json:"repo,omitempty"`          // git URL
	Ref          string   `json:"ref,omitempty"`           // branch, tag, full SHA or "#<PR number>"; default HEAD
	Path         string   `json:"path,omitempty"`          // local git checkout (uncommitted changes included)
	Backend      string   `json:"backend,omitempty"`       // "cuda" (default) or "jetson"
	CudaArch     string   `json:"cuda_arch,omitempty"`     // CMAKE_CUDA_ARCHITECTURES; default: host GPU (cuda) / 87 (jetson)
	BaseImage    string   `json:"base_image,omitempty"`    // build-stage image
	RuntimeImage string   `json:"runtime_image,omitempty"` // runtime-stage image
	CmakeArgs    []string `json:"cmake_args,omitempty"`
	Target       string   `json:"target,omitempty"`     // cmake target, default llama-server
	Dockerfile   string   `json:"dockerfile,omitempty"` // custom Dockerfile (relative to the source tree)
	Keep         int      `json:"keep,omitempty"`       // images to keep per build, default 3
}

// DockerConfig controls how llama-server backends are run as containers.
type DockerConfig struct {
	Bin        string   `json:"bin"`                  // docker CLI (or a compatible one, e.g. podman)
	Image      string   `json:"image"`                // llama.cpp server image
	Entrypoint string   `json:"entrypoint,omitempty"` // override image entrypoint (e.g. "llama-server")
	GPU        string   `json:"gpu"`                  // "nvidia" (--gpus all), "jetson" (--runtime nvidia), "none"
	HFCacheDir string   `json:"hf_cache_dir"`         // host dir mounted for -hf downloads (bridge network only)
	Network    string   `json:"network"`              // "none" (unix socket), "internal" or "bridge"
	ReadOnly   *bool    `json:"read_only,omitempty"`  // read-only root fs (default true)
	PidsLimit  int      `json:"pids_limit"`           // --pids-limit; <= 0 disables
	Args       []string `json:"args,omitempty"`       // extra `docker run` flags
	Build      string   `json:"build,omitempty"`      // default build (entry in `builds`) instead of Image
}

func (d DockerConfig) readOnly() bool { return d.ReadOnly == nil || *d.ReadOnly }

func validNetworkMode(m string) bool {
	return m == netNone || m == netInternal || m == netBridge
}

func defaultHFCacheDir() string {
	if d := os.Getenv("LLAMA_CACHE"); d != "" {
		return d
	}
	return filepath.Join(homeDir(), ".cache", "llama.cpp")
}

func defaultConfig() Config {
	return Config{
		ModelsDir: modelsDir(),
		Docker: DockerConfig{
			Bin:        "docker",
			Image:      defaultDockerImage,
			GPU:        "nvidia",
			HFCacheDir: defaultHFCacheDir(),
			Network:    netNone,
			PidsLimit:  defaultPidsLimit,
		},
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
	def := defaultConfig().Docker
	if cfg.Docker.Bin == "" {
		cfg.Docker.Bin = def.Bin
	}
	if cfg.Docker.Image == "" {
		cfg.Docker.Image = def.Image
	}
	if cfg.Docker.GPU == "" {
		cfg.Docker.GPU = def.GPU
	}
	if cfg.Docker.HFCacheDir == "" {
		cfg.Docker.HFCacheDir = def.HFCacheDir
	}
	if cfg.Docker.Network == "" {
		cfg.Docker.Network = def.Network
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
	if mc.DockerImage != nil {
		cfg.Docker.Image = *mc.DockerImage
	}
	if mc.DockerEntrypoint != nil {
		cfg.Docker.Entrypoint = *mc.DockerEntrypoint
	}
	if mc.DockerArgs != nil {
		cfg.Docker.Args = append(append([]string{}, cfg.Docker.Args...), mc.DockerArgs...)
	}
	if mc.DockerNetwork != nil {
		cfg.Docker.Network = *mc.DockerNetwork
	}
	if mc.DockerBuild != nil {
		cfg.Docker.Build = *mc.DockerBuild
	}
	if mc.DockerImage != nil && mc.DockerBuild == nil {
		// An explicit per-model image wins over a global default build.
		cfg.Docker.Build = ""
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

// ═══════════════════════════════════════════════════════════════════════════
// Instance Registry — tracks multiple running backends
// ═══════════════════════════════════════════════════════════════════════════

type Instance struct {
	Name            string   `json:"name"`
	Model           string   `json:"model"` // full path to .gguf
	Mmproj          string   `json:"mmproj,omitempty"`
	Container       string   `json:"container"` // docker container name
	Image           string   `json:"image"`
	Network         string   `json:"network,omitempty"` // none | internal | bridge
	Socket          string   `json:"socket,omitempty"`  // unix socket path (network none)
	Addr            string   `json:"addr,omitempty"`    // host:port (network internal/bridge)
	Port            int      `json:"port"`
	IsDefault       bool     `json:"is_default"` // default model for unmatched requests
	CtxSize         int      `json:"ctx_size"`
	GpuLayers       int      `json:"gpu_layers"`
	ExtraArgs       []string `json:"extra_args,omitempty"`
	Aliases         []string `json:"aliases,omitempty"`
	AutoUnload      bool     `json:"auto_unload,omitempty"`
	EstimatedVramMB int      `json:"estimated_vram_mb,omitempty"`
	LastUsedAt      int64    `json:"last_used_at,omitempty"`
	Build           string   `json:"build,omitempty"`  // llmctl build name, if any
	Commit          string   `json:"commit,omitempty"` // source commit of the build image
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
	live := liveContainers()
	for i := range r.Instances {
		if live[r.Instances[i].Container] {
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

// CleanDead drops instances whose container is no longer running. Exited
// containers are not started with --rm, so their logs are saved to
// ~/.llmctl-logs before the container is removed.
func (r *Registry) CleanDead() {
	live, err := liveContainersErr()
	if err != nil {
		// Docker unreachable: keep the registry rather than forget backends.
		return
	}
	var alive []Instance
	for _, inst := range r.Instances {
		if live[inst.Container] {
			alive = append(alive, inst)
			continue
		}
		if inst.Container != "" {
			removeContainer(inst.Container, instanceLogPath(inst.Name))
			removeSocketDir(inst.Container)
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

// isRunning reports whether a host process is alive (used for the proxy PID).
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

// ═══════════════════════════════════════════════════════════════════════════
// Docker helpers — llama-server backends run as containers
// ═══════════════════════════════════════════════════════════════════════════

// dockerBin is the container CLI used for all docker calls. It is set from
// config in main() so registry helpers don't need a Config.
var dockerBin = "docker"

// dockerExec runs the docker CLI and returns combined stdout. It is a
// variable so tests can stub it.
var dockerExec = func(args ...string) (string, error) {
	cmd := exec.Command(dockerBin, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return string(out), fmt.Errorf("%s %s: %s", dockerBin, strings.Join(args, " "), msg)
	}
	return string(out), nil
}

var (
	liveMu      sync.Mutex
	liveCache   map[string]bool
	liveCacheAt time.Time
	liveErr     error
)

const liveCacheTTL = time.Second

// liveContainers returns the names of running llmctl-managed containers.
// The result is cached briefly because the proxy calls this on hot paths.
func liveContainers() map[string]bool {
	live, _ := liveContainersErr()
	return live
}

// liveContainersErr is liveContainers plus the error from docker, so callers
// that delete state can tell "nothing running" from "docker unreachable".
func liveContainersErr() (map[string]bool, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	if liveCache != nil && time.Since(liveCacheAt) < liveCacheTTL {
		return liveCache, liveErr
	}
	out, err := dockerExec("ps", "--filter", "label="+containerLabel, "--format", "{{.Names}}")
	live := map[string]bool{}
	if err == nil {
		for _, name := range strings.Fields(out) {
			live[name] = true
		}
	}
	liveCache, liveCacheAt, liveErr = live, time.Now(), err
	return live, err
}

func invalidateLiveContainers() {
	liveMu.Lock()
	liveCache = nil
	liveMu.Unlock()
}

func instanceRunning(inst Instance) bool {
	return inst.Container != "" && liveContainers()[inst.Container]
}

// containerName maps an instance name to a valid docker container name.
func containerName(instance string) string {
	name := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '.', r == '-':
			return r
		}
		return '-'
	}, instance)
	return containerPrefix + name
}

func logDir() string { return filepath.Join(homeDir(), ".llmctl-logs") }

func instanceLogPath(name string) string { return filepath.Join(logDir(), name+".log") }

// saveContainerLogs writes the container's stdout+stderr to path.
func saveContainerLogs(container, path string) {
	if path == "" {
		return
	}
	cmd := exec.Command(dockerBin, "logs", container)
	out, err := cmd.CombinedOutput()
	if err != nil || len(out) == 0 {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	_ = os.WriteFile(path, out, 0644)
}

// removeContainer saves the container's logs (if logPath is set) and removes it.
func removeContainer(container, logPath string) {
	saveContainerLogs(container, logPath)
	_, _ = dockerExec("rm", "-f", container)
	invalidateLiveContainers()
}

// stopInstance stops and removes an instance's container, keeping its logs.
func stopInstance(inst Instance) {
	if inst.Container == "" {
		return
	}
	_, _ = dockerExec("stop", "-t", "10", inst.Container)
	removeContainer(inst.Container, instanceLogPath(inst.Name))
	removeSocketDir(inst.Container)
}

// socketDir is the per-instance host directory holding llama-server's unix
// socket. Each instance gets its own directory so a compromised backend
// cannot reach other backends' sockets.
func socketDir(container string) string {
	return filepath.Join(homeDir(), ".llmctl-run", container)
}

func removeSocketDir(container string) {
	_ = os.RemoveAll(socketDir(container))
}

// ensureInternalNetwork creates the egress-less docker network used in
// "internal" mode. Inter-container traffic is disabled so backends cannot
// talk to each other.
func ensureInternalNetwork() error {
	if _, err := dockerExec("network", "inspect", internalNetwork); err == nil {
		return nil
	}
	_, err := dockerExec("network", "create", "--internal",
		"-o", "com.docker.network.bridge.enable_icc=false",
		"--label", containerLabel+"=network",
		internalNetwork)
	return err
}

func containerIP(container, network string) (string, error) {
	out, err := dockerExec("inspect", "-f", fmt.Sprintf("{{(index .NetworkSettings.Networks %q).IPAddress}}", network), container)
	if err != nil {
		return "", err
	}
	ip := strings.TrimSpace(out)
	if ip == "" || ip == "<no value>" {
		return "", fmt.Errorf("container %s has no IP on network %s", container, network)
	}
	return ip, nil
}

// backendTarget is how the host reaches a backend: TCP or a unix socket.
type backendTarget struct {
	URL       *url.URL
	Socket    string
	Transport http.RoundTripper
}

var socketTransports sync.Map // socket path -> *http.Transport

func unixTransport(sock string) http.RoundTripper {
	if t, ok := socketTransports.Load(sock); ok {
		return t.(http.RoundTripper)
	}
	t := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
		MaxIdleConns:    16,
		IdleConnTimeout: 90 * time.Second,
	}
	actual, _ := socketTransports.LoadOrStore(sock, t)
	return actual.(http.RoundTripper)
}

func instanceBackend(inst Instance) backendTarget {
	if inst.Socket != "" {
		u, _ := url.Parse("http://localhost")
		return backendTarget{URL: u, Socket: inst.Socket, Transport: unixTransport(inst.Socket)}
	}
	addr := inst.Addr
	if addr == "" {
		addr = fmt.Sprintf("127.0.0.1:%d", inst.Port)
	}
	u, _ := url.Parse("http://" + addr)
	return backendTarget{URL: u, Transport: http.DefaultTransport}
}

func (b backendTarget) String() string {
	if b.Socket != "" {
		return "unix://" + b.Socket
	}
	return b.URL.String()
}

func (b backendTarget) client(timeout time.Duration) *http.Client {
	return &http.Client{Transport: b.Transport, Timeout: timeout}
}

// backendStatus reports healthy/loading/running from the backend's /health.
func backendStatus(inst Instance) string {
	resp, err := instanceBackend(inst).client(3 * time.Second).Get(instanceBackend(inst).URL.String() + "/health")
	if err != nil {
		return "running"
	}
	resp.Body.Close()
	if resp.StatusCode == 200 {
		return "healthy"
	}
	return "loading"
}

// containerState returns whether the container is running and its exit code.
// exists is false if the container is gone.
func containerState(container string) (running bool, exitCode int, exists bool) {
	out, err := dockerExec("inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", container)
	if err != nil {
		return false, 0, false
	}
	fields := strings.Fields(out)
	if len(fields) != 2 {
		return false, 0, false
	}
	exitCode, _ = strconv.Atoi(fields[1])
	return fields[0] == "true", exitCode, true
}

// ensureImage pulls the image if it is not present locally, so the first
// pull does not count against the backend health timeout.
func ensureImage(image string, verbose bool) error {
	if _, err := dockerExec("image", "inspect", image); err == nil {
		return nil
	}
	if verbose {
		fmt.Printf("Pulling image %s...\n", image)
	}
	cmd := exec.Command(dockerBin, "pull", image)
	if verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("pull %s: %v: %s", image, err, strings.TrimSpace(string(out)))
	}
	return nil
}

type containerMount struct {
	Host     string
	ReadOnly bool
}

// backendMounts returns host directories to bind-mount (at the same path) so
// every host path llama-server is given also exists inside the container.
// HF-cache model files are symlinks into ../../blobs, so symlink targets are
// mounted too.
func backendMounts(modelsDir, hfCacheDir string, files []string, extraArgs []string) []containerMount {
	var mounts []containerMount
	idx := map[string]int{}
	add := func(dir string, ro bool) {
		if dir == "" || dir == "/" || dir == "." {
			return
		}
		dir = filepath.Clean(dir)
		if i, ok := idx[dir]; ok {
			if !ro {
				mounts[i].ReadOnly = false
			}
			return
		}
		idx[dir] = len(mounts)
		mounts = append(mounts, containerMount{Host: dir, ReadOnly: ro})
	}
	covered := func(path string) bool {
		for _, m := range mounts {
			if path == m.Host || strings.HasPrefix(path, m.Host+string(filepath.Separator)) {
				return true
			}
		}
		return false
	}
	addFile := func(path string, ro bool) {
		for _, p := range []string{path, resolveSymlinks(path)} {
			if p == "" {
				continue
			}
			if dir := filepath.Dir(p); !covered(dir) {
				add(dir, ro)
			}
		}
	}

	if modelsDir != "" {
		add(modelsDir, true)
	}
	if hfCacheDir != "" {
		add(hfCacheDir, false)
	}
	for _, f := range files {
		if filepath.IsAbs(f) {
			addFile(f, true)
		}
	}
	for i, a := range extraArgs {
		if !filepath.IsAbs(a) {
			continue
		}
		// Paths are mounted read-only unless the flag is one llama-server
		// writes to, so a compromised backend can't tamper with templates,
		// LoRAs, etc. used by other instances.
		ro := i == 0 || !writableArgFlags[extraArgs[i-1]]
		if info, err := os.Stat(a); err == nil && info.IsDir() {
			if !covered(a) {
				add(a, ro)
			}
			continue
		}
		if _, err := os.Stat(filepath.Dir(a)); err == nil {
			addFile(a, ro)
		}
	}
	return mounts
}

// writableArgFlags are llama-server flags whose path argument is written to.
var writableArgFlags = map[string]bool{
	"--slot-save-path": true,
	"--log-file":       true,
	"-lf":              true,
}

func resolveSymlinks(path string) string {
	real, err := filepath.EvalSymlinks(path)
	if err != nil || real == path {
		return ""
	}
	return real
}

type dockerRunSpec struct {
	Docker     DockerConfig
	Container  string
	Instance   string
	Port       int
	Mounts     []containerMount
	ServerArgs []string // llama-server arguments
}

// dockerRunArgs builds the `docker run` argument list for a backend.
func dockerRunArgs(spec dockerRunSpec) []string {
	args := []string{
		"run", "-d",
		"--name", spec.Container,
		"--label", containerLabel + "=" + spec.Instance,
		"--user", fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
		// Hardening: no capabilities, no privilege escalation.
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
	}
	if spec.Docker.readOnly() {
		args = append(args, "--read-only", "--tmpfs", "/tmp")
	}
	if spec.Docker.PidsLimit > 0 {
		args = append(args, "--pids-limit", strconv.Itoa(spec.Docker.PidsLimit))
	}
	switch spec.Docker.Network {
	case netBridge:
		args = append(args, "-p", fmt.Sprintf("127.0.0.1:%d:%d", spec.Port, spec.Port))
	case netInternal:
		args = append(args, "--network", internalNetwork)
	default: // netNone
		args = append(args, "--network", "none")
	}
	switch strings.ToLower(spec.Docker.GPU) {
	case "nvidia", "cuda":
		args = append(args, "--gpus", "all")
	case "jetson":
		args = append(args, "--runtime", "nvidia")
	}
	for _, m := range spec.Mounts {
		v := m.Host + ":" + m.Host
		if m.ReadOnly {
			v += ":ro"
		}
		args = append(args, "-v", v)
	}
	if spec.Docker.Network == netBridge && spec.Docker.HFCacheDir != "" {
		args = append(args,
			"-e", "LLAMA_CACHE="+spec.Docker.HFCacheDir,
			"-e", "HF_HUB_CACHE="+spec.Docker.HFCacheDir,
		)
	}
	if spec.Docker.Entrypoint != "" {
		args = append(args, "--entrypoint", spec.Docker.Entrypoint)
	}
	args = append(args, spec.Docker.Args...)
	args = append(args, spec.Docker.Image)
	return append(args, spec.ServerArgs...)
}

// waitForHealth polls the backend's /health until it returns 200, the
// container exits, or the timeout elapses.
func waitForHealth(target backendTarget, container string, timeout time.Duration) (healthy bool, exited bool, exitCode int) {
	deadline := time.Now().Add(timeout)
	addr := target.URL.String() + "/health"
	client := target.client(2 * time.Second)
	for i := 0; time.Now().Before(deadline); i++ {
		resp, err := client.Get(addr)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true, false, 0
			}
		}
		if i%4 == 0 {
			if running, code, _ := containerState(container); !running {
				return false, true, code
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false, false, 0
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

// ═══════════════════════════════════════════════════════════════════════════
// Custom llama.cpp builds — `llmctl build` turns a git ref / PR / local
// checkout of llama.cpp (or a fork) into a llama-server image.
// ═══════════════════════════════════════════════════════════════════════════

//go:embed docker/llama-server.Dockerfile
var llamaServerDockerfile string

const (
	backendCUDA   = "cuda"
	backendJetson = "jetson"

	defaultCUDABuildImage   = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
	defaultCUDARuntimeImage = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"
	defaultJetsonImage      = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
	defaultJetsonCudaArch   = "87" // Orin
	defaultBuildKeep        = 3

	buildLabel       = "llmctl.build"
	buildCommitLabel = "llmctl.commit"
	buildRefLabel    = "llmctl.ref"
	buildSourceLabel = "llmctl.source"
	buildSocketLabel = "llmctl.unix_socket"
	buildTimeLabel   = "llmctl.built_at" // RFC3339Nano; image Created is not reliable for ordering
)

// gitExec runs git and returns stdout. A variable so tests can stub it.
var gitExec = func(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return string(out), fmt.Errorf("git %s: %s", strings.Join(args, " "), msg)
	}
	return string(out), nil
}

// buildRepo is the image repository for a build, e.g. "llmctl/ik".
func buildRepo(name string) string {
	return "llmctl/" + strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		}
		return '-'
	}, name)
}

func expandHome(p string) string {
	if p == "~" {
		return homeDir()
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(homeDir(), p[2:])
	}
	return p
}

var (
	fullSHARe  = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)
	shortSHARe = regexp.MustCompile(`^[0-9a-fA-F]{7,39}$`)
	prRefRe    = regexp.MustCompile(`^#(\d+)$`)
)

// gitRefQuery maps a build ref to what `git ls-remote` should look up.
func gitRefQuery(ref string) string {
	if ref == "" {
		return "HEAD"
	}
	if m := prRefRe.FindStringSubmatch(ref); m != nil {
		return "refs/pull/" + m[1] + "/head"
	}
	return ref
}

// pickLsRemote picks the commit for ref from `git ls-remote` output,
// preferring branches over tags and peeled (^{}) tags over tag objects.
func pickLsRemote(out, ref string) (string, error) {
	refs := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 {
			refs[f[1]] = f[0]
		}
	}
	q := gitRefQuery(ref)
	for _, cand := range []string{q, "refs/heads/" + q, "refs/tags/" + q + "^{}", "refs/tags/" + q} {
		if sha, ok := refs[cand]; ok {
			return sha, nil
		}
	}
	return "", fmt.Errorf("ref %q not found", ref)
}

// resolveRemoteCommit returns the full commit SHA ref points to in repo.
func resolveRemoteCommit(repo, ref string) (string, error) {
	if fullSHARe.MatchString(ref) {
		return strings.ToLower(ref), nil
	}
	if shortSHARe.MatchString(ref) {
		return "", fmt.Errorf("short commit %q can't be resolved remotely; use the full 40-character SHA", ref)
	}
	out, err := gitExec("ls-remote", repo, gitRefQuery(ref))
	if err != nil {
		return "", err
	}
	sha, err := pickLsRemote(out, ref)
	if err != nil {
		return "", fmt.Errorf("%s: %w", repo, err)
	}
	return sha, nil
}

// buildSourceDir is where llmctl keeps the fetched source for a build.
func buildSourceDir(name string) string {
	return filepath.Join(homeDir(), ".cache", "llmctl", "src", strings.TrimPrefix(buildRepo(name), "llmctl/"))
}

// fetchRemoteSource shallow-fetches commit (falling back to the ref) into dir
// and checks it out. Returns the checked-out commit.
func fetchRemoteSource(dir, repo, ref, commit string) (string, error) {
	if _, err := os.Stat(filepath.Join(dir, ".git")); err != nil {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", err
		}
		if _, err := gitExec("-C", dir, "init", "-q"); err != nil {
			return "", err
		}
	}
	if _, err := gitExec("-C", dir, "fetch", "-q", "--depth", "1", "--no-tags", repo, commit); err != nil {
		// Some servers don't allow fetching by SHA; fetch the ref instead.
		if _, err2 := gitExec("-C", dir, "fetch", "-q", "--depth", "1", "--no-tags", repo, gitRefQuery(ref)); err2 != nil {
			return "", err2
		}
	}
	if _, err := gitExec("-C", dir, "checkout", "-q", "--force", "--detach", "FETCH_HEAD^{commit}"); err != nil {
		return "", err
	}
	if _, err := gitExec("-C", dir, "clean", "-q", "-ffdx"); err != nil {
		return "", err
	}
	out, err := gitExec("-C", dir, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

// localSourceState returns the HEAD commit of a local checkout and, if it has
// uncommitted changes, a short hash of them.
func localSourceState(path string) (commit, dirty string, err error) {
	out, err := gitExec("-C", path, "rev-parse", "HEAD")
	if err != nil {
		return "", "", fmt.Errorf("%s is not a git checkout: %w", path, err)
	}
	commit = strings.TrimSpace(out)
	status, err := gitExec("-C", path, "status", "--porcelain", "--untracked-files=all")
	if err != nil {
		return "", "", err
	}
	if strings.TrimSpace(status) == "" {
		return commit, "", nil
	}
	h := sha256.New()
	diff, err := gitExec("-C", path, "diff", "HEAD", "--binary")
	if err != nil {
		return "", "", err
	}
	_, _ = io.WriteString(h, diff)
	untracked, err := gitExec("-C", path, "ls-files", "-z", "--others", "--exclude-standard")
	if err != nil {
		return "", "", err
	}
	for _, f := range strings.Split(untracked, "\x00") {
		if f == "" {
			continue
		}
		_, _ = io.WriteString(h, f+"\x00")
		if data, err := os.ReadFile(filepath.Join(path, f)); err == nil {
			_, _ = h.Write(data)
		}
	}
	return commit, hex.EncodeToString(h.Sum(nil))[:8], nil
}

// copyLocalSource copies a checkout's tracked and untracked-but-not-ignored
// files into dst, so ignored build output (build/, models) stays out of the
// docker build context.
func copyLocalSource(src, dst string) error {
	out, err := gitExec("-C", src, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
	if err != nil {
		return err
	}
	if err := os.RemoveAll(dst); err != nil {
		return err
	}
	for _, rel := range strings.Split(out, "\x00") {
		if rel == "" {
			continue
		}
		from := filepath.Join(src, rel)
		info, err := os.Lstat(from)
		if err != nil {
			continue // deleted but not yet committed
		}
		to := filepath.Join(dst, rel)
		if err := os.MkdirAll(filepath.Dir(to), 0755); err != nil {
			return err
		}
		switch {
		case info.Mode()&os.ModeSymlink != 0:
			target, err := os.Readlink(from)
			if err != nil {
				return err
			}
			if err := os.Symlink(target, to); err != nil {
				return err
			}
		case info.Mode().IsRegular():
			data, err := os.ReadFile(from)
			if err != nil {
				return err
			}
			if err := os.WriteFile(to, data, info.Mode().Perm()); err != nil {
				return err
			}
		}
	}
	return nil
}

// sourceSupportsUnixSocket reports whether llama-server in this source tree
// can listen on a unix socket (upstream PR #12613, `--host *.sock`).
func sourceSupportsUnixSocket(srcDir string) bool {
	found := false
	for _, sub := range []string{"tools/server", "examples/server"} {
		root := filepath.Join(srcDir, sub)
		_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
			if found {
				return filepath.SkipAll
			}
			if err != nil {
				return filepath.SkipDir
			}
			if d.IsDir() {
				return nil
			}
			ext := filepath.Ext(p)
			if ext != ".cpp" && ext != ".h" && ext != ".hpp" {
				return nil
			}
			if data, err := os.ReadFile(p); err == nil && bytes.Contains(data, []byte("AF_UNIX")) {
				found = true
			}
			return nil
		})
	}
	return found
}

// parseComputeCaps turns `nvidia-smi --query-gpu=compute_cap` output
// ("8.9\n8.6\n") into CMAKE_CUDA_ARCHITECTURES ("86;89").
func parseComputeCaps(out string) string {
	seen := map[string]bool{}
	var archs []string
	for _, line := range strings.Split(out, "\n") {
		a := strings.ReplaceAll(strings.TrimSpace(line), ".", "")
		if a == "" || seen[a] {
			continue
		}
		if _, err := strconv.Atoi(a); err != nil {
			continue
		}
		seen[a] = true
		archs = append(archs, a)
	}
	sort.Strings(archs)
	return strings.Join(archs, ";")
}

// jetsonImageForRelease maps /etc/nv_tegra_release to an l4t-jetpack tag,
// e.g. "# R36 (release), REVISION: 4.3, ..." -> nvcr.io/nvidia/l4t-jetpack:r36.4.0.
func jetsonImageForRelease(release string) string {
	m := regexp.MustCompile(`R(\d+) \(release\), REVISION: (\d+)`).FindStringSubmatch(release)
	if m == nil {
		return ""
	}
	return fmt.Sprintf("nvcr.io/nvidia/l4t-jetpack:r%s.%s.0", m[1], m[2])
}

// resolvedBuild is a BuildConfig with all defaults filled in.
type resolvedBuild struct {
	Name         string
	Backend      string
	CudaArch     string
	BaseImage    string
	RuntimeImage string
	CmakeArgs    []string
	Target       string
	Dockerfile   string
	Keep         int
}

func resolveBuildConfig(name string, bc BuildConfig) (resolvedBuild, error) {
	rb := resolvedBuild{
		Name:         name,
		Backend:      strings.ToLower(bc.Backend),
		CudaArch:     bc.CudaArch,
		BaseImage:    bc.BaseImage,
		RuntimeImage: bc.RuntimeImage,
		CmakeArgs:    bc.CmakeArgs,
		Target:       bc.Target,
		Dockerfile:   bc.Dockerfile,
		Keep:         bc.Keep,
	}
	if (bc.Repo == "") == (bc.Path == "") {
		return rb, fmt.Errorf("build %q: set exactly one of repo or path", name)
	}
	if bc.Path != "" && bc.Ref != "" {
		return rb, fmt.Errorf("build %q: ref only applies to repo builds", name)
	}
	if rb.Backend == "" {
		rb.Backend = backendCUDA
	}
	if rb.Target == "" {
		rb.Target = "llama-server"
	}
	if rb.Keep <= 0 {
		rb.Keep = defaultBuildKeep
	}
	switch rb.Backend {
	case backendCUDA:
		if rb.BaseImage == "" {
			rb.BaseImage = defaultCUDABuildImage
		}
		if rb.RuntimeImage == "" {
			rb.RuntimeImage = defaultCUDARuntimeImage
		}
		if rb.CudaArch == "" {
			out, err := exec.Command("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader").Output()
			if err == nil {
				rb.CudaArch = parseComputeCaps(string(out))
			}
			if rb.CudaArch == "" {
				return rb, fmt.Errorf("build %q: can't detect the GPU architecture (nvidia-smi); set cuda_arch, e.g. \"89\"", name)
			}
		}
	case backendJetson:
		if rb.BaseImage == "" {
			if data, err := os.ReadFile("/etc/nv_tegra_release"); err == nil {
				rb.BaseImage = jetsonImageForRelease(string(data))
			}
			if rb.BaseImage == "" {
				rb.BaseImage = defaultJetsonImage
			}
		}
		if rb.RuntimeImage == "" {
			// JetPack 6 containers must ship CUDA themselves; reuse the
			// JetPack image so cudart/cublas are present at run time.
			rb.RuntimeImage = rb.BaseImage
		}
		if rb.CudaArch == "" {
			rb.CudaArch = defaultJetsonCudaArch
		}
	default:
		return rb, fmt.Errorf("build %q: backend must be cuda or jetson, got %q", name, bc.Backend)
	}
	return rb, nil
}

type buildImageSpec struct {
	Build      resolvedBuild
	ContextDir string
	Dockerfile string // path to the Dockerfile to use
	Tag        string // commit tag, e.g. abc123def456 or abc123def456-dirty-1a2b3c4d
	Commit     string
	Ref        string
	Source     string
	UnixSocket bool
	BuiltAt    time.Time
}

// dockerBuildArgs builds the `docker build` argument list.
func dockerBuildArgs(spec buildImageSpec) []string {
	repo := buildRepo(spec.Build.Name)
	return []string{
		"build",
		"-f", spec.Dockerfile,
		"--build-arg", "BUILD_IMAGE=" + spec.Build.BaseImage,
		"--build-arg", "RUNTIME_IMAGE=" + spec.Build.RuntimeImage,
		"--build-arg", "CUDA_ARCH=" + spec.Build.CudaArch,
		"--build-arg", "CMAKE_ARGS=" + strings.Join(spec.Build.CmakeArgs, " "),
		"--build-arg", "TARGET=" + spec.Build.Target,
		"--build-arg", "CCACHE_ID=llmctl-ccache-" + spec.Build.Backend,
		"--label", buildLabel + "=" + spec.Build.Name,
		"--label", buildCommitLabel + "=" + spec.Commit,
		"--label", buildRefLabel + "=" + spec.Ref,
		"--label", buildSourceLabel + "=" + spec.Source,
		"--label", buildSocketLabel + "=" + strconv.FormatBool(spec.UnixSocket),
		"--label", buildTimeLabel + "=" + spec.BuiltAt.UTC().Format(time.RFC3339Nano),
		"-t", repo + ":" + spec.Tag,
		"-t", repo + ":latest",
		spec.ContextDir,
	}
}

type builtImage struct {
	Ref     string // repo:tag
	Tag     string
	ID      string
	Created time.Time
}

func shortImageID(id string) string {
	id = strings.TrimPrefix(id, "sha256:")
	if len(id) > 12 {
		id = id[:12]
	}
	return id
}

// listBuildImages returns the tagged images of a build (including :latest),
// ordered by llmctl's own built_at label (image Created is not reliable:
// BuildKit may reuse or backdate it).
func listBuildImages(name string) ([]builtImage, error) {
	out, err := dockerExec("image", "ls", "--filter", "label="+buildLabel+"="+name,
		"--format", "{{.Repository}}:{{.Tag}}|{{.ID}}")
	if err != nil {
		return nil, err
	}
	var imgs []builtImage
	var refs []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.SplitN(line, "|", 2)
		if len(f) != 2 || strings.HasSuffix(f[0], ":<none>") {
			continue
		}
		imgs = append(imgs, builtImage{
			Ref: f[0],
			Tag: f[0][strings.LastIndex(f[0], ":")+1:],
			ID:  shortImageID(f[1]),
		})
		refs = append(refs, f[0])
	}
	if len(refs) == 0 {
		return imgs, nil
	}
	args := append([]string{"image", "inspect", "-f",
		fmt.Sprintf("{{.Created}}|{{index .Config.Labels %q}}", buildTimeLabel)}, refs...)
	out, err = dockerExec(args...)
	if err != nil {
		return imgs, nil
	}
	for i, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if i >= len(imgs) {
			break
		}
		created, label, _ := strings.Cut(strings.TrimSpace(line), "|")
		if t, err := time.Parse(time.RFC3339Nano, label); err == nil {
			imgs[i].Created = t
		} else if t, err := time.Parse(time.RFC3339Nano, created); err == nil {
			imgs[i].Created = t
		}
	}
	return imgs, nil
}

// imagesToPrune picks commit-tagged images beyond the newest `keep`, never
// touching :latest, the image :latest points to, or protected (in-use) IDs.
func imagesToPrune(imgs []builtImage, keep int, protected map[string]bool) []string {
	latestID := ""
	var tagged []builtImage
	for _, img := range imgs {
		if img.Tag == "latest" {
			latestID = img.ID
			continue
		}
		tagged = append(tagged, img)
	}
	sort.SliceStable(tagged, func(i, j int) bool { return tagged[i].Created.After(tagged[j].Created) })
	var prune []string
	for i, img := range tagged {
		if i < keep || img.ID == latestID || protected[img.ID] {
			continue
		}
		prune = append(prune, img.Ref)
	}
	return prune
}

// inUseImageIDs returns short IDs of images used by llmctl containers.
func inUseImageIDs() map[string]bool {
	ids := map[string]bool{}
	out, err := dockerExec("ps", "-aq", "--filter", "label="+containerLabel)
	if err != nil || strings.TrimSpace(out) == "" {
		return ids
	}
	args := append([]string{"inspect", "-f", "{{.Image}}"}, strings.Fields(out)...)
	if out, err := dockerExec(args...); err == nil {
		for _, id := range strings.Fields(out) {
			ids[shortImageID(id)] = true
		}
	}
	return ids
}

func pruneBuildImages(name string, keep int, verbose bool) {
	imgs, err := listBuildImages(name)
	if err != nil {
		return
	}
	for _, ref := range imagesToPrune(imgs, keep, inUseImageIDs()) {
		if verbose {
			fmt.Printf("  Removing old image %s\n", ref)
		}
		_, _ = dockerExec("image", "rm", ref)
	}
}

type buildOptions struct {
	Ref   string // overrides the configured ref
	Force bool
}

// runBuild builds (or re-tags) the image for a configured build.
func runBuild(cfg Config, name string, opts buildOptions) error {
	bc, ok := cfg.Builds[name]
	if !ok {
		return fmt.Errorf("no build %q in config (see `llmctl builds`)", name)
	}
	if opts.Ref != "" {
		if bc.Path != "" {
			return fmt.Errorf("build %q is a local checkout; --ref only applies to repo builds", name)
		}
		bc.Ref = opts.Ref
	}
	rb, err := resolveBuildConfig(name, bc)
	if err != nil {
		return err
	}
	repo := buildRepo(name)

	// Rollback: an existing image for a (short) commit just becomes :latest.
	if bc.Repo != "" && (shortSHARe.MatchString(bc.Ref) || fullSHARe.MatchString(bc.Ref)) {
		if ref, err := findBuildImage(name, bc.Ref); err == nil && !opts.Force {
			if _, err := dockerExec("tag", ref, repo+":latest"); err != nil {
				return err
			}
			fmt.Printf("✓ %s:latest → %s (already built)\n", repo, ref)
			return nil
		}
	}

	srcDir := buildSourceDir(name)
	var commit, tag, ref, source string
	if bc.Repo != "" {
		ref = bc.Ref
		if ref == "" {
			ref = "HEAD"
		}
		source = bc.Repo
		fmt.Printf("Resolving %s @ %s...\n", bc.Repo, ref)
		commit, err = resolveRemoteCommit(bc.Repo, ref)
		if err != nil {
			return err
		}
		tag = commit[:12]
		if !opts.Force {
			if _, err := dockerExec("image", "inspect", repo+":"+tag); err == nil {
				if _, err := dockerExec("tag", repo+":"+tag, repo+":latest"); err != nil {
					return err
				}
				fmt.Printf("✓ %s is up to date (%s:%s)\n", name, repo, tag)
				return nil
			}
		}
		fmt.Printf("Fetching %s...\n", commit[:12])
		if commit, err = fetchRemoteSource(srcDir, bc.Repo, ref, commit); err != nil {
			return err
		}
		tag = commit[:12]
	} else {
		path := expandHome(bc.Path)
		source = path
		ref = "local"
		head, dirty, err := localSourceState(path)
		if err != nil {
			return err
		}
		commit, tag = head, head[:12]
		if dirty != "" {
			tag += "-dirty-" + dirty
		}
		if !opts.Force {
			if _, err := dockerExec("image", "inspect", repo+":"+tag); err == nil {
				if _, err := dockerExec("tag", repo+":"+tag, repo+":latest"); err != nil {
					return err
				}
				fmt.Printf("✓ %s is up to date (%s:%s)\n", name, repo, tag)
				return nil
			}
		}
		fmt.Printf("Copying %s...\n", path)
		if err := copyLocalSource(path, srcDir); err != nil {
			return err
		}
	}

	dockerfile := rb.Dockerfile
	if dockerfile == "" {
		f, err := os.CreateTemp("", "llmctl-*.Dockerfile")
		if err != nil {
			return err
		}
		defer func() { _ = os.Remove(f.Name()) }()
		_, werr := f.WriteString(llamaServerDockerfile)
		if cerr := f.Close(); werr == nil {
			werr = cerr
		}
		if werr != nil {
			return werr
		}
		dockerfile = f.Name()
	} else if !filepath.IsAbs(expandHome(dockerfile)) {
		dockerfile = filepath.Join(srcDir, dockerfile)
	} else {
		dockerfile = expandHome(dockerfile)
	}

	spec := buildImageSpec{
		Build:      rb,
		ContextDir: srcDir,
		Dockerfile: dockerfile,
		Tag:        tag,
		Commit:     commit,
		Ref:        ref,
		Source:     source,
		UnixSocket: sourceSupportsUnixSocket(srcDir),
		BuiltAt:    time.Now(),
	}
	fmt.Printf("Building %s:%s (%s, CUDA arch %s, base %s)...\n", repo, tag, rb.Backend, rb.CudaArch, rb.BaseImage)
	if !spec.UnixSocket {
		fmt.Println("  note: this source has no unix-socket support in llama-server; load it with --network internal (or docker_network \"internal\")")
	}
	start := time.Now()
	cmd := exec.Command(dockerBin, dockerBuildArgs(spec)...)
	cmd.Env = append(os.Environ(), "DOCKER_BUILDKIT=1")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}
	fmt.Printf("✓ Built %s:%s in %s\n", repo, tag, time.Since(start).Round(time.Second))
	pruneBuildImages(name, rb.Keep, true)
	return nil
}

// findBuildImage returns the image ref for a build, optionally pinned to a
// commit prefix ("" = :latest).
func findBuildImage(name, commit string) (string, error) {
	repo := buildRepo(name)
	if commit == "" {
		if _, err := dockerExec("image", "inspect", repo+":latest"); err != nil {
			return "", fmt.Errorf("build %q has no image yet; run `llmctl build %s`", name, name)
		}
		return repo + ":latest", nil
	}
	imgs, err := listBuildImages(name)
	if err != nil {
		return "", err
	}
	commit = strings.ToLower(commit)
	var matches []string
	for _, img := range imgs {
		if img.Tag != "latest" && (strings.HasPrefix(img.Tag, commit) || strings.HasPrefix(commit, img.Tag)) {
			matches = append(matches, img.Ref)
		}
	}
	switch len(matches) {
	case 0:
		return "", fmt.Errorf("build %q has no image for commit %s; run `llmctl build %s --ref <full sha>`", name, commit, name)
	case 1:
		return matches[0], nil
	}
	return "", fmt.Errorf("commit %s is ambiguous for build %q: %s", commit, name, strings.Join(matches, ", "))
}

// buildImageInfo returns a build image's commit and unix-socket support.
func buildImageInfo(image string) (commit string, unixSocket bool, err error) {
	out, err := dockerExec("image", "inspect", "-f",
		fmt.Sprintf("{{index .Config.Labels %q}}|{{index .Config.Labels %q}}", buildCommitLabel, buildSocketLabel), image)
	if err != nil {
		return "", false, err
	}
	f := strings.SplitN(strings.TrimSpace(out), "|", 2)
	if len(f) != 2 {
		return "", false, fmt.Errorf("unexpected inspect output %q", out)
	}
	return f[0], f[1] == "true", nil
}

// splitBuildRef splits "name@commit" into its parts.
func shortCommit(c string) string {
	if len(c) > 12 {
		return c[:12]
	}
	return c
}

func splitBuildRef(ref string) (name, commit string) {
	if i := strings.LastIndex(ref, "@"); i >= 0 {
		return ref[:i], ref[i+1:]
	}
	return ref, ""
}

var (
	githubURLRe   = regexp.MustCompile(`^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?(?:/(tree|pull|commit)/(.+?))?/?(?:[#?].*)?$`)
	ownerRepoRe   = regexp.MustCompile(`^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$`)
	reservedBuild = map[string]bool{"add": true, "rm": true, "remove": true, "ls": true, "list": true}
)

// parseBuildSource turns what a user pastes (a GitHub URL — repo, branch,
// PR or commit page —, owner/repo, any git URL, or a local path) into a
// BuildConfig and a suggested build name. ok is false if arg doesn't look
// like a source (e.g. it is the name of a configured build).
func parseBuildSource(arg string) (bc BuildConfig, name string, ok bool) {
	if arg == "" {
		return bc, "", false
	}
	if strings.HasPrefix(arg, "/") || strings.HasPrefix(arg, "./") || strings.HasPrefix(arg, "../") ||
		strings.HasPrefix(arg, "~") || arg == "." {
		p := expandHome(arg)
		if abs, err := filepath.Abs(p); err == nil {
			p = abs
		}
		return BuildConfig{Path: p}, filepath.Base(p), true
	}
	if m := githubURLRe.FindStringSubmatch(arg); m != nil {
		owner, repo := m[1], m[2]
		bc.Repo = "https://github.com/" + owner + "/" + repo
		name = owner
		switch m[3] {
		case "tree":
			bc.Ref = m[4]
		case "pull":
			n := strings.SplitN(m[4], "/", 2)[0] // drop /files, /commits, ...
			bc.Ref = "#" + n
			name = owner + "-pr" + n
		case "commit":
			bc.Ref = m[4]
		}
		return bc, name, true
	}
	if strings.Contains(arg, "://") || strings.HasPrefix(arg, "git@") {
		base := arg[strings.LastIndexAny(arg, "/:")+1:]
		return BuildConfig{Repo: arg}, strings.TrimSuffix(base, ".git"), true
	}
	if m := ownerRepoRe.FindStringSubmatch(arg); m != nil {
		if _, err := os.Stat(arg); err != nil { // not a relative directory
			return BuildConfig{Repo: "https://github.com/" + m[1] + "/" + m[2]}, m[1], true
		}
		p, _ := filepath.Abs(arg)
		return BuildConfig{Path: p}, filepath.Base(p), true
	}
	return bc, "", false
}

// sanitizeBuildName makes a build name that is also a valid image/dir name.
func sanitizeBuildName(name string) string {
	n := strings.Trim(strings.TrimPrefix(buildRepo(name), "llmctl/"), "-.")
	if n == "" || reservedBuild[n] {
		n = "build-" + n
	}
	return n
}

// sameBuildSource reports whether two builds fetch the same thing.
func sameBuildSource(a, b BuildConfig) bool {
	return a.Repo == b.Repo && a.Path == b.Path && a.Ref == b.Ref
}

// uniqueBuildName returns name, or name-2, name-3... if name is taken by a
// build with a different source. Reusing an identical source keeps the name.
func uniqueBuildName(builds map[string]BuildConfig, name string, bc BuildConfig) string {
	candidate := name
	for i := 2; ; i++ {
		existing, taken := builds[candidate]
		if !taken || sameBuildSource(existing, bc) {
			return candidate
		}
		candidate = fmt.Sprintf("%s-%d", name, i)
	}
}

// buildFlags are the `llmctl build` flags that define (not just run) a build.
type buildFlags struct {
	Ref, Backend, CudaArch, Dockerfile string
	CmakeArgs                          []string
}

func (f buildFlags) apply(bc *BuildConfig) {
	if f.Ref != "" {
		bc.Ref = f.Ref
	}
	if f.Backend != "" {
		bc.Backend = f.Backend
	}
	if f.CudaArch != "" {
		bc.CudaArch = f.CudaArch
	}
	if f.Dockerfile != "" {
		bc.Dockerfile = f.Dockerfile
	}
	if len(f.CmakeArgs) > 0 {
		bc.CmakeArgs = f.CmakeArgs
	}
}

// addBuild stores a build in the config and returns its (possibly adjusted) name.
func addBuild(cfg *Config, name string, bc BuildConfig) (string, error) {
	if cfg.Builds == nil {
		cfg.Builds = map[string]BuildConfig{}
	}
	if _, err := resolveBuildSpecOnly(name, bc); err != nil {
		return "", err
	}
	cfg.Builds[name] = bc
	return name, saveConfig(*cfg)
}

// resolveBuildSpecOnly validates a build without host probing (nvidia-smi).
func resolveBuildSpecOnly(name string, bc BuildConfig) (resolvedBuild, error) {
	if bc.CudaArch == "" {
		bc.CudaArch = "0" // placeholder: arch is detected at build time
	}
	return resolveBuildConfig(name, bc)
}

// removeBuild deletes a build from the config plus its images and source.
func removeBuild(cfg *Config, name string) error {
	if _, ok := cfg.Builds[name]; !ok {
		return fmt.Errorf("no build %q in config", name)
	}
	imgs, _ := listBuildImages(name)
	inUse := inUseImageIDs()
	for _, img := range imgs {
		if inUse[img.ID] {
			return fmt.Errorf("build %q is used by a loaded model (%s); unload it first", name, img.Ref)
		}
	}
	for _, img := range imgs {
		_, _ = dockerExec("image", "rm", img.Ref)
	}
	_ = os.RemoveAll(buildSourceDir(name))
	delete(cfg.Builds, name)
	return saveConfig(*cfg)
}

// cmdBuildCLI handles `llmctl build ...`.
func cmdBuildCLI(cfg Config, args []string) {
	usage := func() {
		fmt.Fprintln(os.Stderr, `Usage:
  llmctl build <name|source>... [--ref R] [--force] [--backend cuda|jetson] [--cuda-arch A] [--cmake-arg ARG]... [--dockerfile F]
  llmctl build --all
  llmctl build add <name> <source> [flags]
  llmctl build rm <name>

<source> is a GitHub URL (repo, /tree/<branch>, /pull/<N>, /commit/<sha>), owner/repo, any git URL, or a local path.`)
		os.Exit(1)
	}
	var f buildFlags
	var positional []string
	force, all := false, false
	for i := 0; i < len(args); i++ {
		a := args[i]
		val := func() string {
			if k, v, ok := strings.Cut(a, "="); ok && strings.HasPrefix(k, "--") {
				return v
			}
			if i+1 >= len(args) {
				fmt.Fprintf(os.Stderr, "Error: %s needs a value\n", a)
				os.Exit(1)
			}
			i++
			return args[i]
		}
		flag, _, _ := strings.Cut(a, "=")
		switch flag {
		case "--ref":
			f.Ref = val()
		case "--backend":
			f.Backend = val()
		case "--cuda-arch":
			f.CudaArch = val()
		case "--dockerfile":
			f.Dockerfile = val()
		case "--cmake-arg":
			f.CmakeArgs = append(f.CmakeArgs, val())
		case "--force", "-f":
			force = true
		case "--all":
			all = true
		case "-h", "--help":
			usage()
		default:
			if strings.HasPrefix(a, "-") {
				fmt.Fprintf(os.Stderr, "Unknown flag: %s\n", a)
				usage()
			}
			positional = append(positional, a)
		}
	}
	fail := func(err error) {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if len(positional) > 0 && (positional[0] == "rm" || positional[0] == "remove") {
		if len(positional) != 2 {
			usage()
		}
		if err := removeBuild(&cfg, positional[1]); err != nil {
			fail(err)
		}
		fmt.Printf("✓ Removed build %q (config, images, source)\n", positional[1])
		return
	}

	if len(positional) > 0 && positional[0] == "add" {
		if len(positional) != 3 {
			usage()
		}
		bc, _, ok := parseBuildSource(positional[2])
		if !ok {
			fail(fmt.Errorf("%q is not a git URL, owner/repo or path", positional[2]))
		}
		f.apply(&bc)
		name := sanitizeBuildName(positional[1])
		if _, err := addBuild(&cfg, name, bc); err != nil {
			fail(err)
		}
		fmt.Printf("✓ Added build %q. Build it with: llmctl build %s\n", name, name)
		return
	}

	if all {
		cmdBuild(cfg, nil, true, buildOptions{Ref: f.Ref, Force: force})
		return
	}
	if len(positional) == 0 {
		usage()
	}

	// Positional args are configured build names, or sources to add first.
	defines := f.Backend != "" || f.CudaArch != "" || f.Dockerfile != "" || len(f.CmakeArgs) > 0
	var names []string
	refForRun := f.Ref
	for _, arg := range positional {
		if bc, ok := cfg.Builds[arg]; ok {
			if defines {
				f.apply(&bc)
				bc.Ref = cfg.Builds[arg].Ref // --ref stays a one-off override for existing builds
				if _, err := addBuild(&cfg, arg, bc); err != nil {
					fail(err)
				}
			}
			names = append(names, arg)
			continue
		}
		bc, suggested, ok := parseBuildSource(arg)
		if !ok {
			fail(fmt.Errorf("no build %q in config, and it's not a git URL, owner/repo or path (see `llmctl builds`)", arg))
		}
		f.apply(&bc)
		name := uniqueBuildName(cfg.Builds, sanitizeBuildName(suggested), bc)
		if _, exists := cfg.Builds[name]; !exists {
			if _, err := addBuild(&cfg, name, bc); err != nil {
				fail(err)
			}
			fmt.Printf("Added build %q → %s\n", name, describeBuildSource(bc))
		}
		refForRun = "" // the ref is part of the new build's definition
		names = append(names, name)
	}
	if refForRun != "" && len(names) != 1 {
		fail(fmt.Errorf("--ref needs exactly one build"))
	}
	cmdBuild(cfg, names, false, buildOptions{Ref: refForRun, Force: force})
	if len(names) == 1 {
		fmt.Printf("\n  Try it: llmctl load <model> --build %s\n", names[0])
	}
}

func describeBuildSource(bc BuildConfig) string {
	if bc.Path != "" {
		return bc.Path
	}
	ref := bc.Ref
	if ref == "" {
		ref = "HEAD"
	}
	return bc.Repo + " @ " + ref
}

func cmdBuild(cfg Config, names []string, all bool, opts buildOptions) {
	if all {
		names = names[:0]
		for name := range cfg.Builds {
			names = append(names, name)
		}
		sort.Strings(names)
	}
	if len(names) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: llmctl build <name>... [--ref <ref>] [--force] | --all")
		os.Exit(1)
	}
	failed := 0
	for _, name := range names {
		if err := runBuild(cfg, name, opts); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s: %v\n", name, err)
			failed++
		}
	}
	if failed > 0 {
		os.Exit(1)
	}
}

func cmdBuilds(cfg Config) {
	if len(cfg.Builds) == 0 {
		fmt.Println("No builds configured. Add a `builds` section to ~/.llmctl.json.")
		return
	}
	names := make([]string, 0, len(cfg.Builds))
	for name := range cfg.Builds {
		names = append(names, name)
	}
	sort.Strings(names)

	usedBy := map[string][]string{}
	for model, mc := range cfg.Models {
		if mc.DockerBuild != nil {
			b, _ := splitBuildRef(*mc.DockerBuild)
			usedBy[b] = append(usedBy[b], model)
		}
	}
	if cfg.Docker.Build != "" {
		b, _ := splitBuildRef(cfg.Docker.Build)
		usedBy[b] = append(usedBy[b], "(default)")
	}

	fmt.Printf("  %-16s %-8s %-14s %-12s %s\n", "NAME", "BACKEND", "LATEST", "BUILT", "SOURCE")
	fmt.Printf("  %-16s %-8s %-14s %-12s %s\n", "────", "───────", "──────", "─────", "──────")
	for _, name := range names {
		bc := cfg.Builds[name]
		source := describeBuildSource(bc)
		backend := bc.Backend
		if backend == "" {
			backend = backendCUDA
		}
		latest, built := "-", "never"
		imgs, _ := listBuildImages(name)
		latestID := ""
		for _, img := range imgs {
			if img.Tag == "latest" {
				latestID = img.ID
				built = formatAge(img.Created)
			}
		}
		count := 0
		for _, img := range imgs {
			if img.Tag == "latest" {
				continue
			}
			count++
			if img.ID == latestID {
				latest = img.Tag
				if len(latest) > 14 {
					latest = latest[:14]
				}
			}
		}
		fmt.Printf("  %-16s %-8s %-14s %-12s %s\n", name, backend, latest, built, source)
		if count > 1 {
			fmt.Printf("    images: %d kept\n", count)
		}
		if models := usedBy[name]; len(models) > 0 {
			sort.Strings(models)
			fmt.Printf("    used by: %s\n", strings.Join(models, ", "))
		}
	}
}

func formatAge(t time.Time) string {
	if t.IsZero() {
		return "?"
	}
	d := time.Since(t)
	switch {
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	}
	return fmt.Sprintf("%dd ago", int(d.Hours()/24))
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
			candidate := filepath.Join(snapshotsDir, e.Name(), filepath.FromSlash(parts[1]))
			if _, err := os.Stat(candidate); err == nil {
				return candidate
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

func autoswitchModel(cfg Config, targetName string) (backendTarget, string, error) {
	if !cfg.Autoswitch.Enabled {
		return backendTarget{}, "", fmt.Errorf("model %q is not loaded", targetName)
	}

	spec, err := resolveLoadSpec(cfg, loadOptions{ModelName: targetName})
	if err != nil {
		return backendTarget{}, "", err
	}
	if spec.ConfigKey == "" {
		return backendTarget{}, "", fmt.Errorf("model %q is not configured for autoswitch", targetName)
	}
	mc := cfg.Models[spec.ConfigKey]
	if !mc.AutoLoad {
		return backendTarget{}, "", fmt.Errorf("model %q is not configured with auto_load", targetName)
	}
	requiredMB := mc.VramMB

	reg := loadRegistry()
	reg.CleanDead()

	availableMB, err := autoswitchAvailableMB(cfg)
	if err != nil {
		return backendTarget{}, "", err
	}

	if availableMB < requiredMB {
		candidates, hasCandidates := autoswitchEvictionPlan(reg, spec.InstanceName, requiredMB, availableMB)
		if !hasCandidates {
			return backendTarget{}, "", fmt.Errorf("model %q needs %d MB but only %d MB available on GPU 0, no auto_unload models to evict", targetName, requiredMB, availableMB)
		}
		evicted := 0
		for _, inst := range candidates {
			if err := unloadInstance(inst.Name, false); err != nil {
				return backendTarget{}, "", fmt.Errorf("failed to evict %q: %w", inst.Name, err)
			}
			evicted++
			newAvail, checkErr := autoswitchAvailableMB(cfg)
			if checkErr == nil && newAvail >= requiredMB {
				break
			}
		}
		availableMB, err = autoswitchAvailableMB(cfg)
		if err != nil {
			return backendTarget{}, "", err
		}
		if availableMB < requiredMB {
			return backendTarget{}, "", fmt.Errorf("model %q needs %d MB but only %d MB available after evicting %d auto_unload models", targetName, requiredMB, availableMB, evicted)
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
		return backendTarget{}, "", err
	}
	if !healthy {
		stopInstance(inst)
		reg := loadRegistry()
		reg.Remove(inst.Name)
		saveRegistry(reg)
		return backendTarget{}, "", fmt.Errorf("model %q did not become healthy before timeout", inst.Name)
	}
	return instanceBackend(inst), inst.Name, nil
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
			out = append(out, row{name, instanceRunning(*inst), vram, inst.Port})
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
	stopInstance(*inst)
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
	backends := map[string]backendTarget{}

	rebuildBackends := func() {
		mu.Lock()
		defer mu.Unlock()
		r := loadRegistry()
		r.CleanDead()
		for k := range backends {
			delete(backends, k)
		}
		for _, inst := range r.Instances {
			backends[inst.Name] = instanceBackend(inst)
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
				target = instanceBackend(*def)
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
				req.URL.Scheme = target.URL.Scheme
				req.URL.Host = target.URL.Host
				req.Host = target.URL.Host
			},
			Transport: target.Transport,
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
		fmt.Printf("    %-28s → %s%s\n", name, u, def)
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
	Build           string // one-off `--build name[@commit]`, overrides config
	Network         string // one-off `--network mode`, overrides config
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
		if opts.Build != "" {
			// Run a one-off build next to the regular instance, not over it.
			b, _ := splitBuildRef(opts.Build)
			instanceName += "-" + b
		}
	}

	configKey := modelConfigKey(cfg, opts.ModelName, instanceName, modelPath)
	if configKey != "" {
		cfg = configForModel(cfg, configKey)
	}
	if opts.Build != "" {
		cfg.Docker.Build = opts.Build
	}
	if opts.Network != "" {
		if !validNetworkMode(opts.Network) {
			return loadSpec{}, fmt.Errorf("--network must be none, internal or bridge, got %q", opts.Network)
		}
		cfg.Docker.Network = opts.Network
	}

	// Isolated containers can't download, so fetch -hf models on the host
	// and load them as local files.
	hfRepo := opts.HFRepo
	if hfRepo != "" && cfg.Docker.Network != netBridge {
		local, err := prefetchHF(cfg, hfRepo, opts.Verbose)
		if err != nil {
			return loadSpec{}, err
		}
		modelPath, hfRepo = local, ""
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
		HFRepo:       hfRepo,
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
		if opts.Verbose {
			fmt.Printf("Replacing instance '%s'...\n", spec.InstanceName)
		}
		stopInstance(*existing)
		reg.Remove(spec.InstanceName)
	}

	network := cfg.Docker.Network
	if !validNetworkMode(network) {
		return Instance{}, false, fmt.Errorf("invalid docker network mode %q (want none, internal or bridge)", network)
	}
	container := containerName(spec.InstanceName)

	// A model using an llmctl build runs that build's (local) image.
	buildName, buildCommit := "", ""
	if cfg.Docker.Build != "" {
		name, pin := splitBuildRef(cfg.Docker.Build)
		if _, ok := cfg.Builds[name]; !ok {
			return Instance{}, false, fmt.Errorf("build %q is not configured; add it with `llmctl build <url>` (see `llmctl builds`)", name)
		}
		image, err := findBuildImage(name, pin)
		if err != nil {
			return Instance{}, false, err
		}
		commit, unixSocket, err := buildImageInfo(image)
		if err != nil {
			return Instance{}, false, fmt.Errorf("inspecting %s: %w", image, err)
		}
		if network == netNone && !unixSocket {
			return Instance{}, false, fmt.Errorf("build %q (%s) has no unix-socket support in llama-server, so it can't run with docker network \"none\"; load with --network internal or set \"docker_network\": \"internal\" for this model", name, shortCommit(commit))
		}
		cfg.Docker.Image = image
		buildName, buildCommit = name, commit
	}

	// Network none: llama-server listens on a unix socket in a per-instance
	// directory; no TCP port is used.
	backendPort := 0
	sockPath := ""
	if network == netNone {
		dir := socketDir(container)
		removeSocketDir(container)
		if err := os.MkdirAll(dir, 0700); err != nil {
			return Instance{}, false, fmt.Errorf("creating socket dir: %w", err)
		}
		sockPath = filepath.Join(dir, "llama.sock")
	} else {
		backendPort = reg.NextPort()
	}

	displayName := spec.ModelPath
	if spec.HFRepo != "" {
		displayName = spec.HFRepo
	}
	if opts.Verbose {
		where := fmt.Sprintf("backend :%d", backendPort)
		if sockPath != "" {
			where = "no network, unix socket"
		}
		fmt.Printf("Loading %s as '%s' (%s)...\n", shortName(displayName), spec.InstanceName, where)
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
	if sockPath != "" {
		// llama-server binds a unix socket when --host ends in .sock.
		args = append(args, "--host", sockPath)
	} else {
		// Inside the container llama-server must listen on all interfaces;
		// it is only reachable via 127.0.0.1 (bridge) or the internal network.
		args = append(args, "--host", "0.0.0.0", "--port", strconv.Itoa(backendPort))
	}
	args = append(args, "--ctx-size", strconv.Itoa(cfg.CtxSize))
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

	var files []string
	if spec.HFRepo == "" {
		files = append(files, spec.ModelPath)
	}
	if spec.MmprojPath != "" {
		files = append(files, spec.MmprojPath)
	}
	// The HF cache is only mounted (read-write) when the container may
	// download models itself, i.e. on the bridge network.
	hfCache := ""
	if network == netBridge && cfg.Docker.HFCacheDir != "" {
		hfCache = cfg.Docker.HFCacheDir
		// Create it as the current user; otherwise docker creates it as root.
		_ = os.MkdirAll(hfCache, 0755)
	}
	mounts := backendMounts(cfg.ModelsDir, hfCache, files, cfg.ExtraArgs)
	if sockPath != "" {
		mounts = append(mounts, containerMount{Host: filepath.Dir(sockPath)})
	}
	runArgs := dockerRunArgs(dockerRunSpec{
		Docker:     cfg.Docker,
		Container:  container,
		Instance:   spec.InstanceName,
		Port:       backendPort,
		Mounts:     mounts,
		ServerArgs: args,
	})

	if buildName == "" {
		if err := ensureImage(cfg.Docker.Image, opts.Verbose); err != nil {
			return Instance{}, false, fmt.Errorf("error pulling image %s: %w", cfg.Docker.Image, err)
		}
	}
	if network == netInternal {
		if err := ensureInternalNetwork(); err != nil {
			return Instance{}, false, fmt.Errorf("error creating network %s: %w", internalNetwork, err)
		}
	}
	// Clear any leftover container with the same name (e.g. from a crash).
	_, _ = dockerExec("rm", "-f", container)
	if _, err := dockerExec(runArgs...); err != nil {
		_, _ = dockerExec("rm", "-f", container)
		removeSocketDir(container)
		return Instance{}, false, fmt.Errorf("error starting container: %w", err)
	}
	invalidateLiveContainers()

	addr := ""
	switch network {
	case netInternal:
		ip, err := containerIP(container, internalNetwork)
		if err != nil {
			removeContainer(container, instanceLogPath(spec.InstanceName))
			return Instance{}, false, err
		}
		addr = net.JoinHostPort(ip, strconv.Itoa(backendPort))
	case netBridge:
		addr = fmt.Sprintf("127.0.0.1:%d", backendPort)
	}

	isDefault := len(reg.Instances) == 0

	inst := Instance{
		Name:            spec.InstanceName,
		Model:           spec.ModelPath,
		Mmproj:          spec.MmprojPath,
		Container:       container,
		Image:           cfg.Docker.Image,
		Build:           buildName,
		Commit:          buildCommit,
		Network:         network,
		Socket:          sockPath,
		Addr:            addr,
		Port:            backendPort,
		IsDefault:       isDefault,
		CtxSize:         cfg.CtxSize,
		GpuLayers:       cfg.GpuLayers,
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
	target := instanceBackend(inst)
	healthy, exited, exitCode := waitForHealth(target, container, timeout)
	if healthy {
		if opts.Verbose {
			fmt.Printf("✓ '%s' ready (container %s, network %s)\n", spec.InstanceName, container, network)
		}
	} else {
		if exited {
			logPath := instanceLogPath(spec.InstanceName)
			removeContainer(container, logPath)
			removeSocketDir(container)
			reg := loadRegistry()
			reg.Remove(spec.InstanceName)
			if isDefault && len(reg.Instances) > 0 {
				reg.Instances[0].IsDefault = true
			}
			saveRegistry(reg)
			if opts.Verbose {
				fmt.Fprintf(os.Stderr, "Error: '%s' exited (code %d) before becoming healthy. Check: llmctl logs %s\n",
					spec.InstanceName, exitCode, spec.InstanceName)
				printLogTail(logPath, 20)
			}
			return Instance{}, false, fmt.Errorf("%q exited with code %d before becoming healthy", spec.InstanceName, exitCode)
		}
		if opts.Verbose {
			fmt.Printf("⚠ '%s' started (container %s) but not yet healthy. Check: llmctl logs %s\n",
				spec.InstanceName, container, spec.InstanceName)
		}
		return inst, false, nil
	}

	if opts.Verbose && isDefault {
		fmt.Printf("  ★ Set as default model\n")
	}
	if opts.Verbose {
		fmt.Printf("\n  Proxy endpoint: http://%s:%d/v1\n", displayHost(cfg.Host), cfg.Port)
		fmt.Printf("  Model name:     %s\n", spec.InstanceName)
		if sockPath != "" {
			fmt.Printf("  Direct backend: curl --unix-socket %s http://localhost/health\n", sockPath)
		} else {
			fmt.Printf("  Direct backend: %s\n", target)
		}
	}
	return inst, true, nil
}

func cmdLoad(cfg Config, opts loadOptions) {
	opts.WaitTimeout = 60 * time.Second
	opts.Verbose = true
	_, _, err := loadInstance(cfg, opts)
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
		fmt.Printf("Stopping '%s' (container %s)...\n", inst.Name, inst.Container)
	}
	stopInstance(*inst)
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
		if inst.Container != "" {
			fmt.Printf("Stopping '%s' (container %s)...\n", inst.Name, inst.Container)
			stopInstance(inst)
		}
	}
	// Also remove any llmctl-managed containers the registry lost track of.
	if out, err := dockerExec("ps", "-a", "--filter", "label="+containerLabel, "--format", "{{.Names}}"); err == nil {
		for _, name := range strings.Fields(out) {
			fmt.Printf("Removing orphan container %s...\n", name)
			_, _ = dockerExec("stop", "-t", "10", name)
			removeContainer(name, "")
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
	fmt.Printf("  %-3s %-20s %-26s %-8s %-8s %s\n", "", "NAME", "MODEL", "PORT", "CTX", "STATUS")
	fmt.Printf("  %-3s %-20s %-26s %-8s %-8s %s\n", "", "────", "─────", "────", "───", "──────")

	for _, inst := range reg.Instances {
		marker := "  "
		if inst.IsDefault {
			marker = "★ "
		}
		status := "stopped"
		if instanceRunning(inst) {
			status = backendStatus(inst)
		}
		model := shortName(inst.Model)
		if len(model) > 26 {
			model = model[:23] + "..."
		}
		ctxStr := "-"
		if inst.CtxSize > 0 {
			ctxStr = strconv.Itoa(inst.CtxSize)
		}
		portStr := strconv.Itoa(inst.Port)
		if inst.Socket != "" {
			portStr = "socket"
		}
		fmt.Printf("  %s%-20s %-26s %-8s %-8s %s\n",
			marker, inst.Name, model, portStr, ctxStr, status)
		if inst.AutoUnload || inst.EstimatedVramMB > 0 {
			fmt.Printf("    autoswitch: auto_unload=%t estimated_vram_mb=%d\n", inst.AutoUnload, inst.EstimatedVramMB)
		}
		if len(inst.Aliases) > 0 {
			fmt.Printf("    aliases: %s\n", strings.Join(inst.Aliases, ", "))
		}
		if inst.Build != "" {
			fmt.Printf("    build: %s @ %s\n", inst.Build, shortCommit(inst.Commit))
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
	if instanceRunning(*inst) {
		status = backendStatus(*inst)
	}

	fmt.Printf("Instance:   %s\n", inst.Name)
	if len(inst.Aliases) > 0 {
		fmt.Printf("Aliases:    %s\n", strings.Join(inst.Aliases, ", "))
	}
	if inst.IsDefault {
		fmt.Println("Default:    ★ yes")
	}
	fmt.Printf("Status:     %s\n", status)
	fmt.Printf("Container:  %s\n", inst.Container)
	fmt.Printf("Image:      %s\n", inst.Image)
	if inst.Build != "" {
		fmt.Printf("Build:      %s @ %s\n", inst.Build, inst.Commit)
	}
	fmt.Printf("Model:      %s\n", inst.Model)
	if inst.Mmproj != "" {
		fmt.Printf("Mmproj:     %s\n", inst.Mmproj)
	}
	fmt.Printf("Backend:    %s\n", instanceBackend(*inst))
	if inst.Network != "" {
		fmt.Printf("Network:    %s\n", inst.Network)
	}
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
	// Running instance: read straight from the container.
	reg := loadRegistry()
	if inst := findInstanceByNameFuzzy(reg, name); inst != nil && inst.Container != "" {
		if _, _, exists := containerState(inst.Container); exists {
			cmd := exec.Command(dockerBin, "logs", "--tail", "50", inst.Container)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stdout // llama-server logs to stderr
			if cmd.Run() == nil {
				return
			}
		}
	}

	// Stopped instance: logs saved when the container was removed.
	logPath := instanceLogPath(name)
	if _, err := os.Stat(logPath); err != nil {
		entries, _ := os.ReadDir(logDir())
		lower := strings.ToLower(name)
		for _, e := range entries {
			if strings.Contains(strings.ToLower(e.Name()), lower) {
				logPath = filepath.Join(logDir(), e.Name())
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

// f16Re matches an F16 (not BF16) file name, the mmproj llama.cpp prefers.
var f16Re = regexp.MustCompile(`(?i)(^|[^b])f16`)

var splitGGUFRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})\.gguf$`)

// hfSelectFiles picks the repo files to fetch for `-hf repo[:tag]`, mirroring
// llama.cpp: the gguf whose name contains the quant tag (default Q4_K_M,
// falling back to the first gguf when no tag was given), all parts of a split
// model, and an mmproj file if the repo has one. A tag ending in .gguf names
// the file directly.
func hfSelectFiles(files []string, tag string) (model []string, mmproj string, err error) {
	var ggufs, mmprojs []string
	for _, f := range files {
		lower := strings.ToLower(f)
		if !strings.HasSuffix(lower, ".gguf") {
			continue
		}
		if strings.HasPrefix(strings.ToLower(path.Base(f)), "mmproj") {
			mmprojs = append(mmprojs, f)
		} else {
			ggufs = append(ggufs, f)
		}
	}
	sort.Strings(ggufs)
	sort.Strings(mmprojs)
	if len(ggufs) == 0 {
		return nil, "", fmt.Errorf("no .gguf files in repo")
	}

	var chosen string
	if strings.HasSuffix(strings.ToLower(tag), ".gguf") {
		for _, f := range ggufs {
			if f == tag || path.Base(f) == tag {
				chosen = f
				break
			}
		}
		if chosen == "" {
			return nil, "", fmt.Errorf("file %q not found in repo", tag)
		}
	} else {
		want := tag
		if want == "" {
			want = "Q4_K_M"
		}
		want = strings.ToLower(want)
		for _, f := range ggufs {
			if strings.Contains(strings.ToLower(f), want) {
				chosen = f
				break
			}
		}
		if chosen == "" {
			if tag != "" {
				return nil, "", fmt.Errorf("no .gguf matching %q; available: %s", tag, strings.Join(ggufs, ", "))
			}
			chosen = ggufs[0]
		}
	}

	if m := splitGGUFRe.FindStringSubmatchIndex(chosen); m != nil {
		stem := chosen[:m[0]]
		total := chosen[m[4]:m[5]]
		for _, f := range ggufs {
			if sm := splitGGUFRe.FindStringSubmatchIndex(f); sm != nil && f[:sm[0]] == stem && f[sm[4]:sm[5]] == total {
				model = append(model, f)
			}
		}
	} else {
		model = []string{chosen}
	}

	if len(mmprojs) > 0 {
		mmproj = mmprojs[0]
		for _, f := range mmprojs {
			if f16Re.MatchString(path.Base(f)) {
				mmproj = f
				break
			}
		}
	}
	return model, mmproj, nil
}

func hfListFiles(user, repoName string) ([]string, error) {
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/%s", user, repoName)
	resp, err := http.Get(apiURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface API %s: %s", apiURL, resp.Status)
	}
	var repoInfo struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&repoInfo); err != nil {
		return nil, err
	}
	files := make([]string, 0, len(repoInfo.Siblings))
	for _, s := range repoInfo.Siblings {
		files = append(files, s.Filename)
	}
	return files, nil
}

// hfDownloadCmd builds the host-side huggingface_hub download command that
// stores the file in the HF cache layout under models_dir.
func hfDownloadCmd(cfg Config, repoID, file string) (*exec.Cmd, error) {
	var cmd *exec.Cmd
	if p, err := exec.LookPath("hf"); err == nil {
		cmd = exec.Command(p, "download", repoID, file)
	} else if p, err := exec.LookPath("huggingface-cli"); err == nil {
		cmd = exec.Command(p, "download", repoID, file)
	} else if p, err := exec.LookPath("python3"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repoID, file)
	} else if p, err := exec.LookPath("python"); err == nil {
		cmd = exec.Command(p, "-m", "huggingface_hub", "download", repoID, file)
	} else {
		return nil, fmt.Errorf("huggingface_hub not installed (pip install huggingface_hub)")
	}
	cmd.Env = append(os.Environ(), "HF_HOME="+cfg.ModelsDir)
	return cmd, nil
}

// prefetchHF downloads the files `-hf repo[:tag]` needs into models_dir on
// the host and returns the local model path (first part for split models).
func prefetchHF(cfg Config, hfRepo string, verbose bool) (string, error) {
	user, repoName, repoID, tag, err := parsePullTarget(hfRepo)
	if err != nil {
		return "", fmt.Errorf("invalid -hf repo %q", hfRepo)
	}
	cacheKey := func(f string) string { return "models--" + user + "--" + repoName + "/" + f }

	// A tag naming a file that is already cached needs no network.
	if strings.HasSuffix(strings.ToLower(tag), ".gguf") {
		if p := findInHFCache(cfg.ModelsDir, cacheKey(tag)); p != "" {
			return p, nil
		}
	}

	files, err := hfListFiles(user, repoName)
	if err != nil {
		return "", fmt.Errorf("listing %s: %w", repoID, err)
	}
	modelFiles, mmproj, err := hfSelectFiles(files, tag)
	if err != nil {
		return "", fmt.Errorf("%s: %w", repoID, err)
	}
	toFetch := append([]string{}, modelFiles...)
	if mmproj != "" {
		toFetch = append(toFetch, mmproj)
	}
	for _, f := range toFetch {
		if findInHFCache(cfg.ModelsDir, cacheKey(f)) != "" {
			continue
		}
		if verbose {
			fmt.Printf("Downloading %s:%s on the host (containers have no network)...\n", repoID, f)
		}
		cmd, err := hfDownloadCmd(cfg, repoID, f)
		if err != nil {
			return "", err
		}
		if verbose {
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			err = cmd.Run()
		} else if out, cerr := cmd.CombinedOutput(); cerr != nil {
			err = fmt.Errorf("%v: %s", cerr, strings.TrimSpace(string(out)))
		}
		if err != nil {
			return "", fmt.Errorf("downloading %s:%s: %w", repoID, f, err)
		}
	}
	p := findInHFCache(cfg.ModelsDir, cacheKey(modelFiles[0]))
	if p == "" {
		return "", fmt.Errorf("downloaded %s:%s but could not find it under %s", repoID, modelFiles[0], cfg.ModelsDir)
	}
	return p, nil
}

func cmdPull(cfg Config, repo string) {
	user, repoName, repoID, specificFile, err := parsePullTarget(repo)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Usage: llmctl pull <user/repo>[:<file>]")
		os.Exit(1)
	}

	if specificFile == "" {
		files, err := hfListFiles(user, repoName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		var ggufFiles []string
		for _, f := range files {
			if strings.HasSuffix(strings.ToLower(f), ".gguf") {
				ggufFiles = append(ggufFiles, f)
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

	cmd, err := hfDownloadCmd(cfg, repoID, specificFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

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
		if inst.Model == modelPath && instanceRunning(inst) {
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
	fmt.Println("  Docker:")
	fmt.Printf("    bin:          %s\n", cfg.Docker.Bin)
	fmt.Printf("    image:        %s\n", cfg.Docker.Image)
	if cfg.Docker.Build != "" {
		fmt.Printf("    build:        %s\n", cfg.Docker.Build)
	}
	if cfg.Docker.Entrypoint != "" {
		fmt.Printf("    entrypoint:   %s\n", cfg.Docker.Entrypoint)
	}
	fmt.Printf("    gpu:          %s\n", cfg.Docker.GPU)
	fmt.Printf("    network:      %s\n", cfg.Docker.Network)
	fmt.Printf("    read_only:    %t\n", cfg.Docker.readOnly())
	fmt.Printf("    pids_limit:   %d\n", cfg.Docker.PidsLimit)
	fmt.Printf("    hf_cache_dir: %s\n", cfg.Docker.HFCacheDir)
	if len(cfg.Docker.Args) > 0 {
		fmt.Printf("    args:         %s\n", strings.Join(cfg.Docker.Args, " "))
	}
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
	if len(cfg.Builds) > 0 {
		fmt.Println("  Builds:")
		names := make([]string, 0, len(cfg.Builds))
		for name := range cfg.Builds {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			bc := cfg.Builds[name]
			src := bc.Path
			if bc.Repo != "" {
				src = bc.Repo
				if bc.Ref != "" {
					src += " @ " + bc.Ref
				}
			}
			fmt.Printf("    %s → %s\n", name, src)
		}
	}
	if len(cfg.Models) > 0 {
		fmt.Println("  Model overrides:")
		for name, mc := range cfg.Models {
			fmt.Printf("    [%s]\n", name)
			if mc.DockerImage != nil {
				fmt.Printf("      docker_image: %s\n", *mc.DockerImage)
			}
			if mc.DockerEntrypoint != nil {
				fmt.Printf("      docker_entrypoint: %s\n", *mc.DockerEntrypoint)
			}
			if mc.DockerArgs != nil {
				fmt.Printf("      docker_args: %s\n", strings.Join(mc.DockerArgs, " "))
			}
			if mc.DockerNetwork != nil {
				fmt.Printf("      docker_network: %s\n", *mc.DockerNetwork)
			}
			if mc.DockerBuild != nil {
				fmt.Printf("      docker_build: %s\n", *mc.DockerBuild)
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
	case "docker.bin":
		cfg.Docker.Bin = value
	case "docker.image":
		cfg.Docker.Image = value
	case "docker.entrypoint":
		cfg.Docker.Entrypoint = value
	case "docker.gpu":
		switch value {
		case "nvidia", "jetson", "none":
		default:
			fmt.Fprintln(os.Stderr, "Error: docker.gpu must be nvidia, jetson or none")
			os.Exit(1)
		}
		cfg.Docker.GPU = value
	case "docker.hf_cache_dir":
		cfg.Docker.HFCacheDir = value
	case "docker.network":
		if !validNetworkMode(value) {
			fmt.Fprintln(os.Stderr, "Error: docker.network must be none, internal or bridge")
			os.Exit(1)
		}
		cfg.Docker.Network = value
	case "docker.read_only":
		v, err := strconv.ParseBool(value)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error: must be true or false")
			os.Exit(1)
		}
		cfg.Docker.ReadOnly = &v
	case "docker.pids_limit":
		n, err := strconv.Atoi(value)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error: must be a number (<= 0 disables)")
			os.Exit(1)
		}
		cfg.Docker.PidsLimit = n
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

Custom llama.cpp builds:
  build <url|owner/repo|path> Add a build for a fork/branch/PR/commit/local checkout and build it
  build <name>... [--ref R] [--force]  (Re)build configured builds; --ref is a one-off override
  build add <name> <source>   Add a build under a chosen name without building
  build rm <name>             Remove a build, its images and fetched source
  build --all                 Build every configured build (skips ones already up to date)
  builds                      List builds, their latest commit and which models use them
    flags: --ref R  --backend cuda|jetson  --cuda-arch A  --cmake-arg ARG (repeatable)  --dockerfile F

  load <model> --build <name>[@commit]   Run a model with a build (instance "<model>-<name>")
  load <model> --network internal        For builds without unix-socket support

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
  llmctl build https://github.com/someone/llama.cpp/tree/branch
  llmctl load mistral --build someone             # try a fork next to the regular one
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
	dockerBin = cfg.Docker.Bin

	switch os.Args[1] {
	case "list", "ls":
		cmdList(cfg)
	case "load", "run":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: llmctl load <model> [-hf <huggingface_repo>] [--name NAME] [--mmproj <path>] [--build <build>[@commit]] [--network none|internal|bridge]")
			os.Exit(1)
		}
		name, args := extractFlag(os.Args[3:], "--name")
		hf, args := extractFlag(args, "-hf")
		mmproj, args := extractFlag(args, "--mmproj")
		build, args := extractFlag(args, "--build")
		network, _ := extractFlag(args, "--network")
		cmdLoad(cfg, loadOptions{
			ModelName:    os.Args[2],
			InstanceName: name,
			HFRepo:       hf,
			MmprojArg:    mmproj,
			Build:        build,
			Network:      network,
		})
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
	case "build":
		cmdBuildCLI(cfg, os.Args[2:])
	case "builds":
		cmdBuilds(cfg)
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

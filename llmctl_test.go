package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func intPtr(n int) *int {
	return &n
}

func testConfig(modelsDir string) Config {
	return Config{
		ModelsDir: modelsDir,
		Aliases:   map[string]string{},
		Models:    map[string]ModelConfig{},
	}
}

func writeHFModel(t *testing.T, root, user, repo, sha, file string) string {
	t.Helper()
	path := filepath.Join(root, "hub", "models--"+user+"--"+repo, "snapshots", sha, file)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("gguf"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeLocalModel(t *testing.T, path string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("gguf"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeSizedModel(t *testing.T, path string, size int64) string {
	t.Helper()
	path = writeLocalModel(t, path)
	if err := os.Truncate(path, size); err != nil {
		t.Fatal(err)
	}
	return path
}

type responseRecorder struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (r *responseRecorder) Header() http.Header {
	if r.header == nil {
		r.header = http.Header{}
	}
	return r.header
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	return r.body.Write(b)
}

func (r *responseRecorder) WriteHeader(status int) {
	r.status = status
}

func TestResolveModelDistinguishesSameHFFileNameByRepo(t *testing.T) {
	root := t.TempDir()
	cfg := testConfig(root)
	file := "Qwen3.6-27B-Q4_K_M.gguf"
	mtp := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", file)
	base := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-GGUF", "sha-base", file)

	got, err := resolveModel(cfg, "unsloth/Qwen3.6-27B-MTP-GGUF:"+file)
	if err != nil {
		t.Fatal(err)
	}
	if got != mtp {
		t.Fatalf("MTP ref resolved to %q, want %q", got, mtp)
	}

	got, err = resolveModel(cfg, "unsloth/Qwen3.6-27B-GGUF:"+file)
	if err != nil {
		t.Fatal(err)
	}
	if got != base {
		t.Fatalf("base ref resolved to %q, want %q", got, base)
	}
}

func TestValidateExtraArgsSuggestsLlamaCppPenaltyFlags(t *testing.T) {
	err := validateExtraArgs([]string{"--temp", "0.6", "presence_penalty", "0.0"})
	if err == nil {
		t.Fatal("expected validation error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "presence_penalty") || !strings.Contains(msg, "--presence-penalty") {
		t.Fatalf("validation error = %q, want presence penalty suggestion", msg)
	}

	err = validateExtraArgs([]string{"repetition_penalty", "1.0"})
	if err == nil {
		t.Fatal("expected validation error")
	}
	if !strings.Contains(err.Error(), "--repeat-penalty") {
		t.Fatalf("validation error = %q, want repeat penalty suggestion", err)
	}

	err = validateExtraArgs([]string{"--repetition-penalty", "1.0"})
	if err == nil {
		t.Fatal("expected validation error")
	}
	if !strings.Contains(err.Error(), "--repeat-penalty") {
		t.Fatalf("validation error = %q, want repeat penalty suggestion", err)
	}
}

func TestValidateExtraArgsAcceptsDashedPenaltyFlags(t *testing.T) {
	err := validateExtraArgs([]string{
		"--presence-penalty", "0.0",
		"--repeat-penalty", "1.0",
	})
	if err != nil {
		t.Fatalf("valid extra args rejected: %v", err)
	}
}

func TestResolveModelReportsAmbiguousDuplicateBasename(t *testing.T) {
	root := t.TempDir()
	cfg := testConfig(root)
	file := "Qwen3.6-27B-Q4_K_M.gguf"
	writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", file)
	writeHFModel(t, root, "unsloth", "Qwen3.6-27B-GGUF", "sha-base", file)

	_, err := resolveModel(cfg, "Qwen3.6-27B-Q4_K_M")
	if err == nil {
		t.Fatal("expected ambiguity error")
	}
	msg := err.Error()
	for _, want := range []string{
		"ambiguous model",
		"unsloth/Qwen3.6-27B-MTP-GGUF:" + file,
		"unsloth/Qwen3.6-27B-GGUF:" + file,
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("ambiguity error %q does not contain %q", msg, want)
		}
	}
}

func TestResolveRepoOnlyRequiresSingleCachedGGUF(t *testing.T) {
	root := t.TempDir()
	cfg := testConfig(root)
	writeHFModel(t, root, "unsloth", "One-GGUF", "sha-one", "only.gguf")
	writeHFModel(t, root, "unsloth", "One-GGUF", "sha-old", "only.gguf")

	got, err := resolveModel(cfg, "unsloth/One-GGUF")
	if err != nil {
		t.Fatal(err)
	}
	if ref := modelRef(root, got); ref != "unsloth/One-GGUF:only.gguf" {
		t.Fatalf("repo-only ref resolved to %q (%s), want unsloth/One-GGUF:only.gguf", got, ref)
	}

	writeHFModel(t, root, "unsloth", "Many-GGUF", "sha-many", "a.gguf")
	writeHFModel(t, root, "unsloth", "Many-GGUF", "sha-many", "b.gguf")
	_, err = resolveModel(cfg, "unsloth/Many-GGUF")
	if err == nil || !strings.Contains(err.Error(), "ambiguous model") {
		t.Fatalf("expected ambiguous repo-only error, got %v", err)
	}
}

func TestModelRefUsesCanonicalHFRepoAndFile(t *testing.T) {
	root := t.TempDir()
	model := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", "Qwen3.6-27B-Q4_K_M.gguf")

	got := modelRef(root, model)
	want := "unsloth/Qwen3.6-27B-MTP-GGUF:Qwen3.6-27B-Q4_K_M.gguf"
	if got != want {
		t.Fatalf("modelRef() = %q, want %q", got, want)
	}
}

func TestLoadedModelsByPathDoesNotCollapseBasenames(t *testing.T) {
	root := t.TempDir()
	file := "Qwen3.6-27B-Q4_K_M.gguf"
	loadedPath := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", file)
	otherPath := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-GGUF", "sha-base", file)

	loaded := loadedModelsByPath(Registry{Instances: []Instance{{Name: "mtp", Model: loadedPath}}})
	if got := loaded[filepath.Clean(loadedPath)]; got != "mtp" {
		t.Fatalf("loaded path marker = %q, want mtp", got)
	}
	if got := loaded[filepath.Clean(otherPath)]; got != "" {
		t.Fatalf("unloaded duplicate basename was marked as %q", got)
	}
}

func TestDeriveInstanceNameUsesExactHFFileName(t *testing.T) {
	got := deriveInstanceName("unsloth/Qwen3.6-27B-MTP-GGUF:Qwen3.6-27B-Q4_K_M.gguf")
	want := "qwen3.6-27b-q4_k_m"
	if got != want {
		t.Fatalf("deriveInstanceName() = %q, want %q", got, want)
	}
}

func TestModelConfigKeyUsesAliasMatchingResolvedBasename(t *testing.T) {
	root := t.TempDir()
	file := "Qwen3.6-27B-Q4_K_M.gguf"
	model := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", file)
	writeHFModel(t, root, "unsloth", "Qwen3.6-27B-GGUF", "sha-base", file)
	cfg := testConfig(root)
	cfg.CtxSize = 8192
	cfg.Aliases = map[string]string{
		"qwen3627b_code": "Qwen3.6-27B-Q4_K_M",
	}
	cfg.Models = map[string]ModelConfig{
		"qwen3627b_code": {
			CtxSize: intPtr(130000),
		},
	}

	key := modelConfigKey(cfg, "unsloth/Qwen3.6-27B-MTP-GGUF:"+file, "qwen3.6-27b-q4_k_m", model)
	if key != "qwen3627b_code" {
		t.Fatalf("modelConfigKey() = %q, want qwen3627b_code", key)
	}
	if got := configForModel(cfg, key).CtxSize; got != 130000 {
		t.Fatalf("ctx_size = %d, want 130000", got)
	}
}

func TestModelConfigKeyPrefersExactCanonicalRef(t *testing.T) {
	root := t.TempDir()
	file := "Qwen3.6-27B-Q4_K_M.gguf"
	model := writeHFModel(t, root, "unsloth", "Qwen3.6-27B-MTP-GGUF", "sha-mtp", file)
	cfg := testConfig(root)
	cfg.Models = map[string]ModelConfig{
		"unsloth/Qwen3.6-27B-MTP-GGUF:" + file: {
			CtxSize: intPtr(130000),
		},
	}

	key := modelConfigKey(cfg, "unsloth/Qwen3.6-27B-MTP-GGUF:"+file, "qwen3.6-27b-q4_k_m", model)
	if key != "unsloth/Qwen3.6-27B-MTP-GGUF:"+file {
		t.Fatalf("modelConfigKey() = %q, want canonical ref", key)
	}
}

func TestResolveMmprojPrefersModelDirectory(t *testing.T) {
	root := t.TempDir()
	cfg := testConfig(root)
	model := writeHFModel(t, root, "unsloth", "Vision-GGUF", "sha", "model.gguf")
	mmproj := writeLocalModel(t, filepath.Join(filepath.Dir(model), "mmproj-F16.gguf"))
	writeHFModel(t, root, "unsloth", "Other-GGUF", "sha-other", "mmproj-F16.gguf")

	got, err := resolveMmproj(cfg, model, "mmproj-F16.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if got != mmproj {
		t.Fatalf("resolveMmproj() = %q, want %q", got, mmproj)
	}
}

func TestResolveMmprojErrorsOnAmbiguousGlobalName(t *testing.T) {
	root := t.TempDir()
	cfg := testConfig(root)
	model := writeHFModel(t, root, "unsloth", "Text-GGUF", "sha", "model.gguf")
	writeHFModel(t, root, "unsloth", "Vision-A-GGUF", "sha-a", "mmproj-F16.gguf")
	writeHFModel(t, root, "unsloth", "Vision-B-GGUF", "sha-b", "mmproj-F16.gguf")

	got, err := resolveMmproj(cfg, model, "mmproj-F16.gguf")
	if err == nil {
		t.Fatalf("resolveMmproj() = %q, want ambiguity error", got)
	}
	msg := err.Error()
	if !strings.Contains(msg, "mmproj") || !strings.Contains(msg, "ambiguous model") {
		t.Fatalf("resolveMmproj() error = %q, want mmproj ambiguity", msg)
	}
}

func TestParsePullTargetRepoOnly(t *testing.T) {
	user, repoName, repoID, file, err := parsePullTarget("unsloth/Qwen3.6-27B-GGUF")
	if err != nil {
		t.Fatal(err)
	}
	if user != "unsloth" || repoName != "Qwen3.6-27B-GGUF" || repoID != "unsloth/Qwen3.6-27B-GGUF" || file != "" {
		t.Fatalf("parsePullTarget() = %q %q %q %q", user, repoName, repoID, file)
	}
}

func TestParsePullTargetSlashFile(t *testing.T) {
	user, repoName, repoID, file, err := parsePullTarget("unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if user != "unsloth" || repoName != "Qwen3.6-27B-GGUF" || repoID != "unsloth/Qwen3.6-27B-GGUF" || file != "mmproj-F16.gguf" {
		t.Fatalf("parsePullTarget() = %q %q %q %q", user, repoName, repoID, file)
	}
}

func TestParsePullTargetColonFile(t *testing.T) {
	user, repoName, repoID, file, err := parsePullTarget("https://huggingface.co/unsloth/Qwen3.6-27B-GGUF:mmproj-F16.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if user != "unsloth" || repoName != "Qwen3.6-27B-GGUF" || repoID != "unsloth/Qwen3.6-27B-GGUF" || file != "mmproj-F16.gguf" {
		t.Fatalf("parsePullTarget() = %q %q %q %q", user, repoName, repoID, file)
	}
}

func TestParsePullTargetNestedFile(t *testing.T) {
	user, repoName, repoID, file, err := parsePullTarget("unsloth/Repo-GGUF/subdir/model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if user != "unsloth" || repoName != "Repo-GGUF" || repoID != "unsloth/Repo-GGUF" || file != "subdir/model.gguf" {
		t.Fatalf("parsePullTarget() = %q %q %q %q", user, repoName, repoID, file)
	}
}

func TestAutoswitchConfigParsing(t *testing.T) {
	var cfg Config
	data := []byte(`{
		"autoswitch": {"enabled": true, "total_vram_mb": 24576, "startup_timeout_sec": 90},
		"models": {
			"qwen": {"auto_load": true, "auto_unload": true, "vram_mb": 18000}
		}
	}`)
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.Autoswitch.Enabled || cfg.Autoswitch.TotalVramMB != 24576 || cfg.Autoswitch.StartupTimeoutSec != 90 {
		t.Fatalf("autoswitch config = %+v", cfg.Autoswitch)
	}
	mc := cfg.Models["qwen"]
	if mc.AutoLoad && mc.AutoUnload && mc.VramMB == 18000 {
		// good
	} else {
		t.Fatalf("model autoswitch config = %+v", mc)
	}
}

func TestValidateAutoswitchConfigRequiresVramMB(t *testing.T) {
	cfg := Config{Autoswitch: AutoswitchConfig{Enabled: true}}
	cfg.Models = map[string]ModelConfig{
		"no-vram":   {AutoLoad: true, AutoUnload: true},
		"with-vram": {AutoLoad: true, AutoUnload: true, VramMB: 12000},
	}
	err := validateAutoswitchConfig(cfg)
	if err == nil {
		t.Fatal("expected error for model without vram_mb")
	}

	mc := cfg.Models["no-vram"]
	mc.VramMB = 8000
	cfg.Models["no-vram"] = mc
	err = validateAutoswitchConfig(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestExplicitModelWithAutoswitchErrorDoesNotFallbackToDefault(t *testing.T) {
	if shouldFallbackToDefault("qwen3627b_code", os.ErrNotExist) {
		t.Fatal("explicit model with autoswitch error should not fall back to default")
	}
}

func TestMissingModelCanFallbackToDefault(t *testing.T) {
	if !shouldFallbackToDefault("", nil) {
		t.Fatal("request without model should fall back to default")
	}
	if !shouldFallbackToDefault("loaded-model", nil) {
		t.Fatal("request with no autoswitch error may use existing default fallback behavior")
	}
}

func TestHandleListModelsIncludesAutoLoadModels(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cfg := testConfig(t.TempDir())
	cfg.Models = map[string]ModelConfig{
		"autoload": {AutoLoad: true},
		"manual":   {},
	}

	rec := &responseRecorder{}
	handleListModels(rec, cfg)

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range body.Data {
		ids = append(ids, m.ID)
	}
	if strings.Join(ids, ",") != "autoload" {
		t.Fatalf("/v1/models ids = %v, want [autoload]", ids)
	}
}

func TestSystemdUnitFile(t *testing.T) {
	data, err := os.ReadFile("llmctl.service")
	if err != nil {
		t.Skipf("llmctl.service not found: %v", err)
	}
	content := string(data)
	for _, want := range []string{"[Unit]", "[Service]", "[Install]", "After=network-online.target docker.service", "ExecStart=/usr/local/bin/llmctl proxy", "Restart=always", "WantedBy=multi-user.target"} {
		if !strings.Contains(content, want) {
			t.Errorf("unit file missing: %s", want)
		}
	}
}

func TestHandleUIModelsListsOnlyConfiguredModels(t *testing.T) {
	tmp := t.TempDir()
	cfg := testConfig(tmp)
	cfg.Aliases = map[string]string{
		"qwen27b":      "Qwen3.5-27B-GGUF/UD-Q4_K_XL.gguf",
		"qwen27b_code": "Qwen3.5-27B-GGUF/UD-Q4_K_XL.gguf", // same target, distinct alias
	}
	cfg.Models = map[string]ModelConfig{
		"qwen27b":      {VramMB: 18000},
		"qwen27b_code": {VramMB: 16000},
	}

	// A stray model file on disk that is NOT referenced by any alias/model
	// in config must NOT appear in the UI list.
	writeHFModel(t, tmp, "test", "stray-model", "abc123", "stray-model.Q4.gguf")

	rec := &responseRecorder{}
	handleUIModels(rec, cfg)

	var models []struct {
		Name    string `json:"name"`
		Running bool   `json:"running"`
		Vram    int    `json:"vram"`
	}
	if err := json.Unmarshal(rec.body.Bytes(), &models); err != nil {
		t.Fatal(err)
	}

	got := make(map[string]int) // name -> vram
	for _, m := range models {
		got[m.Name] = m.Vram
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 configured models, got %d: %v", len(got), got)
	}
	for name := range cfg.Aliases {
		if _, ok := got[name]; !ok {
			t.Errorf("expected configured alias %q in UI list, got: %v", name, got)
		}
	}
	if got["qwen27b"] != 18000 {
		t.Errorf("qwen27b vram = %d, want 18000", got["qwen27b"])
	}
	if got["qwen27b_code"] != 16000 {
		t.Errorf("qwen27b_code vram = %d, want 16000", got["qwen27b_code"])
	}

	// The stray on-disk model must never leak in.
	for name := range got {
		if strings.Contains(name, "stray-model") {
			t.Errorf("stray disk model leaked into UI list: %q", name)
		}
	}
}

// ── Docker backend ──────────────────────────────────────────────────────────

func stubDocker(t *testing.T, fn func(args ...string) (string, error)) *[][]string {
	t.Helper()
	var calls [][]string
	orig := dockerExec
	dockerExec = func(args ...string) (string, error) {
		calls = append(calls, args)
		return fn(args...)
	}
	invalidateLiveContainers()
	t.Cleanup(func() {
		dockerExec = orig
		invalidateLiveContainers()
	})
	return &calls
}

func argValue(args []string, flag string) []string {
	var vals []string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			vals = append(vals, args[i+1])
		}
	}
	return vals
}

func TestContainerNameSanitizes(t *testing.T) {
	if got := containerName("qwen3.5-27b"); got != "llmctl-qwen3.5-27b" {
		t.Fatalf("got %q", got)
	}
	if got := containerName("org/model:Q4 K"); got != "llmctl-org-model-Q4-K" {
		t.Fatalf("got %q", got)
	}
}

func TestDockerRunArgsNvidia(t *testing.T) {
	args := dockerRunArgs(dockerRunSpec{
		Docker:     DockerConfig{Image: "img:tag", GPU: "nvidia", Network: netBridge, HFCacheDir: "/cache", Args: []string{"--shm-size", "1g"}},
		Container:  "llmctl-chat",
		Instance:   "chat",
		Port:       9100,
		Mounts:     []containerMount{{Host: "/models", ReadOnly: true}, {Host: "/cache"}},
		ServerArgs: []string{"--model", "/models/a.gguf", "--port", "9100"},
	})
	joined := strings.Join(args, " ")
	if args[0] != "run" || args[1] != "-d" {
		t.Fatalf("expected detached run, got %v", args[:2])
	}
	for _, want := range []string{
		"--name llmctl-chat",
		"--label llmctl.instance=chat",
		"--cap-drop ALL",
		"--security-opt no-new-privileges",
		"--read-only --tmpfs /tmp",
		"-p 127.0.0.1:9100:9100",
		"--gpus all",
		"-v /models:/models:ro",
		"-v /cache:/cache ",
		"-e LLAMA_CACHE=/cache",
		"--shm-size 1g img:tag --model /models/a.gguf --port 9100",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in: %s", want, joined)
		}
	}
	if strings.Contains(joined, "--rm") {
		t.Errorf("containers must not use --rm (logs are kept until llmctl removes them)")
	}
	if strings.Contains(joined, "--entrypoint") || strings.Contains(joined, "--runtime") {
		t.Errorf("unexpected flags: %s", joined)
	}
}

func TestDockerRunArgsJetsonAndEntrypoint(t *testing.T) {
	args := dockerRunArgs(dockerRunSpec{
		Docker:    DockerConfig{Image: "jetson:img", GPU: "jetson", Entrypoint: "llama-server"},
		Container: "llmctl-x", Instance: "x", Port: 9101,
	})
	if got := argValue(args, "--runtime"); len(got) != 1 || got[0] != "nvidia" {
		t.Fatalf("expected --runtime nvidia, got %v", args)
	}
	if got := argValue(args, "--entrypoint"); len(got) != 1 || got[0] != "llama-server" {
		t.Fatalf("expected --entrypoint llama-server, got %v", args)
	}
	if len(argValue(args, "--gpus")) != 0 {
		t.Fatalf("jetson must not use --gpus: %v", args)
	}
	if args[len(args)-1] != "jetson:img" {
		t.Fatalf("image must be last when no server args: %v", args)
	}
}

func TestDockerRunArgsNoGPU(t *testing.T) {
	args := dockerRunArgs(dockerRunSpec{Docker: DockerConfig{Image: "cpu", GPU: "none"}, Container: "c", Instance: "i", Port: 1})
	if len(argValue(args, "--gpus"))+len(argValue(args, "--runtime")) != 0 {
		t.Fatalf("expected no GPU flags: %v", args)
	}
}

func TestBackendMountsCoversSymlinkTargetsAndExtraArgs(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	other := filepath.Join(root, "other")
	blobs := filepath.Join(root, "store", "blobs")
	tmpl := filepath.Join(root, "templates")
	for _, d := range []string{models, other, blobs, tmpl} {
		if err := os.MkdirAll(d, 0755); err != nil {
			t.Fatal(err)
		}
	}
	inModels := filepath.Join(models, "a.gguf")
	if err := os.WriteFile(inModels, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(blobs, "abc")
	if err := os.WriteFile(blob, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(other, "b.gguf")
	if err := os.Symlink(blob, link); err != nil {
		t.Fatal(err)
	}
	tmplFile := filepath.Join(tmpl, "chat.jinja")
	if err := os.WriteFile(tmplFile, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	slots := filepath.Join(root, "slots")
	if err := os.MkdirAll(slots, 0755); err != nil {
		t.Fatal(err)
	}
	mounts := backendMounts(models, filepath.Join(root, "cache"),
		[]string{inModels, link, "relative.gguf"},
		[]string{"--chat-template-file", tmplFile, "--temp", "0.7", "--foo", "/nonexistent-dir/x", "--slot-save-path", slots})

	got := map[string]bool{}
	for _, m := range mounts {
		got[m.Host] = m.ReadOnly
	}
	want := map[string]bool{
		models:                       true,
		filepath.Join(root, "cache"): false,
		other:                        true,
		blobs:                        true,
		tmpl:                         true, // read-only: not a writable flag
		slots:                        false,
	}
	// t.TempDir may itself be behind a symlink (e.g. macOS /var → /private/var).
	if real, _ := filepath.EvalSymlinks(blobs); real != blobs {
		delete(want, blobs)
		want[real] = true
	}
	for dir, ro := range want {
		gotRO, ok := got[dir]
		if !ok {
			t.Errorf("missing mount %s (got %v)", dir, mounts)
			continue
		}
		if gotRO != ro {
			t.Errorf("mount %s read-only=%t, want %t", dir, gotRO, ro)
		}
	}
	if len(mounts) != len(want) {
		t.Errorf("unexpected extra mounts: %v", mounts)
	}
}

func TestConfigForModelDockerOverrides(t *testing.T) {
	img := "custom:img"
	ep := "llama-server"
	cfg := defaultConfig()
	cfg.Docker.Args = []string{"--shm-size", "1g"}
	cfg.Models = map[string]ModelConfig{
		"m": {DockerImage: &img, DockerEntrypoint: &ep, DockerArgs: []string{"--ipc", "host"}},
	}
	got := configForModel(cfg, "m")
	if got.Docker.Image != img || got.Docker.Entrypoint != ep {
		t.Fatalf("overrides not applied: %+v", got.Docker)
	}
	if strings.Join(got.Docker.Args, " ") != "--shm-size 1g --ipc host" {
		t.Fatalf("docker args not appended: %v", got.Docker.Args)
	}
	if strings.Join(cfg.Docker.Args, " ") != "--shm-size 1g" {
		t.Fatalf("global docker args mutated: %v", cfg.Docker.Args)
	}
}

func TestLoadConfigFillsDockerDefaults(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLAMA_CACHE", "")
	if err := os.WriteFile(filepath.Join(home, configFile), []byte(`{"docker":{"image":"my:img"}}`), 0644); err != nil {
		t.Fatal(err)
	}
	cfg := loadConfig()
	if cfg.Docker.Image != "my:img" {
		t.Fatalf("image = %q", cfg.Docker.Image)
	}
	if cfg.Docker.Bin != "docker" || cfg.Docker.GPU != "nvidia" {
		t.Fatalf("defaults not filled: %+v", cfg.Docker)
	}
	if cfg.Docker.HFCacheDir != filepath.Join(home, ".cache", "llama.cpp") {
		t.Fatalf("hf_cache_dir = %q", cfg.Docker.HFCacheDir)
	}
}

func TestCleanDeadRemovesExitedContainers(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	calls := stubDocker(t, func(args ...string) (string, error) {
		if args[0] == "ps" {
			return "llmctl-alive\n", nil
		}
		return "", nil
	})
	reg := Registry{Instances: []Instance{
		{Name: "alive", Container: "llmctl-alive"},
		{Name: "dead", Container: "llmctl-dead"},
		{Name: "legacy"}, // pre-docker registry entry
	}}
	reg.CleanDead()
	if len(reg.Instances) != 1 || reg.Instances[0].Name != "alive" {
		t.Fatalf("instances = %+v", reg.Instances)
	}
	removed := false
	for _, c := range *calls {
		if c[0] == "rm" && c[len(c)-1] == "llmctl-dead" {
			removed = true
		}
		if c[0] == "rm" && c[len(c)-1] == "llmctl-alive" {
			t.Fatalf("live container removed")
		}
	}
	if !removed {
		t.Fatalf("dead container not removed: %v", *calls)
	}
}

func TestCleanDeadKeepsRegistryWhenDockerUnreachable(t *testing.T) {
	stubDocker(t, func(args ...string) (string, error) {
		return "", fmt.Errorf("cannot connect to the Docker daemon")
	})
	reg := Registry{Instances: []Instance{{Name: "a", Container: "llmctl-a"}}}
	reg.CleanDead()
	if len(reg.Instances) != 1 {
		t.Fatalf("registry wiped when docker unreachable: %+v", reg.Instances)
	}
}

func TestRegistryDefaultFallsBackToLiveContainer(t *testing.T) {
	stubDocker(t, func(args ...string) (string, error) { return "llmctl-b\n", nil })
	reg := Registry{Instances: []Instance{
		{Name: "a", Container: "llmctl-a"},
		{Name: "b", Container: "llmctl-b"},
	}}
	if def := reg.Default(); def == nil || def.Name != "b" {
		t.Fatalf("default = %+v", def)
	}
}

func TestDockerRunArgsNetworkModes(t *testing.T) {
	base := dockerRunSpec{Container: "c", Instance: "i", Port: 9100, ServerArgs: []string{"--host", "/run/x.sock"}}

	base.Docker = DockerConfig{Image: "img", Network: netNone, HFCacheDir: "/cache", PidsLimit: 512}
	args := dockerRunArgs(base)
	if got := argValue(args, "--network"); len(got) != 1 || got[0] != "none" {
		t.Fatalf("none: expected --network none, got %v", args)
	}
	if len(argValue(args, "-p")) != 0 {
		t.Fatalf("none: must not publish ports: %v", args)
	}
	for _, e := range argValue(args, "-e") {
		if strings.HasPrefix(e, "LLAMA_CACHE=") {
			t.Fatalf("none: HF cache env must not be set: %v", args)
		}
	}
	if got := argValue(args, "--pids-limit"); len(got) != 1 || got[0] != "512" {
		t.Fatalf("expected --pids-limit 512: %v", args)
	}

	base.Docker.Network = netInternal
	args = dockerRunArgs(base)
	if got := argValue(args, "--network"); len(got) != 1 || got[0] != internalNetwork {
		t.Fatalf("internal: expected --network %s, got %v", internalNetwork, args)
	}
	if len(argValue(args, "-p")) != 0 {
		t.Fatalf("internal: must not publish ports: %v", args)
	}

	base.Docker.Network = netBridge
	args = dockerRunArgs(base)
	if len(argValue(args, "--network")) != 0 {
		t.Fatalf("bridge: no --network flag expected: %v", args)
	}
	if got := argValue(args, "-p"); len(got) != 1 || got[0] != "127.0.0.1:9100:9100" {
		t.Fatalf("bridge: expected loopback publish: %v", args)
	}
}

func TestDockerRunArgsHardeningToggles(t *testing.T) {
	off := false
	args := dockerRunArgs(dockerRunSpec{Docker: DockerConfig{Image: "img", Network: netNone, ReadOnly: &off, PidsLimit: 0}, Container: "c", Instance: "i"})
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--read-only") || strings.Contains(joined, "--pids-limit") {
		t.Fatalf("read_only=false / pids_limit=0 should drop flags: %s", joined)
	}
	if !strings.Contains(joined, "--cap-drop ALL") || !strings.Contains(joined, "--security-opt no-new-privileges") {
		t.Fatalf("cap-drop/no-new-privileges must always be set: %s", joined)
	}
}

func TestLoadConfigDefaultsToIsolatedHardenedContainers(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.WriteFile(filepath.Join(home, configFile), []byte(`{"docker":{"image":"x"}}`), 0644); err != nil {
		t.Fatal(err)
	}
	cfg := loadConfig()
	if cfg.Docker.Network != netNone || !cfg.Docker.readOnly() || cfg.Docker.PidsLimit != defaultPidsLimit {
		t.Fatalf("unexpected defaults: %+v", cfg.Docker)
	}
}

func TestConfigForModelNetworkOverride(t *testing.T) {
	bridge := netBridge
	cfg := defaultConfig()
	cfg.Models = map[string]ModelConfig{"m": {DockerNetwork: &bridge}}
	if got := configForModel(cfg, "m").Docker.Network; got != netBridge {
		t.Fatalf("network = %q", got)
	}
	if cfg.Docker.Network != netNone {
		t.Fatalf("global network mutated")
	}
}

func TestInstanceBackendAddressing(t *testing.T) {
	if got := instanceBackend(Instance{Socket: "/run/a.sock"}).String(); got != "unix:///run/a.sock" {
		t.Fatalf("socket: %s", got)
	}
	if got := instanceBackend(Instance{Addr: "172.18.0.2:9100"}).String(); got != "http://172.18.0.2:9100" {
		t.Fatalf("addr: %s", got)
	}
	if got := instanceBackend(Instance{Port: 9101}).String(); got != "http://127.0.0.1:9101" {
		t.Fatalf("legacy port: %s", got)
	}
}

func TestBackendOverUnixSocket(t *testing.T) {
	dir, err := os.MkdirTemp("", "llmsock")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	sock := filepath.Join(dir, "llama.sock")
	ln, err := net.Listen("unix", sock)
	if err != nil {
		t.Skipf("unix sockets unavailable: %v", err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_, _ = w.Write([]byte(`{"status":"ok"}`))
			return
		}
		_, _ = w.Write([]byte("chat:" + r.URL.Path))
	})}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	inst := Instance{Name: "x", Socket: sock}
	if got := backendStatus(inst); got != "healthy" {
		t.Fatalf("status = %q", got)
	}
	if healthy, _, _ := waitForHealth(instanceBackend(inst), "unused", 2*time.Second); !healthy {
		t.Fatal("waitForHealth over unix socket failed")
	}

	// Reverse proxy wiring as used by startProxy.
	target := instanceBackend(inst)
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.URL.Scheme
			req.URL.Host = target.URL.Host
			req.Host = target.URL.Host
		},
		Transport: target.Transport,
	}
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`)))
	if rec.Body.String() != "chat:/v1/chat/completions" {
		t.Fatalf("proxied body = %q (code %d)", rec.Body.String(), rec.Code)
	}
}

func TestHFSelectFiles(t *testing.T) {
	files := []string{
		"README.md",
		"Model-Q4_K_M.gguf",
		"Model-Q8_0.gguf",
		"Model-UD-Q4_K_XL.gguf",
		"BF16/Model-BF16-00001-of-00002.gguf",
		"BF16/Model-BF16-00002-of-00002.gguf",
		"mmproj-BF16.gguf",
		"mmproj-F16.gguf",
	}
	cases := []struct {
		tag    string
		model  []string
		errSub string
	}{
		{"", []string{"Model-Q4_K_M.gguf"}, ""},
		{"q8_0", []string{"Model-Q8_0.gguf"}, ""},
		{"UD-Q4_K_XL", []string{"Model-UD-Q4_K_XL.gguf"}, ""},
		{"BF16", []string{"BF16/Model-BF16-00001-of-00002.gguf", "BF16/Model-BF16-00002-of-00002.gguf"}, ""},
		{"Model-Q8_0.gguf", []string{"Model-Q8_0.gguf"}, ""},
		{"IQ1_S", nil, "no .gguf matching"},
	}
	for _, c := range cases {
		model, mmproj, err := hfSelectFiles(files, c.tag)
		if c.errSub != "" {
			if err == nil || !strings.Contains(err.Error(), c.errSub) {
				t.Errorf("tag %q: err = %v, want %q", c.tag, err, c.errSub)
			}
			continue
		}
		if err != nil {
			t.Errorf("tag %q: %v", c.tag, err)
			continue
		}
		if strings.Join(model, ",") != strings.Join(c.model, ",") {
			t.Errorf("tag %q: model = %v, want %v", c.tag, model, c.model)
		}
		if mmproj != "mmproj-F16.gguf" {
			t.Errorf("tag %q: mmproj = %q", c.tag, mmproj)
		}
	}

	model, mmproj, err := hfSelectFiles([]string{"only-IQ2.gguf"}, "")
	if err != nil || len(model) != 1 || model[0] != "only-IQ2.gguf" || mmproj != "" {
		t.Fatalf("fallback to first gguf: %v %q %v", model, mmproj, err)
	}
}

func TestFindInHFCacheNestedFile(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "hub", "models--u--r", "snapshots", "abc", "BF16")
	if err := os.MkdirAll(nested, 0755); err != nil {
		t.Fatal(err)
	}
	f := filepath.Join(nested, "m-00001-of-00002.gguf")
	if err := os.WriteFile(f, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if got := findInHFCache(dir, "models--u--r/BF16/m-00001-of-00002.gguf"); got != f {
		t.Fatalf("got %q", got)
	}
}

func TestResolveLoadSpecLeavesHFToContainerOnBridge(t *testing.T) {
	cfg := testConfig(t.TempDir())
	cfg.Docker.Network = netBridge
	spec, err := resolveLoadSpec(cfg, loadOptions{HFRepo: "u/r:Q4_K_M"})
	if err != nil {
		t.Fatal(err)
	}
	if spec.HFRepo != "u/r:Q4_K_M" {
		t.Fatalf("bridge mode should pass -hf through, got %+v", spec)
	}
}

func TestResolveLoadSpecUsesCachedHFFileWhenIsolated(t *testing.T) {
	dir := t.TempDir()
	snap := filepath.Join(dir, "models--u--r", "snapshots", "abc")
	if err := os.MkdirAll(snap, 0755); err != nil {
		t.Fatal(err)
	}
	f := filepath.Join(snap, "m-Q4_K_M.gguf")
	if err := os.WriteFile(f, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	cfg := testConfig(dir)
	cfg.Docker.Network = netNone
	spec, err := resolveLoadSpec(cfg, loadOptions{HFRepo: "u/r:m-Q4_K_M.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if spec.HFRepo != "" || spec.ModelPath != f {
		t.Fatalf("expected local file load, got %+v", spec)
	}
}

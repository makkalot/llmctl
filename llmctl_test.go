package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	for _, want := range []string{"[Unit]", "[Service]", "[Install]", "ExecStart=/usr/local/bin/llmctl proxy", "Restart=always", "WantedBy=multi-user.target"} {
		if !strings.Contains(content, want) {
			t.Errorf("unit file missing: %s", want)
		}
	}
}

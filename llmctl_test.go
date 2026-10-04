package main

import (
	"bytes"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
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

func TestModelConcurrencyJSONParsingAndInheritance(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(`{"models":{"big":{"concurrency":2}}}`), &cfg); err != nil {
		t.Fatal(err)
	}
	mc := cfg.Models["big"]
	field, ok := reflect.TypeOf(mc).FieldByName("Concurrency")
	if !ok || field.Type != reflect.TypeOf((*int)(nil)) {
		t.Fatalf("ModelConfig.Concurrency must be *int, got %v", field.Type)
	}
	value := reflect.ValueOf(mc).FieldByName("Concurrency")
	if value.IsNil() || value.Elem().Int() != 2 {
		t.Fatalf("decoded concurrency = %v, want pointer to 2", value.Interface())
	}
	base := testConfig(t.TempDir())
	base.Models = map[string]ModelConfig{"big": mc}
	inherited := reflect.ValueOf(configForModel(base, "big")).FieldByName("Concurrency")
	if !inherited.IsValid() || inherited.IsNil() || inherited.Elem().Int() != 2 {
		t.Fatalf("configForModel did not inherit concurrency: %v", inherited)
	}
	unset := reflect.ValueOf(configForModel(base, "missing")).FieldByName("Concurrency")
	if unset.IsValid() && !unset.IsNil() {
		t.Fatalf("unset model concurrency = %v, want nil", unset.Interface())
	}
}

func TestModelConcurrencySemaphoreConfigUsesResolvedAlias(t *testing.T) {
	cfg := testConfig(t.TempDir())
	cfg.Aliases = map[string]string{"code": "backend"}
	cfg.Models = map[string]ModelConfig{"code": {Concurrency: intPtr(2)}}
	if got := concurrencyConfigKey(cfg, "code"); got != "code" {
		t.Fatalf("alias route config key = %q, want code", got)
	}
	if got := concurrencyConfigKey(cfg, "backend"); got != "" {
		t.Fatalf("resolved backend without matching model key = %q, want empty", got)
	}
}

func TestConcurrencyParallelConflictIsRejected(t *testing.T) {
	for _, extraArgs := range [][]string{
		{"--parallel", "9"},
		{"--parallel=9"},
	} {
		cfg := testConfig(t.TempDir())
		cfg.Models = map[string]ModelConfig{"model": {Concurrency: intPtr(2), ExtraArgs: extraArgs}}
		model := writeLocalModel(t, filepath.Join(t.TempDir(), "model.gguf"))
		cfg.ServerBin = filepath.Join(t.TempDir(), "unused")
		_, _, err := loadInstance(cfg, loadOptions{ModelName: model, InstanceName: "model"})
		if err == nil || !strings.Contains(err.Error(), "must not contain --parallel") {
			t.Fatalf("extra args %v: expected configured parallel conflict, got %v", extraArgs, err)
		}
	}
}

func TestConcurrencyAddsParallelAndUnsetPreservesArgs(t *testing.T) {
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "model.gguf"))
	cfg := testConfig(root)
	cfg.Models = map[string]ModelConfig{"model": {Concurrency: intPtr(3), ExtraArgs: []string{"--foo", "bar"}}}
	spec, err := resolveLoadSpec(cfg, loadOptions{ModelName: model, InstanceName: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Config.Concurrency == nil || *spec.Config.Concurrency != 3 {
		t.Fatalf("resolved concurrency = %v", spec.Config.Concurrency)
	}
	if !containsArgPair(spec.Config.ExtraArgs, "--foo", "bar") {
		t.Fatalf("resolved extra args = %v", spec.Config.ExtraArgs)
	}
	unset, err := resolveLoadSpec(testConfig(root), loadOptions{ModelName: model, InstanceName: "unset"})
	if err != nil {
		t.Fatal(err)
	}
	if unset.Config.Concurrency != nil {
		t.Fatal("unset model gained concurrency")
	}
}

func containsArgPair(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestConcurrencyDocumentationDescribesQueueAndUnlimitedBehavior(t *testing.T) {
	data, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		"concurrency",
		"wait in a queue",
		"aliases",
		"unlimited",
		"--parallel <concurrency>",
		"ctx_size / concurrency",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("README is missing concurrency documentation %q", want)
		}
	}
}

func TestConcurrencyRouteUsesResolvedNameBeforeAutoswitch(t *testing.T) {
	cfg := testConfig(t.TempDir())
	cfg.Aliases = map[string]string{"alias": "model"}
	cfg.Models = map[string]ModelConfig{"alias": {Concurrency: intPtr(1)}}
	if got := concurrencyConfigKey(cfg, "alias"); got != "alias" {
		t.Fatalf("route key = %q", got)
	}
	// A resolved route must still identify the alias-owned configuration.
	if got := concurrencyConfigKey(cfg, "model"); got != "" {
		t.Fatalf("raw target incorrectly selected config: %q", got)
	}
}

func TestConcurrencyAliasesAndResolvedFuzzyNamesShareQueueKey(t *testing.T) {
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "canonical-model.gguf"))
	cfg := testConfig(root)
	cfg.Aliases = map[string]string{"code": model}
	cfg.Models = map[string]ModelConfig{"code": {Concurrency: intPtr(1)}}

	aliasKey := concurrencyConfigKey(cfg, "code")
	resolvedName := deriveInstanceName(model)
	resolvedKey := concurrencyConfigKey(cfg, resolvedName)
	if aliasKey == "" {
		t.Fatal("alias did not resolve to a configured concurrency key")
	}
	if resolvedKey != aliasKey {
		t.Fatalf("alias key %q and resolved/fuzzy key %q differ; requests can bypass one semaphore", aliasKey, resolvedKey)
	}

	// A request for the unloaded canonical model must use the same queue key
	// before autoswitch resolves it, rather than bypassing the alias semaphore.
	canonicalKey := concurrencyConfigKey(cfg, model)
	if canonicalKey != aliasKey {
		t.Fatalf("canonical key %q and alias key %q differ for unloaded request", canonicalKey, aliasKey)
	}
}

func TestConcurrencyResolvedQueueKeyIsStableAcrossFuzzySpellings(t *testing.T) {
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "Qwen3.5-27B-Q4_K_M.gguf"))
	cfg := testConfig(root)
	cfg.Aliases = map[string]string{"big": model}
	cfg.Models = map[string]ModelConfig{"big": {Concurrency: intPtr(1)}}

	want := concurrencyConfigKey(cfg, "big")
	for _, route := range []string{
		model,
		deriveInstanceName(model),
		"qwen3.5-27b-q4_k_m",
		"QWEN3.5-27B-Q4_K_M",
	} {
		if got := concurrencyConfigKey(cfg, route); got != want {
			t.Fatalf("route %q got queue key %q, want canonical key %q", route, got, want)
		}
	}
}

func TestConcurrencyQueueKeyUsesConfiguredAliasForCanonicalAndFuzzyRoutes(t *testing.T) {
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "models", "Qwen3.5-27B-Q4_K_M.gguf"))
	cfg := testConfig(filepath.Join(root, "models"))
	cfg.Aliases = map[string]string{"big": model}
	cfg.Models = map[string]ModelConfig{"big": {Concurrency: intPtr(2)}}

	want := "big"
	routes := []string{
		"big",                     // alias
		model,                     // canonical path before autoswitch
		deriveInstanceName(model), // loaded instance name
		"qwen3.5-27b-q4_k_m",      // fuzzy basename
		"QWEN3.5-27B-Q4_K_M",      // case-insensitive fuzzy name
	}
	for _, route := range routes {
		if got := concurrencyConfigKey(cfg, route); got != want {
			t.Fatalf("route %q got queue key %q, want %q; route can bypass shared semaphore", route, got, want)
		}
	}
}

func TestLoadInstanceFinalArgumentsAutoPassParallelAndPreserveUnset(t *testing.T) {
	root := t.TempDir()
	capture := filepath.Join(root, "args")
	server := filepath.Join(root, "fake-server.sh")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"" + capture + "\"\n"
	if err := os.WriteFile(server, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	model := writeLocalModel(t, filepath.Join(root, "model.gguf"))

	cfg := testConfig(root)
	cfg.ServerBin = server
	cfg.Models = map[string]ModelConfig{"model": {Concurrency: intPtr(3), ExtraArgs: []string{"--foo", "bar"}}}
	_, _, _ = loadInstance(cfg, loadOptions{ModelName: model, InstanceName: "model", WaitTimeout: time.Millisecond})
	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(strings.TrimSpace(string(data)), "\n")
	for _, pair := range [][2]string{
		{"--model", model},
		{"--ctx-size", "0"},
		{"--parallel", "3"},
		{"--foo", "bar"},
	} {
		if !containsArgPair(args, pair[0], pair[1]) {
			t.Fatalf("final llama-server args = %v, missing %s %s", args, pair[0], pair[1])
		}
	}
	parallelCount := 0
	for _, arg := range args {
		if arg == "--parallel" || strings.HasPrefix(arg, "--parallel=") {
			parallelCount++
		}
	}
	if parallelCount != 1 {
		t.Fatalf("final llama-server args = %v, want exactly one automatic --parallel", args)
	}

	captureUnset := filepath.Join(root, "args-unset")
	serverUnset := filepath.Join(root, "fake-server-unset.sh")
	scriptUnset := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"" + captureUnset + "\"\n"
	if err := os.WriteFile(serverUnset, []byte(scriptUnset), 0755); err != nil {
		t.Fatal(err)
	}
	unset := testConfig(root)
	unset.ServerBin = serverUnset
	unset.ExtraArgs = []string{"--foo", "bar"}
	_, _, _ = loadInstance(unset, loadOptions{ModelName: model, InstanceName: "unset", WaitTimeout: time.Millisecond})
	unsetData, err := os.ReadFile(captureUnset)
	if err != nil {
		t.Fatal(err)
	}
	if containsArgPair(strings.Split(strings.TrimSpace(string(unsetData)), "\n"), "--parallel", "3") {
		t.Fatal("unset concurrency unexpectedly added --parallel")
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

func startProxyForConcurrencyTest(t *testing.T, cfg Config, reg Registry) (string, func()) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	cfgData, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, configFile), cfgData, 0644); err != nil {
		t.Fatal(err)
	}
	if err := saveRegistry(reg); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "run", "llmctl.go", "proxy")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	cleanup := func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}
	addr := net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port))
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://" + addr + "/health")
		if err == nil {
			resp.Body.Close()
			return "http://" + addr, cleanup
		}
		time.Sleep(25 * time.Millisecond)
	}
	cleanup()
	t.Fatal("proxy did not become ready")
	return "", func() {}
}

func TestProxyConcurrencyQueuesResolvedRoutesAndDefaultFallback(t *testing.T) {
	backendBlock := make(chan struct{})
	backendStarted := make(chan struct{}, 4)
	var backendCalls chan struct{}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendStarted <- struct{}{}
		if backendCalls != nil {
			backendCalls <- struct{}{}
		}
		<-backendBlock
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	}))
	defer backend.Close()
	backendURL := strings.TrimPrefix(backend.URL, "http://")
	backendPort, err := strconv.Atoi(strings.TrimPrefix(backendURL, "127.0.0.1:"))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "canonical-model.gguf"))
	cfg := testConfig(root)
	cfg.Host = "127.0.0.1"
	cfg.Port = freeTCPPort(t)
	cfg.Aliases = map[string]string{"code": model}
	cfg.Models = map[string]ModelConfig{
		"code":      {Concurrency: intPtr(1)},
		"unlimited": {},
	}
	backendProcess := exec.Command("sleep", "60")
	if err := backendProcess.Start(); err != nil {
		t.Fatal(err)
	}
	defer backendProcess.Process.Kill()
	reg := Registry{Instances: []Instance{
		{Name: "code", Model: model, Port: backendPort, PID: backendProcess.Process.Pid},
		{Name: "canonical-model", Model: model, Port: backendPort, PID: backendProcess.Process.Pid, IsDefault: true},
		{Name: "unlimited", Model: model, Port: backendPort, PID: backendProcess.Process.Pid},
	}}
	base, stop := startProxyForConcurrencyTest(t, cfg, reg)
	defer stop()

	post := func(modelName string) *http.Request {
		body := `{"messages":[{"role":"user","content":"hello"}]}`
		req, _ := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("X-Model", modelName)
		return req
	}
	// Alias, canonical, and fuzzy names must contend for one semaphore.
	backendCalls = make(chan struct{}, 4)
	firstDone := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(post("code"))
		if resp != nil {
			resp.Body.Close()
		}
		firstDone <- err
	}()
	select {
	case <-backendStarted:
	case <-time.After(time.Second):
		t.Fatal("first request did not reach backend")
	}
	select {
	case <-backendCalls:
	case <-time.After(time.Second):
		t.Fatal("first backend call was not recorded")
	}
	secondDone := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(post("canonical-model"))
		if resp != nil {
			resp.Body.Close()
		}
		secondDone <- err
	}()
	select {
	case <-backendCalls:
		t.Fatal("canonical route bypassed alias queue")
	case <-time.After(150 * time.Millisecond):
	}
	close(backendBlock)
	select {
	case err := <-firstDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("first request did not release")
	}
	select {
	case err := <-secondDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("queued request was not released")
	}

	// Unset concurrency must not serialize independent requests.
	backendBlock = make(chan struct{})
	backendStarted = make(chan struct{}, 4)
	backendCalls = make(chan struct{}, 4)
	unlimited1 := make(chan error, 1)
	unlimited2 := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(post("unlimited"))
		if resp != nil {
			resp.Body.Close()
		}
		unlimited1 <- err
	}()
	go func() {
		resp, err := http.DefaultClient.Do(post("unlimited"))
		if resp != nil {
			resp.Body.Close()
		}
		unlimited2 <- err
	}()
	select {
	case <-backendCalls:
	case <-time.After(time.Second):
		t.Fatal("unlimited request did not reach backend")
	}
	select {
	case <-backendCalls:
	case <-time.After(time.Second):
		t.Fatal("nil concurrency unexpectedly blocked second request")
	}
	close(backendBlock)
	select {
	case <-unlimited1:
	case <-time.After(time.Second):
		t.Fatal("unlimited request did not finish")
	}
	select {
	case <-unlimited2:
	case <-time.After(time.Second):
		t.Fatal("unlimited request did not finish")
	}

}

func TestProxyConcurrencyQueuesConfiguredDefaultFallback(t *testing.T) {
	backendBlock := make(chan struct{})
	backendStarted := make(chan struct{}, 2)
	backendCalls := make(chan struct{}, 2)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendStarted <- struct{}{}
		backendCalls <- struct{}{}
		<-backendBlock
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	}))
	defer backend.Close()

	backendPort, err := strconv.Atoi(strings.TrimPrefix(backend.URL, "http://127.0.0.1:"))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	model := writeLocalModel(t, filepath.Join(root, "default-model.gguf"))
	cfg := testConfig(root)
	cfg.Host = "127.0.0.1"
	cfg.Port = freeTCPPort(t)
	cfg.Models = map[string]ModelConfig{"default": {Concurrency: intPtr(1)}}
	backendProcess := exec.Command("sleep", "60")
	if err := backendProcess.Start(); err != nil {
		t.Fatal(err)
	}
	defer backendProcess.Process.Kill()
	reg := Registry{Instances: []Instance{{Name: "default", Model: model, Port: backendPort, PID: backendProcess.Process.Pid, IsDefault: true}}}
	base, stop := startProxyForConcurrencyTest(t, cfg, reg)
	defer stop()

	postWithoutModel := func() *http.Request {
		body := `{"messages":[{"role":"user","content":"hello"}]}`
		req, _ := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(body))
		return req
	}
	do := func(done chan<- error) {
		resp, err := http.DefaultClient.Do(postWithoutModel())
		if resp != nil {
			resp.Body.Close()
		}
		done <- err
	}
	firstDone := make(chan error, 1)
	go do(firstDone)
	select {
	case <-backendStarted:
	case <-time.After(time.Second):
		t.Fatal("fallback request did not reach backend")
	}
	select {
	case <-backendCalls:
	case <-time.After(time.Second):
		t.Fatal("first fallback call was not recorded")
	}
	secondDone := make(chan error, 1)
	go do(secondDone)
	select {
	case <-backendCalls:
		t.Fatal("default fallback bypassed concurrency queue")
	case <-time.After(150 * time.Millisecond):
	}
	close(backendBlock)
	for name, done := range map[string]chan error{"first": firstDone, "second": secondDone} {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("%s fallback request: %v", name, err)
			}
		case <-time.After(time.Second):
			t.Fatalf("%s fallback request was not released", name)
		}
	}
}

func freeTCPPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	return port
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

func TestREADMEDocumentsConcurrencyQueueingAndParallelEffects(t *testing.T) {
	data, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	readme := string(data)
	checks := []struct {
		name    string
		pattern string
	}{
		{"per-model concurrency setting", `(?i)concurrency`},
		{"queued requests", `(?i)queue`},
		{"automatic backend parallelism", `(?is)automatically.{0,200}--parallel`},
		{"llama-server parallel flag", `(?i)--parallel`},
		{"VRAM impact", `(?i)VRAM`},
		{"context per slot impact", `(?i)context[^[:space:]]*[[:space:]-]*per[[:space:]-]*slot|context per slot`},
	}
	for _, check := range checks {
		matched, err := regexp.MatchString(check.pattern, readme)
		if err != nil {
			t.Fatalf("invalid README check %s: %v", check.name, err)
		}
		if !matched {
			t.Errorf("README.md missing %s documentation (/%s/)", check.name, check.pattern)
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

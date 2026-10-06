package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

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

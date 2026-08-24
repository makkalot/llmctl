# Plan: Per-API-Key Model Scoping

Branch: `feature/auth-rate-limit` (extends existing auth + rate-limit work)

## Goal

Allow restricting which models a given API key can use through the proxy.
Scope entries are **aliases or full model names** — anything resolvable via
`llmctl.json` (`Aliases`, `Models` keys, or a discoverable `.gguf`).

## Decisions (agreed)

1. **Model-level scoping** — a scope entry matches the *model weights*
   (cleaned `.gguf` path), not the instance name. A key scoped to an alias
   also covers the same model loaded under its full name, and vice versa.
2. **`auth models` subcommand** — no strong preference; include it (edit a
   key's scope without regenerating the key).
3. **`/v1/models` filtering** — yes, list only models the key may access.

## Design

### 1. Data model

```go
type APIKey struct {
    Key       string   `json:"key"`
    Name      string   `json:"name"`
    RateLimit int      `json:"rate_limit"`
    CreatedAt string   `json:"created_at"`
    Models    []string `json:"models,omitempty"` // aliases or full model names; empty = all
}
```

- `Models` empty/omitted = unrestricted (matches `rate_limit: 0 = unlimited`
  convention).
- Stored in `~/.llmctl.json` under `api_keys`, same as today.

### 2. CLI

- `llmctl auth generate <name> [--limit N] [--model <ref>]...`
  - `--model` repeatable; accepts aliases and full model names.
  - Validation: each ref must resolve (alias in `cfg.Aliases`, key in
    `cfg.Models`, or a matching `.gguf` via existing discovery/fuzzy
    helpers). Unknown ref → hard error listing valid names.
- `llmctl auth list` — add `Models` column (`all` when empty).
- `llmctl auth models <name> <ref>...` — set a key's scope (same validation
  as generate). No refs = clear restriction (unrestricted).

### 3. Enforcement

Location: `innerHandler` (after `targetName` resolution), not
`authMiddleware` — the middleware doesn't parse the body and the check must
run against the *resolved* target (exact → fuzzy → default fallback).

Match rule: a scope entry grants access to an instance if either

1. it equals the instance name exactly, or
2. both the entry and the instance resolve (via `resolveModel` / alias
   lookup) to the same cleaned `.gguf` path.

The default-fallback path is also subject to the check (a restricted key
with no model in the request must still be denied if the default instance
is out of scope).

Deny response — 403, OpenAI error shape:

```json
{"error": {"message": "API key '<name>' is not allowed to use model '<target>'", "type": "permission_error"}}
```

Implementation notes:

- Precompute a per-key set of allowed cleaned paths + exact instance names
  once per request (or lazily), reusing `resolveModel` /
  `aliasTargetMatchesModel` helpers where possible.
- Resolve scope entries against `cfg` loaded at proxy start (same caveat as
  existing auth: config changes require a proxy restart).

### 4. `/v1/models` filtering

- Middleware sets the authenticated `*APIKey` on the request context after
  validation (e.g. `context.WithValue` with a private key type).
- `handleListModels` reads the key from context; if `Models` is non-empty,
  filter the returned list to in-scope models (same model-level match
  rule as enforcement).
- Unrestricted keys see the full list.

### 5. Tests (`llmctl_test.go`)

- `generate` with `--model` persists `Models`; unknown ref errors.
- Access check:
  - exact instance-name match → allowed
  - alias ↔ full-name cross-match (same .gguf) → allowed
  - out-of-scope model → 403 `permission_error`
  - empty `Models` → unrestricted
  - default-fallback instance is subject to the check
- `auth models` set/clear behavior.
- `/v1/models` filtered per key; unrestricted key sees all.

### 6. Docs

- README auth section: scoping syntax + examples.
- `auth` help text updated.

## Files touched

- `llmctl.go` — `APIKey` struct, `cmdAuth*` functions, `authMiddleware`
  (context), `innerHandler` (access check), `handleListModels` (filter)
- `llmctl_test.go` — new tests
- `README.md` — docs

## Out of scope

- Global (non-per-key) model restrictions
- Per-key overrides of rate limit at request time
- Hot-reloading of config changes (existing restart-required caveat stands)

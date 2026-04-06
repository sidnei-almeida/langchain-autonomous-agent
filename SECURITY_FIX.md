# Security advisory — credential handling

This document describes recommended practices for API keys and related secrets used with this repository. It also records a class of incident (accidental commit of environment files) and the mitigations applied in the project configuration.

---

## Scope

- Groq API keys (`GROQ_API_KEY`)
- Any future third-party tokens added to the codebase

---

## Policy

1. **Never commit secrets** — `.env` and similar files must remain local or in platform secret stores only.
2. **Use `.gitignore`** — The repository excludes `.env`, virtual environments, and common cache paths.
3. **Rotate on exposure** — If a key appears in Git history, chat logs, or CI output, revoke it at the provider and issue a new key.
4. **Least privilege** — Create keys with the minimum scopes required by your use case.

---

## If a key was committed

1. **Revoke** the exposed key in the Groq console.
2. **Generate** a new key and update deployment secrets and local `.env` only.
3. **Remove** the file from the index going forward; for historical commits, use `git filter-repo` or GitHub’s secret scanning remediation if available — coordinate with repository admins before rewriting public history.

---

## Production hardening

- Serve the API behind TLS.
- Restrict CORS (`allow_origins`) to known front-end origins.
- Add authentication (API keys, OAuth2, mutual TLS) at the reverse proxy or application layer.
- Apply rate limiting and request size limits at the edge.

---

## Verification

To confirm `.env` is not tracked:

```bash
git show HEAD:.env
```

A healthy repository reports that the path does not exist in `HEAD`.

---

## References

- [Groq console](https://console.groq.com) — key management
- [GitHub push protection](https://docs.github.com/code-security/secret-scanning/about-secret-scanning) — automated detection

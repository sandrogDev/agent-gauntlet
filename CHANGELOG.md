# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-30

### Added
- YAML suites with per-scenario Python tool registries and named business rules.
- Agent loop with pre-execution JSON Schema validation, single-shot fault injection and full trace capture.
- Providers: Anthropic Messages API, OpenAI-compatible Chat Completions, and `python:` plugins.
- Resilience layer: per-call timeout, full-jitter retries, circuit breaker.
- Deterministic checks (`schema_valid`, `completed`, `tool_called`, `tool_not_called`, `call_order`, `max_tool_calls`, `rule`, response checks) and an opt-in LLM judge.
- Mutators: `prompt_injection`, `authority_pressure`, `tool_failure`, `typo_noise`, `context_flood`.
- Run storage, scenario-level regression diff, console and self-contained HTML reports.
- CLI with quality gates and stable exit codes; GitHub Actions workflow; Docker image.
- Example airline-support suite and an offline demo agent with planted flaws.

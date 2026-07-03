# Architecture Overview

## System Design

The Churn Research Framework is built on a modular, registry-based architecture:

```
┌─────────────────────────────────────────────────┐
│                    CLI Layer                      │
│              (Typer + Rich)                       │
├─────────────────────────────────────────────────┤
│                Python API Layer                   │
│            (ChurnFramework class)                 │
├─────────────────────────────────────────────────┤
│              Core Infrastructure                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Registry │ │ Config   │ │ Pipeline Context │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────┤
│              Plugin Registries                    │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │Datasets│ │Models│ │Metrics│ │Churn │ │Reports│ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
├─────────────────────────────────────────────────┤
│              Pipeline Engine                      │
│  Load → Validate → Churn → Features → Train →    │
│  Eval → SHAP → Calibration → Risk → Reports      │
└─────────────────────────────────────────────────┘
```

## Key Principles

1. **Registry-based extensibility** — All pluggable components register through `PluginRegistry`
2. **Graceful degradation** — Optional failures are logged but don't halt execution
3. **Backward compatibility** — New features wrap existing code; nothing breaks
4. **Research-first** — No silent auto-selection; researchers make decisions

## Data Flow

```
Raw CSV → Adapter → Preprocess → Standardize → Churn Labels →
Features → Temporal Split → (Resampling) → Train → Evaluate →
SHAP → Calibration → Risk → Segmentation → Reports → Exports
```

## Module Organization

| Directory | Purpose |
|-----------|---------|
| `src/core/` | Registry, context, infrastructure |
| `src/datasets/` | Dataset adapters (one per dataset) |
| `src/churn/` | Churn labeling strategies |
| `src/models/` | Model wrappers |
| `src/metrics/` | Evaluation metrics |
| `src/resamplers/` | Data resampling methods |
| `src/reports/` | Report generators |
| `src/wizard/` | Dataset registration |
| `src/cli/` | CLI commands |
| `src/profiling/` | Data profiling |
| `configs/` | YAML configurations |

## Adding New Components

See [Plugin Development](plugins.md) for extending each registry.

---

*Developed by Nikhil Wankhede*

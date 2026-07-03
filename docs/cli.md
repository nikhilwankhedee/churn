# CLI Reference

The `churn` CLI provides 20+ commands for running experiments, profiling datasets, and managing configurations.

## Commands

### Pipeline Execution

```bash
churn run one <dataset>           # Run single dataset
churn run all                     # Run all registered datasets
churn run all --only olist,rees46 # Run specific datasets
churn run one olist --sensitivity # With sensitivity analysis
churn run one olist --window 90   # Override churn window
```

### Dataset Management

```bash
churn datasets                    # List all registered datasets
churn register data.csv           # Register new dataset from CSV
churn register data.csv --name my_ds --ecosystem subscription
churn profile <dataset>           # Profile a dataset
churn validate <dataset>          # Validate schema + behavior
churn features <dataset>          # Show feature groups
```

### Configuration

```bash
churn config show                 # Display current config
churn config init my_experiment   # Create new config from defaults
churn validate-config config.yaml # Validate a YAML config
```

### Experiment History

```bash
churn experiments list            # List recent experiments
churn experiments list --dataset olist
churn experiments compare olist,rees46
churn experiments features        # Feature group comparison
churn compare olist,rees46        # Quick comparison
```

### Registry Introspection

```bash
churn models                      # List registered models
churn strategies                  # List churn strategies
churn metrics                     # List evaluation metrics
churn resamplers                  # List resamplers
churn plugins                     # List all plugins
```

### System

```bash
churn doctor                      # Health check
churn version                     # Version info
churn info                        # Framework overview
churn docs                        # List documentation topics
churn docs quickstart             # Show quickstart guide
```

## Global Options

- `--config, -c <path>` — Load a YAML configuration file
- `--help` — Show help for any command
- `--no-banner` — Suppress startup banner (in run commands)

## Output Locations

| Output | Default Path |
|--------|-------------|
| Models | `models/` |
| Results | `results/` |
| Figures | `figures/` |
| Reports | `results/reports/` |
| Experiments | `results/experiments/` |

---

*Developed by Nikhil Wankhede*

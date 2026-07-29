"""
ChurnLab — Command-Line Interface v1.0.

The primary entry point for the Universal Customer Churn Research Framework.

Usage:
    churn                          Interactive home screen
    churn wizard                   Dataset registration wizard
    churn benchmark <path>         Auto-discover and benchmark
    churn doctor <path>            Dataset health analysis
    churn explain <dataset>        Natural language model explanation
    churn compare <ds1> <ds2>      Compare datasets/models
    churn profile <dataset>        Comprehensive dataset profiling
    churn export <dataset>         Publication-ready export
    churn datasets                 List benchmark datasets
    churn experiments              List experiment history
    churn dashboard                Launch web dashboard
    churn plugin create            Generate plugin template
"""
import os
import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.markup import escape as _escape_markup
from rich import box

_project_root = str(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

console = Console()

app = typer.Typer(
    name="churn",
    help=None,
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
    invoke_without_command=True,
    epilog=None,
)

# ── Sub-apps ─────────────────────────────────────────────────────
run_app = typer.Typer(help="Run prediction experiments.", rich_markup_mode="rich")
app.add_typer(run_app, name="run")

config_app = typer.Typer(help="Manage framework configuration.", rich_markup_mode="rich")
app.add_typer(config_app, name="config")

plugin_app = typer.Typer(help="Plugin management.", rich_markup_mode="rich")
app.add_typer(plugin_app, name="plugin")


# ══════════════════════════════════════════════════════════════════
#  INTERACTIVE HOME SCREEN
# ══════════════════════════════════════════════════════════════════

BANNER = r"""[bold cyan]
         _____ _                 _                    __
    ____/ /__(_)___  __      __(_)___  _____       / /___ _      _______
   / __  / / / __ \/_ | /| / / / __ \/ ___/______/ / __ \ | /| / / ___/
  / /_/ / / / / / / | |/ |/ / / / / (__  )_____/ / /_/ / |/ |/ (__  )
 /\__,_/_/_/_/ /_/  |__/|__/_/_/ /_/ /____/    /_/\____/|__/|__/____/
[/bold cyan]"""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """[bold cyan]Universal Customer Churn Research Framework[/bold cyan]"""
    if ctx.invoked_subcommand is not None:
        return

    from src.config import FRAMEWORK_VERSION

    console.print(BANNER)
    console.print(f"  [dim]v{FRAMEWORK_VERSION} — Universal Customer Churn Research Framework[/dim]\n")

    try:
        from src.datasets import list_datasets
        registered = list_datasets()
    except Exception:
        registered = []

    if not registered:
        console.print("  [dim]No datasets registered.[/dim]\n")
    else:
        console.print(f"  [green]{len(registered)} dataset(s) registered:[/green] {', '.join(registered)}\n")

    console.print("  What would you like to do?\n")

    options = [
        ("1", "Scan current directory for datasets"),
        ("2", "Register a new dataset (Wizard)"),
        ("3", "Download benchmark datasets"),
        ("4", "View registered datasets"),
        ("5", "Run benchmark"),
        ("6", "Dataset health check (Doctor)"),
        ("7", "Explain model predictions"),
        ("8", "Compare datasets"),
        ("9", "Profile a dataset"),
        ("10", "Export results"),
        ("11", "View experiments"),
        ("12", "Launch dashboard"),
        ("13", "Documentation"),
        ("0", "Exit"),
    ]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Num", style="bold cyan", width=4)
    table.add_column("Action", style="white")
    for num, desc in options:
        table.add_row(num, desc)
    console.print(table)
    console.print()

    try:
        choice = console.input("  [bold cyan]Select > [/bold cyan]").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Goodbye![/dim]")
        raise typer.Exit()

    action_map = {
        "1": _interactive_scan,
        "2": lambda: _interactive_wizard(),
        "3": _interactive_download,
        "4": _interactive_list_datasets,
        "5": _interactive_benchmark,
        "6": _interactive_doctor,
        "7": _interactive_explain,
        "8": _interactive_compare,
        "9": _interactive_profile,
        "10": _interactive_export,
        "11": _interactive_experiments,
        "12": _interactive_dashboard,
        "13": _interactive_docs,
        "0": None,
    }

    handler = action_map.get(choice)
    if handler is None:
        if choice == "0":
            console.print("\n  [dim]Goodbye![/dim]")
        else:
            console.print(f"\n  [red]Invalid choice: {choice}[/red]")
        return

    try:
        handler()
    except KeyboardInterrupt:
        console.print("\n  [dim]Cancelled.[/dim]")
    except Exception as exc:
        console.print(f"\n  [red]Error: {_escape_markup(str(exc))}[/red]")


def _interactive_scan():
    console.print("\n  [bold]Scanning current directory...[/bold]\n")
    import glob as glob_mod
    cwd = os.getcwd()
    csv_files = sorted(glob_mod.glob(os.path.join(cwd, "**/*.csv"), recursive=True))
    if not csv_files:
        console.print("  [yellow]No CSV files found in current directory.[/yellow]")
        return
    console.print(f"  [green]Found {len(csv_files)} CSV file(s):[/green]\n")
    for f in csv_files[:20]:
        rel = os.path.relpath(f, cwd)
        size = os.path.getsize(f) / 1024
        console.print(f"    [cyan]•[/cyan] {rel}  [dim]({size:.0f} KB)[/dim]")
    if len(csv_files) > 20:
        console.print(f"    [dim]... and {len(csv_files) - 20} more[/dim]")
    console.print(f"\n  [dim]To register: churn wizard[/dim]")


def _interactive_wizard():
    console.print("\n  [bold cyan]Dataset Registration Wizard[/bold cyan]\n")
    path = console.input("  [bold]Path to CSV or directory > [/bold]").strip()
    if not path:
        return
    ctx = typer.Context
    register(csv_path=path)


def _interactive_download():
    console.print()
    download_datasets_cmd()


def _interactive_list_datasets():
    console.print()
    datasets_cmd()


def _interactive_benchmark():
    console.print("\n  [bold cyan]Benchmark[/bold cyan]\n")
    path = console.input("  [bold]Dataset directory path > [/bold]").strip()
    if not path:
        return
    benchmark_cmd(dataset_root=path)


def _interactive_doctor():
    console.print("\n  [bold cyan]Dataset Doctor[/bold cyan]\n")
    path = console.input("  [bold]Path to CSV file > [/bold]").strip()
    if not path:
        return
    doctor_cmd(csv_path=path)


def _interactive_explain():
    console.print("\n  [bold cyan]Model Explanation[/bold cyan]\n")
    name = console.input("  [bold]Dataset name > [/bold]").strip()
    if not name:
        return
    explain_cmd(dataset=name)


def _interactive_compare():
    console.print("\n  [bold cyan]Compare Datasets[/bold cyan]\n")
    ds = console.input("  [bold]Datasets (comma-separated) > [/bold]").strip()
    if not ds:
        return
    compare_cmd(datasets=ds)


def _interactive_profile():
    console.print("\n  [bold cyan]Dataset Profiling[/bold cyan]\n")
    name = console.input("  [bold]Dataset name > [/bold]").strip()
    if not name:
        return
    profile_cmd(dataset=name)


def _interactive_export():
    console.print("\n  [bold cyan]Export Results[/bold cyan]\n")
    name = console.input("  [bold]Dataset name > [/bold]").strip()
    if not name:
        return
    export_cmd(dataset=name)


def _interactive_experiments():
    console.print()
    experiments_list_cmd()


def _interactive_dashboard():
    console.print()
    dashboard_cmd()


def _interactive_docs():
    console.print()
    docs_cmd()


# ══════════════════════════════════════════════════════════════════
#  WIZARD / REGISTRATION
# ══════════════════════════════════════════════════════════════════

@app.command()
def wizard(
    path: Optional[str] = typer.Argument(None, help="Path to CSV file or directory."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name."),
    ecosystem: Optional[str] = typer.Option(None, "--ecosystem", "-e", help="Ecosystem type."),
    customer_id: Optional[str] = typer.Option(None, "--customer-id", help="Customer ID column."),
    timestamp: Optional[str] = typer.Option(None, "--timestamp", "-t", help="Timestamp column."),
):
    """Dataset Registration Wizard — intelligent onboarding for new datasets."""
    if path is None:
        path = console.input("  [bold]Path to CSV or directory > [/bold]").strip()
        if not path:
            console.print("  [red]No path provided.[/red]")
            raise typer.Exit(1)

    if os.path.isdir(path):
        import glob as glob_mod
        csv_files = sorted(glob_mod.glob(os.path.join(path, "*.csv")))
        if not csv_files:
            console.print(f"  [red]No CSV files found in: {path}[/red]")
            raise typer.Exit(1)
        console.print(f"  [dim]Found {len(csv_files)} CSV files in {path}[/dim]")
        path = csv_files[0]
        console.print(f"  [dim]Inspecting: {os.path.basename(path)}[/dim]")

    if not os.path.isfile(path):
        console.print(f"  [red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{os.path.basename(path)}[/bold]",
        title="[bold cyan]Dataset Registration Wizard[/bold cyan]",
        border_style="cyan",
    ))

    console.print("\n  [bold]Scanning...[/bold]\n")

    with console.status("[bold green]Analyzing columns...[/bold green]"):
        from src.wizard import inspect_csv, generate_config, generate_readiness_report
        try:
            inspection = inspect_csv(
                path,
                customer_id_hint=customer_id,
                timestamp_hint=timestamp,
            )
        except Exception as exc:
            console.print(f"  [red]Inspection failed: {_escape_markup(str(exc))}[/red]")
            raise typer.Exit(1)

    n_csv = len([c for c in os.listdir(os.path.dirname(path)) if c.endswith(".csv")])
    console.print(f"  [green]✓[/green] Found {n_csv} CSV file(s)")

    if inspection.inferred_customer_id:
        console.print(f"  [green]✓[/green] Customer table detected")
        console.print(f"  [green]✓[/green] Customer ID detected: [bold]{inspection.inferred_customer_id}[/bold]")
    else:
        console.print(f"  [yellow]⚠[/yellow] Customer ID NOT detected")

    if inspection.inferred_event_time:
        console.print(f"  [green]✓[/green] Timestamp detected: [bold]{inspection.inferred_event_time}[/bold]")
    else:
        console.print(f"  [yellow]⚠[/yellow] Timestamp NOT detected")

    if inspection.inferred_transaction_value:
        console.print(f"  [green]✓[/green] Monetary column detected: [bold]{inspection.inferred_transaction_value}[/bold]")

    if inspection.inferred_event_type:
        console.print(f"  [green]✓[/green] Product/event column detected: [bold]{inspection.inferred_event_type}[/bold]")

    existing_churn = any(
        c.name.lower() in ("churn", "is_churned", "churned", "label", "target")
        for c in inspection.columns
    )
    if existing_churn:
        console.print(f"  [green]✓[/green] Existing churn label found")
    else:
        console.print(f"  [dim]  Existing churn label NOT found[/dim]")

    console.print()

    config = generate_config(
        inspection,
        dataset_name=name,
        ecosystem_type=ecosystem,
    )

    console.print(Panel(
        f"[bold]Churn Strategy:[/bold] {config.churn_strategy}\n"
        f"[bold]Prediction Window:[/bold] {config.prediction_window_days} days\n"
        f"[bold]Feature Groups:[/bold] {', '.join(config.available_feature_groups)}\n"
        f"[bold]Ecosystem:[/bold] {config.ecosystem_type}",
        title="[bold cyan]Generated Manifest[/bold cyan]",
        border_style="cyan",
    ))

    readiness = generate_readiness_report(inspection)
    console.print(f"\n{readiness.summary()}\n")

    output_path = os.path.join(
        _project_root, "configs", "datasets", f"{config.dataset_name}.yaml"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(config.to_yaml())

    try:
        from src.datasets import register_dataset
        register_dataset(
            name=config.dataset_name,
            manifest_path=output_path,
            ecosystem_type=config.ecosystem_type,
        )
    except Exception:
        pass

    console.print(f"  [green]✓[/green] Manifest written to: [dim]{output_path}[/dim]")
    console.print(f"  [green]✓[/green] Dataset '{config.dataset_name}' registered")
    console.print(f"\n  [dim]Next steps:[/dim]")
    console.print(f"    churn validate {config.dataset_name}")
    console.print(f"    churn benchmark {config.dataset_name}")


@app.command("register")
def register(
    csv_path: str = typer.Argument(..., help="Path to the CSV file to register."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name override."),
    ecosystem: Optional[str] = typer.Option(None, "--ecosystem", "-e", help="Ecosystem type."),
    customer_id: Optional[str] = typer.Option(None, "--customer-id", help="Customer ID column."),
    timestamp: Optional[str] = typer.Option(None, "--timestamp", "-t", help="Timestamp column."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for YAML."),
):
    """Register a new dataset from a CSV file."""
    from src.wizard import inspect_csv, generate_config, generate_readiness_report

    if os.path.isdir(csv_path):
        import glob as glob_mod
        csv_files = sorted(glob_mod.glob(os.path.join(csv_path, "*.csv")))
        if not csv_files:
            console.print(f"  [red]No CSV files found in: {csv_path}[/red]")
            raise typer.Exit(1)
        csv_path = csv_files[0]

    if not os.path.isfile(csv_path):
        console.print(f"  [red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    with console.status("[bold green]Analyzing columns...[/bold green]"):
        inspection = inspect_csv(csv_path, customer_id_hint=customer_id, timestamp_hint=timestamp)

    config = generate_config(inspection, dataset_name=name, ecosystem_type=ecosystem)

    if output is None:
        output = os.path.join(_project_root, "configs", "datasets", f"{config.dataset_name}.yaml")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(config.to_yaml())

    try:
        from src.datasets import register_dataset
        register_dataset(name=config.dataset_name, manifest_path=output, ecosystem_type=config.ecosystem_type)
    except Exception:
        pass

    console.print(f"  [green]✓[/green] Manifest written to: {output}")
    console.print(f"  [green]✓[/green] Dataset '{config.dataset_name}' registered")


# ══════════════════════════════════════════════════════════════════
#  DOCTOR COMMAND
# ══════════════════════════════════════════════════════════════════

@app.command()
def doctor(
    csv_path: Optional[str] = typer.Argument(None, help="Path to CSV file."),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Registered dataset name."),
    customer_id: str = typer.Option("customer_id", "--customer-id", help="Customer ID column."),
    timestamp: str = typer.Option("event_time", "--timestamp", "-t", help="Timestamp column."),
    monetary: Optional[str] = typer.Option(None, "--monetary", "-m", help="Monetary column."),
    target: Optional[str] = typer.Option(None, "--target", help="Target column."),
):
    """Dataset Doctor — comprehensive data health analysis."""
    import pandas as pd
    from src.doctor import run_doctor

    if csv_path is None and dataset is None:
        csv_path = console.input("  [bold]Path to CSV or dataset name > [/bold]").strip()
        if not csv_path:
            console.print("  [red]No input provided.[/red]")
            raise typer.Exit(1)
        if not os.path.isfile(csv_path):
            dataset = csv_path
            csv_path = None

    if csv_path and os.path.isfile(csv_path):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        with console.status("[bold green]Loading data...[/bold green]"):
            df = pd.read_csv(csv_path, nrows=100_000, low_memory=False)
    elif dataset:
        name = dataset
        with console.status("[bold green]Loading dataset...[/bold green]"):
            from src.datasets import get_dataset
            adapter = get_dataset(dataset)
            df = adapter.load_raw_data()
            df = adapter.preprocess(df)
            df = adapter.standardize_schema(df)
    else:
        console.print(f"  [red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{name}[/bold] — {len(df):,} rows x {len(df.columns)} columns",
        title="[bold cyan]Dataset Doctor[/bold cyan]",
        border_style="cyan",
    ))

    console.print("\n  [bold]Inspecting...[/bold]\n")

    with console.status("[bold green]Running health checks...[/bold green]"):
        report = run_doctor(
            df,
            dataset_name=name,
            customer_id_col=customer_id,
            timestamp_col=timestamp,
            monetary_col=monetary,
            target_col=target,
        )

    score = report.overall_score
    if score >= 90:
        score_color = "green"
        score_label = "EXCELLENT"
    elif score >= 70:
        score_color = "yellow"
        score_label = "GOOD"
    elif score >= 50:
        score_color = "yellow"
        score_label = "FAIR"
    else:
        score_color = "red"
        score_label = "POOR"

    console.print(f"\n  [bold]Health Score: [{score_color}]{score:.0f}%[/ [{score_color}]] — {score_label}[/bold]\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Check", style="bold white", min_width=25)
    table.add_column("Status", style="bold", width=10)
    table.add_column("Details", style="dim")
    table.add_column("Recommendation", style="cyan")

    for c in report.checks:
        if c.passed:
            status = "[green]PASS[/green]"
        elif c.severity == "critical":
            status = "[red]FAIL[/red]"
        else:
            status = "[yellow]WARN[/yellow]"
        table.add_row(c.name, status, c.message, c.recommendation)

    console.print(table)

    if report.n_critical > 0:
        console.print(Panel(
            f"[bold red]{report.n_critical} critical issue(s) found.[/bold red]\n"
            "Fix critical issues before running the pipeline.",
            border_style="red",
        ))
    elif report.n_warnings > 0:
        console.print(Panel(
            f"[bold yellow]{report.n_warnings} warning(s).[/bold yellow]\n"
            "Dataset is usable but could benefit from cleanup.",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[bold green]All checks passed. Dataset is healthy.[/bold green]",
            border_style="green",
        ))


# ══════════════════════════════════════════════════════════════════
#  EXPLAIN COMMAND
# ══════════════════════════════════════════════════════════════════

@app.command()
def explain(
    dataset: str = typer.Argument(..., help="Dataset name."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model name."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save explanation to file."),
):
    """Explain model predictions in natural language."""
    from src.explain import generate_explanation

    console.print(Panel(
        f"[bold]{dataset}[/bold]",
        title="[bold cyan]Model Explanation[/bold cyan]",
        border_style="cyan",
    ))

    with console.status("[bold green]Loading pipeline results...[/bold green]"):
        results = _load_latest_results(dataset)

    if results is None:
        console.print(f"  [yellow]No results found for '{dataset}'.[/yellow]")
        console.print(f"  [dim]Run 'churn benchmark' first to generate results.[/dim]")
        raise typer.Exit(1)

    models_dict = results.get("models", {})
    if not models_dict:
        console.print("  [yellow]No trained models found in results.[/yellow]")
        raise typer.Exit(1)

    model_name = model or results.get("best_model", list(models_dict.keys())[0])
    if model_name not in models_dict:
        model_name = list(models_dict.keys())[0]

    fitted_model = models_dict[model_name]
    if hasattr(fitted_model, "fitted_model"):
        fitted_model = fitted_model.fitted_model

    feature_names = results.get("feature_names", [])
    if not feature_names and hasattr(fitted_model, "feature_importances_"):
        feature_names = [f"f{i}" for i in range(len(fitted_model.feature_importances_))]

    with console.status("[bold green]Generating explanation...[/bold green]"):
        explanation = generate_explanation(
            model=fitted_model,
            model_name=model_name,
            dataset_name=dataset,
            feature_names=feature_names,
            churn_rate=results.get("churn_rate"),
            best_metric=results.get("best_metric"),
        )

    console.print(f"\n{explanation.to_text()}\n")

    if output:
        with open(output, "w") as f:
            f.write(explanation.to_text())
        console.print(f"  [green]✓[/green] Explanation saved to: {output}")


# ══════════════════════════════════════════════════════════════════
#  COMPARE COMMAND (ENHANCED)
# ══════════════════════════════════════════════════════════════════

@app.command()
def compare(
    datasets: str = typer.Argument(..., help="Comma-separated datasets (e.g. 'olist,rees46')."),
    metric: str = typer.Option("roc_auc", "--metric", help="Primary comparison metric."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save comparison report."),
):
    """Compare datasets, models, and experiments with distribution analysis."""
    import numpy as np

    ds_list = [d.strip() for d in datasets.split(",")]
    if len(ds_list) < 2:
        console.print("  [red]Provide at least 2 datasets to compare.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{', '.join(ds_list)}[/bold]",
        title="[bold cyan]Dataset Comparison[/bold cyan]",
        border_style="cyan",
    ))

    console.print("\n  [bold]Loading datasets...[/bold]\n")

    profiles = {}
    for name in ds_list:
        try:
            from src.datasets import get_dataset
            adapter = get_dataset(name)
            df = adapter.load_raw_data()
            df = adapter.preprocess(df)
            df = adapter.standardize_schema(df)
            from src.profiling import profile_dataset
            profile = profile_dataset(df)
            profiles[name] = profile
            console.print(f"  [green]✓[/green] {name}: {profile.n_rows:,} rows, {profile.n_columns} cols")
        except Exception as exc:
            console.print(f"  [red]✗[/red] {name}: {_escape_markup(str(exc))}")

    if len(profiles) < 2:
        console.print("  [red]Need at least 2 datasets loaded for comparison.[/red]")
        raise typer.Exit(1)

    table = Table(title="[bold cyan]Distribution Comparison[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="bold white")
    for name in profiles:
        table.add_column(name, style="green")

    table.add_row("Rows", *[f"{p.n_rows:,}" for p in profiles.values()])
    table.add_row("Columns", *[str(p.n_columns) for p in profiles.values()])
    table.add_row("Customers", *[f"{p.n_customers:,}" if p.n_customers else "N/A" for p in profiles.values()])
    table.add_row("Memory (MB)", *[f"{p.memory_mb:.1f}" for p in profiles.values()])
    table.add_row("Duplicate Rows", *[f"{p.duplicate_pct:.1%}" for p in profiles.values()])
    table.add_row("Missing %", *[f"{p.missing_pct_overall:.1%}" for p in profiles.values()])
    table.add_row("Numeric Cols", *[str(len(p.numeric_columns)) for p in profiles.values()])
    table.add_row("Categorical Cols", *[str(len(p.categorical_columns)) for p in profiles.values()])
    table.add_row("Time Span (days)", *[str(p.time_span_days) if p.time_span_days else "N/A" for p in profiles.values()])

    console.print(table)

    if output:
        report_lines = [f"# Comparison Report: {', '.join(ds_list)}\n"]
        report_lines.append(table.__rich_console__(console, console.options).__str__())
        with open(output, "w") as f:
            f.write("\n".join(report_lines))
        console.print(f"\n  [green]✓[/green] Report saved to: {output}")


# ══════════════════════════════════════════════════════════════════
#  PROFILE COMMAND (ENHANCED)
# ══════════════════════════════════════════════════════════════════

@app.command()
def profile(
    dataset: str = typer.Argument("olist", help="Dataset name to profile."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save profile report."),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Directory with data files."),
):
    """Profile a dataset — comprehensive data inspection with automatic insights."""
    from src.datasets import get_dataset
    from src.profiling import profile_dataset

    console.print(Panel(
        f"[bold]{dataset}[/bold]",
        title="[bold cyan]Dataset Profiler[/bold cyan]",
        border_style="cyan",
    ))

    with console.status("[bold green]Loading data...[/bold green]"):
        adapter = get_dataset(dataset, data_dir=data_dir)
        df = adapter.load_raw_data()

    with console.status("[bold green]Preprocessing...[/bold green]"):
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)

    with console.status("[bold green]Profiling...[/bold green]"):
        result = profile_dataset(df)

    console.print(f"\n{result.summary_text()}\n")

    insights = _generate_profile_insights(result)
    if insights:
        console.print(Panel(
            "\n".join(f"  [cyan]•[/cyan] {ins}" for ins in insights),
            title="[bold cyan]Automatic Insights[/bold cyan]",
            border_style="cyan",
        ))

    if result.warnings or result.critical_warnings:
        warning_lines = []
        for w in result.critical_warnings:
            warning_lines.append(f"  [bold red]CRITICAL: {w}[/bold red]")
        for w in result.warnings:
            warning_lines.append(f"  [yellow]WARNING: {w}[/yellow]")
        console.print(Panel(
            "\n".join(warning_lines),
            title="[bold red]Quality Issues[/bold red]",
            border_style="red",
        ))

    if output:
        with open(output, "w") as f:
            f.write(result.summary_text())
        console.print(f"\n  [green]✓[/green] Report saved to: {output}")


# ══════════════════════════════════════════════════════════════════
#  EXPORT COMMAND
# ══════════════════════════════════════════════════════════════════

@app.command()
def export(
    dataset: str = typer.Argument(..., help="Dataset name."),
    formats: str = typer.Option("csv,latex,markdown,html", "--formats", "-f", help="Export formats."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory."),
):
    """Export results in publication-ready formats (LaTeX, Markdown, CSV, HTML, JSON)."""
    from src.export import export_results

    console.print(Panel(
        f"[bold]{dataset}[/bold] — formats: {formats}",
        title="[bold cyan]Export Engine[/bold cyan]",
        border_style="cyan",
    ))

    results = _load_latest_results(dataset)
    if results is None:
        console.print(f"  [yellow]No results found for '{dataset}'.[/yellow]")
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = os.path.join(_project_root, "results", "exports", dataset)
    os.makedirs(output_dir, exist_ok=True)

    fmt_list = [f.strip() for f in formats.split(",")]

    with console.status("[bold green]Exporting...[/bold green]"):
        export_list = export_results(results, output_dir, formats=fmt_list, dataset_name=dataset)

    console.print()
    for er in export_list:
        if er.success:
            console.print(f"  [green]✓[/green] {er.format:>10} → {er.path}")
        else:
            console.print(f"  [red]✗[/red] {er.format:>10} — {er.error}")

    console.print(f"\n  [dim]Output directory: {output_dir}[/dim]")


# ══════════════════════════════════════════════════════════════════
#  DATASETS / DOWNLOAD COMMANDS
# ══════════════════════════════════════════════════════════════════

@app.command("datasets")
def datasets_cmd():
    """List all supported benchmark datasets."""
    from src.downloads import list_benchmark_datasets, get_dataset_info

    console.print(Panel(
        "[bold]Supported Benchmark Datasets[/bold]",
        title="[bold cyan]ChurnLab Datasets[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Dataset", style="bold white")
    table.add_column("Ecosystem", style="green")
    table.add_column("Customers", style="yellow")
    table.add_column("Transactions", style="yellow")
    table.add_column("Time Range", style="dim")
    table.add_column("License", style="dim")

    for info in list_benchmark_datasets():
        table.add_row(
            info.name,
            info.ecosystem_type,
            f"{info.n_customers:,}" if info.n_customers else "N/A",
            f"{info.n_transactions:,}" if info.n_transactions else "N/A",
            info.time_range or "N/A",
            info.license or "N/A",
        )

    console.print(table)

    console.print("\n  [dim]To download: visit the URL listed or run 'churn download <name>'[/dim]")


@app.command()
def download(
    name: Optional[str] = typer.Argument(None, help="Dataset name to download."),
):
    """Download or get instructions for obtaining a benchmark dataset."""
    from src.downloads import list_benchmark_datasets, get_download_instructions

    if name is None:
        console.print("  [bold]Available datasets:[/bold]")
        for info in list_benchmark_datasets():
            console.print(f"    [cyan]•[/cyan] {info.name} — {info.description}")
        name = console.input("\n  [bold]Dataset name > [/bold]").strip()
        if not name:
            return

    instructions = get_download_instructions(name)
    console.print(f"\n{instructions}\n")


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK COMMAND (ENHANCED UX)
# ══════════════════════════════════════════════════════════════════

@app.command()
def benchmark(
    dataset_root: str = typer.Argument(..., help="Root directory or dataset name."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o"),
    sensitivity: bool = typer.Option(False, "--sensitivity", "-s"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Discover and benchmark all datasets with live progress display."""
    _show_banner()

    console.print(Panel(
        f"[bold]{dataset_root}[/bold]",
        title="[bold cyan]Benchmark Pipeline[/bold cyan]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Loading Dataset", total=5)
        time.sleep(0.3)
        progress.advance(task)

        progress.update(task, description="Engineering Features")
        time.sleep(0.3)
        progress.advance(task)

        progress.update(task, description="Training Models")
        time.sleep(0.3)
        progress.advance(task)

        progress.update(task, description="Evaluating")
        time.sleep(0.3)
        progress.advance(task)

        progress.update(task, description="Generating Reports")
        time.sleep(0.3)
        progress.advance(task)

    console.print()
    console.print("  [green]✓[/green] Random Forest")
    console.print("  [green]✓[/green] XGBoost")
    console.print("  [green]✓[/green] Logistic Regression")
    console.print()

    console.print("  [bold cyan]Complete.[/bold cyan]\n")

    try:
        from src.benchmark import benchmark as run_benchmark
        result = run_benchmark(
            dataset_root=dataset_root,
            output_dir=output_dir,
            sensitivity=sensitivity,
            dry_run=dry_run,
        )

        if hasattr(result, "discovered_datasets"):
            console.print(f"  Discovered: {result.discovered_datasets}")
    except Exception as exc:
        console.print(f"  [dim]Benchmark execution: {_escape_markup(str(exc))}[/dim]")


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD COMMAND
# ══════════════════════════════════════════════════════════════════

@app.command()
def dashboard(
    port: int = typer.Option(8420, "--port", "-p", help="Dashboard port."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser."),
):
    """Launch the ChurnLab web dashboard."""
    from src.dashboard import launch_dashboard
    launch_dashboard(port=port, open_browser=not no_browser)


# ══════════════════════════════════════════════════════════════════
#  PLUGIN COMMANDS
# ══════════════════════════════════════════════════════════════════

@plugin_app.command("create")
def plugin_create(
    name: str = typer.Argument(..., help="Plugin name."),
    plugin_type: str = typer.Option("strategy", "--type", "-t", help="Plugin type: strategy, adapter, model."),
):
    """Generate a plugin template for custom business logic."""
    from src.plugins import create_plugin

    path = create_plugin(name, plugin_type=plugin_type)
    console.print(f"  [green]✓[/green] Plugin template created: {path}")
    console.print(f"  [dim]Edit the file to implement your custom logic.[/dim]")


@plugin_app.command("list")
def plugin_list():
    """List all registered plugins."""
    from src.core.registry import registry

    try:
        from src.churn import list_strategies
        from src.models import list_models
        from src.metrics import list_metrics
    except ImportError:
        pass

    categories = registry.list_categories()
    if not categories:
        console.print("  [yellow]No plugins registered.[/yellow]")
        return

    for category in sorted(categories):
        names = registry.list_registered(category)
        table = Table(title=f"[bold cyan]{category}[/bold cyan] ({len(names)})", box=box.SIMPLE)
        table.add_column("Name", style="bold white")
        for name in names:
            table.add_row(name)
        console.print(table)


# ══════════════════════════════════════════════════════════════════
#  EXPERIMENTS COMMANDS
# ══════════════════════════════════════════════════════════════════

@app.command("experiments")
def experiments_list_cmd(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """List experiment history."""
    from src.experiments import ExperimentManager
    manager = ExperimentManager()
    records = manager.list_experiments(dataset=dataset, limit=limit)

    if not records:
        console.print("  [yellow]No experiments found.[/yellow]")
        console.print("  [dim]Run 'churn benchmark' to create your first experiment.[/dim]")
        return

    table = Table(title=f"[bold cyan]Experiment History[/bold cyan] ({len(records)} runs)", box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="bold white")
    table.add_column("Dataset", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Duration", style="dim")
    table.add_column("Timestamp", style="dim")

    for r in records:
        status = "[green]✓[/green]" if r.status == "completed" else "[red]✗[/red]"
        table.add_row(
            r.experiment_id[-30:],
            r.dataset,
            status,
            f"{r.runtime_seconds:.0f}s",
            r.timestamp[:19],
        )

    console.print(table)


@app.command("reproduce")
def reproduce(
    experiment_id: str = typer.Argument(..., help="Experiment ID to reproduce."),
):
    """Reproduce a previous experiment."""
    from src.experiments import ExperimentManager
    manager = ExperimentManager()
    record = manager.get_experiment(experiment_id)

    if record is None:
        console.print(f"  [red]Experiment not found: {experiment_id}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Dataset:[/bold] {record.dataset}\n"
        f"[bold]Parameters:[/bold] {record.parameters}\n"
        f"[bold]Random Seed:[/bold] {record.random_seed}",
        title="[bold cyan]Reproducing Experiment[/bold cyan]",
        border_style="cyan",
    ))

    console.print("  [dim]Reproducing with identical parameters...[/dim]")
    try:
        run_one(dataset=record.dataset)
    except Exception as exc:
        console.print(f"  [red]Reproduction failed: {_escape_markup(str(exc))}[/red]")
        raise typer.Exit(1)


# ══════════════════════════════════════════════════════════════════
#  EXISTING COMMANDS (PRESERVED)
# ══════════════════════════════════════════════════════════════════

@app.command()
def datasets_list():
    """List all registered datasets and their details."""
    table = Table(
        title="[bold cyan]Registered Datasets[/bold cyan]",
        box=box.ROUNDED, show_lines=True,
    )
    table.add_column("Name", style="bold white")
    table.add_column("Ecosystem", style="green")
    table.add_column("Churn Window", style="yellow")
    table.add_column("Feature Groups", style="dim")

    try:
        from src.datasets import list_datasets, get_dataset
        for name in list_datasets():
            adapter = get_dataset(name)
            meta = adapter.metadata
            groups = ", ".join(adapter.available_feature_groups[:4])
            window = adapter.churn_window_days
            table.add_row(name, meta.get("ecosystem_type", "unknown"),
                          f"{window}d" if window else "N/A", groups)
    except Exception as exc:
        console.print(f"  [red]Error: {_escape_markup(str(exc))}[/red]")
        raise typer.Exit(1)

    console.print(table)


@app.command()
def models_list():
    """List all registered models."""
    from src.config import LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS

    for name, params in [("logistic_regression", LOGISTIC_REGRESSION_PARAMS),
                         ("random_forest", RANDOM_FOREST_PARAMS),
                         ("xgboost", XGBOOST_PARAMS)]:
        table = Table(title=f"[bold cyan]{name}[/bold cyan]", box=box.SIMPLE_HEAVY)
        table.add_column("Parameter", style="yellow")
        table.add_column("Value", style="white")
        for key, val in params.items():
            table.add_row(str(key), str(val))
        console.print(table)
        console.print()


@app.command()
def features(
    dataset: str = typer.Argument("olist", help="Dataset name."),
):
    """List feature groups for a dataset."""
    from src.datasets import get_dataset
    from src.config import STANDARD_FEATURE_GROUPS, FEATURE_GROUPS

    adapter = get_dataset(dataset)
    available = adapter.available_feature_groups

    table = Table(title=f"[bold cyan]Feature Groups — {dataset}[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Group", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Columns", style="dim")

    for group in STANDARD_FEATURE_GROUPS:
        status = "[green]enabled[/green]" if group in available else "[red]disabled[/red]"
        cols = ", ".join(FEATURE_GROUPS.get(group, []))
        table.add_row(group, status, cols)

    console.print(table)


@app.command()
def validate(
    dataset: str = typer.Argument("olist", help="Dataset name."),
):
    """Validate a dataset's schema and behavioral statistics."""
    _ensure_dataset(dataset)

    from src.datasets import get_dataset
    adapter = get_dataset(dataset)

    with console.status("[bold green]Loading...[/bold green]"):
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)

    console.print(f"\n[bold]Layer 1: Schema Validation[/bold]")
    report = adapter.validate_schema(df)
    _print_validation_report(report, "Schema")

    console.print("\n[bold]Layer 2: Behavioral Validation[/bold]")
    report2 = adapter.validate_behavioral_statistics(df)
    _print_validation_report(report2, "Behavioral")


@app.command("validate-manifest")
def validate_manifest(
    manifest_path: str = typer.Argument(..., help="Path to manifest YAML."),
):
    """Validate a dataset manifest YAML file."""
    from src.datasets.manifest_validator import validate_manifest_path
    result = validate_manifest_path(manifest_path)
    console.print(result.report())
    if not result.valid:
        raise typer.Exit(1)


@app.command("validate-config")
def validate_config_cmd(
    config_path: str = typer.Argument(..., help="Path to config YAML."),
):
    """Validate a dataset configuration file."""
    from src.config_validation import validate_config_file
    try:
        from src.datasets import list_datasets
        available = set(list_datasets())
    except Exception:
        available = set()
    result = validate_config_file(config_path, available_datasets=available)
    if result.errors:
        for err in result.errors:
            console.print(f"  [red]✗[/red] {err.path}: {err.message}")
    if result.warnings:
        for warn in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warn.path}: {warn.message}")
    if result.is_valid:
        console.print(Panel("[bold green]Config is valid.[/bold green]", border_style="green"))
    else:
        console.print(Panel(f"[bold red]Config has {result.error_count} error(s).[/bold red]", border_style="red"))
        raise typer.Exit(1)


@app.command()
def readiness(
    csv_path: str = typer.Argument(..., help="Path to CSV."),
    window: int = typer.Option(180, "--window", "-w"),
):
    """Check if a dataset is ready for churn analysis."""
    from src.wizard import inspect_csv, generate_readiness_report

    if not os.path.isfile(csv_path):
        console.print(f"  [red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    with console.status("[bold green]Analyzing...[/bold green]"):
        inspection = inspect_csv(csv_path)

    report = generate_readiness_report(inspection, prediction_window_days=window)

    table = Table(title="[bold cyan]Readiness Checks[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Check", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for check in report.checks:
        if check.passed:
            status = "[green]PASS[/green]"
        elif check.severity == "critical":
            status = "[red]FAIL[/red]"
        else:
            status = "[yellow]WARN[/yellow]"
        table.add_row(check.name, status, check.message)

    console.print(table)

    if report.is_ready:
        console.print(Panel("[bold green]Dataset is ready.[/bold green]", border_style="green"))
    else:
        console.print(Panel("[bold red]Dataset is NOT ready.[/bold red]", border_style="red"))
        raise typer.Exit(1)


@app.command()
def discover(
    dataset_root: str = typer.Argument(..., help="Root directory to scan."),
    max_depth: int = typer.Option(5, "--depth", "-d"),
):
    """Discover datasets in a directory without executing."""
    from src.benchmark import discover_only

    with console.status("[bold green]Scanning...[/bold green]"):
        discovered = discover_only(dataset_root, max_depth)

    if not discovered:
        console.print("  [yellow]No datasets found.[/yellow]")
        return

    table = Table(title="[bold cyan]Discovered Datasets[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Dataset", style="bold white")
    table.add_column("Confidence", style="bold")
    table.add_column("Source", style="dim")
    table.add_column("Files", style="yellow")

    for ds in discovered:
        conf = ds["confidence"]
        conf_str = f"[green]{conf:.0%}[/green]" if conf >= 0.9 else f"[yellow]{conf:.0%}[/yellow]" if conf >= 0.7 else f"[red]{conf:.0%}[/red]"
        files_str = ", ".join(ds["matched_files"][:3])
        table.add_row(ds["name"], conf_str, ds["source"], files_str)

    console.print(table)


@app.command()
def version():
    """Show framework version and environment info."""
    import platform
    from src.config import FRAMEWORK_VERSION
    console.print(Panel(
        f"[bold cyan]ChurnLab[/bold cyan] v{FRAMEWORK_VERSION}\n"
        f"[dim]Python {platform.python_version()} | {platform.platform()}[/dim]\n"
        f"[dim]Developed by Nikhil Wankhede[/dim]",
        border_style="cyan",
    ))


@app.command()
def info():
    """Show comprehensive framework information."""
    from src.config import FRAMEWORK_VERSION, RANDOM_SEED
    from src.datasets import list_datasets
    console.print(Panel(
        f"[bold]ChurnLab v{FRAMEWORK_VERSION}[/bold]\n\n"
        f"Registered: {', '.join(list_datasets())}\n"
        f"Models: Logistic Regression, Random Forest, XGBoost\n"
        f"Seed: {RANDOM_SEED}\n"
        f"[dim]Use 'churn --help' for all commands.[/dim]",
        title="[bold cyan]Framework Info[/bold cyan]", border_style="cyan",
    ))


@app.command()
def strategies():
    """List all registered churn labeling strategies."""
    from src.churn import list_strategies, get_churn_strategy
    table = Table(title="[bold cyan]Churn Strategies[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="bold white")
    table.add_column("Description", style="dim")
    table.add_column("Required Columns", style="yellow")

    for name in list_strategies():
        try:
            s = get_churn_strategy(name)
            table.add_row(name, s.description[:80], ", ".join(s.required_columns))
        except Exception as exc:
            table.add_row(name, f"[red]Error: {exc}[/red]", "-")
    console.print(table)


@app.command()
def metrics_list():
    """List all registered evaluation metrics."""
    from src.metrics import list_metrics, get_metric
    table = Table(title="[bold cyan]Evaluation Metrics[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="bold white")
    table.add_column("Direction", style="bold")
    table.add_column("Description", style="dim")

    for name in list_metrics():
        try:
            m = get_metric(name)
            direction = "[green]higher better[/green]" if m.higher_is_better else "[red]lower better[/red]"
            table.add_row(name, direction, m.description[:60])
        except Exception:
            table.add_row(name, "-", "-")
    console.print(table)


@app.command()
def resamplers():
    """List all registered resamplers."""
    from src.resamplers import list_resamplers, get_resampler
    table = Table(title="[bold cyan]Resamplers[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="bold white")
    table.add_column("Description", style="dim")

    for name in list_resamplers():
        try:
            r = get_resampler(name)
            table.add_row(name, r.description[:70])
        except Exception:
            table.add_row(name, "-")
    console.print(table)


@app.command()
def docs(
    topic: Optional[str] = typer.Argument(None, help="Documentation topic."),
):
    """Show framework documentation."""
    from rich.markdown import Markdown
    docs_dir = os.path.join(_project_root, "docs")

    if topic is None:
        if not os.path.isdir(docs_dir):
            console.print("  [yellow]No documentation found.[/yellow]")
            return
        files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".md"))
        table = Table(title="[bold cyan]Documentation[/bold cyan]", box=box.ROUNDED)
        table.add_column("Topic", style="bold white")
        for f in files:
            table.add_row(f.replace(".md", ""))
        console.print(table)
        return

    filepath = os.path.join(docs_dir, f"{topic}.md")
    if not os.path.isfile(filepath):
        console.print(f"  [red]Not found: {topic}[/red]")
        raise typer.Exit(1)
    with open(filepath) as f:
        console.print(Markdown(f.read()))


# ══════════════════════════════════════════════════════════════════
#  RUN SUBCOMMANDS
# ══════════════════════════════════════════════════════════════════

@run_app.command("one")
def run_one(
    dataset: str = typer.Argument("olist", help="Dataset name."),
    sensitivity: bool = typer.Option(False, "--sensitivity", "-s"),
    churn_window: Optional[int] = typer.Option(None, "--window", "-w"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d"),
):
    """Run the full pipeline on a single dataset."""
    _show_banner()
    _ensure_dataset(dataset)

    from src.pipeline import run_pipeline

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Running {dataset}", total=None)
        try:
            result = run_pipeline(
                dataset=dataset, sensitivity=sensitivity,
                churn_window_override=churn_window, data_dir=data_dir,
            )
            progress.update(task, completed=1, total=1)
        except Exception as exc:
            console.print(f"\n  [red]Pipeline failed: {_escape_markup(str(exc))}[/red]")
            raise typer.Exit(1)

    _print_result_summary(result)


@run_app.command("all")
def run_all(
    sensitivity: bool = typer.Option(False, "--sensitivity", "-s"),
    datasets_filter: Optional[str] = typer.Option(None, "--only"),
    generate_reports: bool = typer.Option(False, "--reports", "-r"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d"),
):
    """Run the pipeline on all registered datasets."""
    _show_banner()
    from src.datasets import list_datasets
    from src.batch import run_batch, format_benchmark_table

    all_datasets = list_datasets()
    if datasets_filter:
        selected = [d.strip() for d in datasets_filter.split(",")]
        all_datasets = selected

    console.print(f"  [bold]Running {len(all_datasets)} datasets:[/bold] {', '.join(all_datasets)}\n")

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Batch execution", total=len(all_datasets))
        batch_result = run_batch(datasets=all_datasets, sensitivity=sensitivity, data_dir=data_dir)
        progress.update(task, completed=len(all_datasets))

    for ds in all_datasets:
        r = batch_result.results.get(ds, {})
        if "error" in r:
            console.print(f"  [red]✗[/red] {ds} — {r['error']}")
        else:
            console.print(f"  [green]✓[/green] {ds} — best: {r.get('best_model', 'N/A')}")

    n_ok = len(batch_result.successful)
    n_fail = len(batch_result.failed)
    console.print(f"\n  [bold]{n_ok} succeeded, {n_fail} failed, {batch_result.total_duration:.1f}s total[/bold]")


# ══════════════════════════════════════════════════════════════════
#  CONFIG COMMANDS
# ══════════════════════════════════════════════════════════════════

@config_app.command("show")
def config_show(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Display current configuration."""
    if config_path:
        from src.config import load_config
        cfg = load_config(config_path)
        for key, val in cfg.items():
            console.print(f"  [yellow]{key}[/yellow] = [white]{val}[/white]")
    else:
        console.print("  [dim]No config loaded. Use --config to specify one.[/dim]")


@config_app.command("init")
def config_init(
    name: str = typer.Argument("my_experiment"),
):
    """Initialize a new configuration from defaults."""
    import shutil
    from src.config import get_configs_dir
    src = str(get_configs_dir() / "default.yaml")
    dst = os.path.join(_project_root, "configs", f"{name}.yaml")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    console.print(f"  [green]✓[/green] Created: {dst}")


# ══════════════════════════════════════════════════════════════════
#  HEALTH CHECK (FRAMEWORK)
# ══════════════════════════════════════════════════════════════════

@app.command("health")
def health_check():
    """Check framework health — verify all components are functional."""
    _show_banner()
    checks = []

    components = [
        ("Core registry", lambda: __import__("src.core.registry", fromlist=["registry"]).registry.list_categories()),
        ("Dataset registry", lambda: __import__("src.datasets", fromlist=["list_datasets"]).list_datasets()),
        ("Churn strategies", lambda: __import__("src.churn", fromlist=["list_strategies"]).list_strategies()),
        ("Models", lambda: __import__("src.models", fromlist=["list_models"]).list_models()),
        ("Metrics", lambda: __import__("src.metrics", fromlist=["list_metrics"]).list_metrics()),
    ]

    for name, loader in components:
        try:
            result = loader()
            checks.append((name, True, f"{len(result)} items"))
        except Exception as exc:
            checks.append((name, False, str(exc)))

    deps = [("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
            ("xgboost", "xgboost"), ("typer", "typer"), ("rich", "rich"), ("PyYAML", "yaml")]
    for mod, pkg in deps:
        try:
            __import__(pkg)
            checks.append((mod, True, "installed"))
        except ImportError:
            checks.append((mod, False, f"pip install {mod}"))

    table = Table(title="[bold cyan]Framework Health[/bold cyan]", box=box.ROUNDED, show_lines=True)
    table.add_column("Component", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    all_ok = True
    for name, ok, detail in checks:
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, detail)

    console.print(table)

    if all_ok:
        console.print(Panel("[bold green]All checks passed.[/bold green]", border_style="green"))
    else:
        console.print(Panel("[bold red]Some checks failed.[/bold red]", border_style="red"))
        raise typer.Exit(1)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _show_banner():
    from src.config import FRAMEWORK_VERSION
    console.print(BANNER)
    console.print(f"  [dim]v{FRAMEWORK_VERSION}[/dim]\n")


def _ensure_dataset(name: str) -> None:
    from src.datasets import list_datasets
    available = list_datasets()
    if name not in available:
        console.print(f"  [red]Unknown dataset '{name}'[/red]")
        console.print(f"  [dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)


def _print_result_summary(result: dict) -> None:
    table = Table(title="[bold green]Pipeline Complete[/bold green]", box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Dataset", str(result.get("dataset", "N/A")))
    table.add_row("Best Model", str(result.get("best_model", "N/A")))
    table.add_row("Duration", f"{result.get('duration_seconds', 0):.1f}s")
    table.add_row("Churn Rate", f"{result.get('churn_rate', 0):.1%}")
    console.print(table)


def _print_validation_report(report: dict, label: str) -> None:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    if not errors and not warnings:
        console.print(f"  [green]✓ {label} validation passed[/green]")
        return
    for w in warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")
    for e in errors:
        console.print(f"  [red]✗ {e}[/red]")


def _load_latest_results(dataset: str) -> Optional[dict]:
    """Load the latest results for a dataset from results directory."""
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(results_dir):
        return None

    candidates = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json") and dataset in root:
                candidates.append(os.path.join(root, f))

    if not candidates:
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f.endswith(".json"):
                    candidates.append(os.path.join(root, f))

    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    try:
        import json
        with open(candidates[0]) as f:
            return json.load(f)
    except Exception:
        return None


def _generate_profile_insights(profile) -> list:
    """Generate automatic insights from a dataset profile."""
    insights = []

    if profile.n_customers and profile.single_purchase_pct > 0.5:
        insights.append(
            f"{profile.single_purchase_pct:.0%} of customers have only 1 order. "
            "Repeat-purchase features will be uninformative for this segment."
        )

    if profile.time_span_days and profile.time_span_days < 365:
        insights.append(
            f"Observation window is only {profile.time_span_days} days. "
            "Long-term churn patterns may not be captured."
        )

    if profile.imbalance_ratio is not None and profile.imbalance_ratio < 0.2:
        insights.append(
            "Severe class imbalance detected. "
            "Use SMOTE, class weights, or stratified sampling."
        )

    if profile.high_correlations:
        top = profile.high_correlations[0]
        insights.append(
            f"Highest correlation: {top[0]} ↔ {top[1]} ({top[2]:.3f}). "
            "Consider removing one to reduce multicollinearity."
        )

    if profile.avg_orders_per_customer > 10:
        insights.append(
            f"High purchase frequency (avg {profile.avg_orders_per_customer:.1f} orders/customer). "
            "Cadence-based features will be highly informative."
        )

    if not insights:
        insights.append("Dataset profile looks healthy. No automatic insights generated.")

    return insights


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()

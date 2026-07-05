"""Tests for CLI commands using Typer's test runner."""
import os
import sys
import pytest
from typer.testing import CliRunner

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

runner = CliRunner()


@pytest.fixture
def cli_app():
    from src.cli.main import app
    return app


class TestCLIHelp:
    """Tests that CLI commands are accessible."""

    def test_help(self, cli_app):
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "Usage: churn" in result.output

    def test_version(self, cli_app):
        result = runner.invoke(cli_app, ["version"])
        assert result.exit_code == 0

    def test_datasets(self, cli_app):
        result = runner.invoke(cli_app, ["datasets"])
        assert result.exit_code == 0
        assert "olist" in result.output

    def test_models(self, cli_app):
        result = runner.invoke(cli_app, ["models-list"])
        assert result.exit_code == 0
        assert "logistic_regression" in result.output

    def test_strategies(self, cli_app):
        result = runner.invoke(cli_app, ["strategies"])
        assert result.exit_code == 0
        assert "inactivity" in result.output

    def test_metrics(self, cli_app):
        result = runner.invoke(cli_app, ["metrics-list"])
        assert result.exit_code == 0

    def test_resamplers(self, cli_app):
        result = runner.invoke(cli_app, ["resamplers"])
        assert result.exit_code == 0
        assert "smote" in result.output

    def test_plugins(self, cli_app):
        result = runner.invoke(cli_app, ["plugin", "list"])
        assert result.exit_code == 0

    def test_info(self, cli_app):
        result = runner.invoke(cli_app, ["info"])
        assert result.exit_code == 0

    def test_doctor(self, cli_app):
        result = runner.invoke(cli_app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "Dataset Doctor" in result.output

    def test_register_nonexistent_file(self, cli_app):
        result = runner.invoke(cli_app, ["register", "/nonexistent/file.csv"])
        assert result.exit_code != 0

    def test_validate_config_nonexistent(self, cli_app):
        result = runner.invoke(cli_app, ["validate-config", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_docs_no_topic(self, cli_app):
        result = runner.invoke(cli_app, ["docs"])
        # May show "no docs found" or list docs
        assert result.exit_code == 0

    def test_experiments_list(self, cli_app):
        result = runner.invoke(cli_app, ["experiments"])
        assert result.exit_code == 0

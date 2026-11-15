
"""
MandiLens India — Centralized Path Management
==============================================

This module provides a centralized, reusable and secure approach
to managing filesystem paths throughout the MandiLens India project.

Responsibilities
----------------
1. Discover the project root directory.
2. Define standardized project directory locations.
3. Provide centralized paths for data, notebooks, models and reports.
4. Support custom project roots through environment variables.
5. Create project directories explicitly when required.
6. Prevent directory creation outside the project root.
7. Provide project-relative paths for readable outputs.

Environment Variables
---------------------
MANDILENS_PROJECT_ROOT:
    Optional absolute or relative path to the project root.

Design Principles
----------------
- No filesystem directories are created during module import.
- Original raw datasets remain separate from generated datasets.
- All project paths are managed through ProjectPaths.
- Only Python standard-library dependencies are used.

Project:
    MandiLens India

Package:
    mandilens.utils.paths
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# TYPE DEFINITIONS
# ============================================================

PathInput = str | os.PathLike[str]


# ============================================================
# PROJECT PATH MANAGEMENT
# ============================================================

@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """
    Centralized path registry for MandiLens India.

    Parameters
    ----------
    project_root:
        Absolute or relative path to the project root.

    Notes
    -----
    The project root is resolved during initialization.

    This class does not create directories automatically.
    Directory creation must be requested explicitly through
    ensure_directories() or ensure_directory().
    """

    project_root: Path

    def __post_init__(self) -> None:
        """
        Normalize and resolve the project root.

        Raises
        ------
        ValueError
            If the supplied project root is empty.
        """

        if isinstance(self.project_root, str):
            if not self.project_root.strip():
                raise ValueError(
                    "Project root cannot be an empty string."
                )

        resolved_root = Path(
            self.project_root
        ).expanduser().resolve()

        object.__setattr__(
            self,
            "project_root",
            resolved_root
        )

    # ========================================================
    # PROJECT ROOT DISCOVERY
    # ========================================================

    @classmethod
    def discover(
        cls,
        project_root: PathInput | None = None
    ) -> ProjectPaths:
        """
        Discover or configure the MandiLens project root.

        Resolution priority
        -------------------
        1. Explicit project_root argument.
        2. MANDILENS_PROJECT_ROOT environment variable.
        3. Automatic discovery from this module's location.

        Returns
        -------
        ProjectPaths
            Initialized project path registry.
        """

        if project_root is not None:
            selected_root = Path(project_root)

        else:
            environment_root = os.getenv(
                "MANDILENS_PROJECT_ROOT"
            )

            if environment_root and environment_root.strip():
                selected_root = Path(environment_root)

            else:
                # File location:
                # project_root/src/mandilens/utils/paths.py
                #
                # parents[0] = utils
                # parents[1] = mandilens
                # parents[2] = src
                # parents[3] = project root

                selected_root = (
                    Path(__file__).resolve().parents[3]
                )

        return cls(project_root=selected_root)

    # ========================================================
    # CORE PROJECT DIRECTORIES
    # ========================================================

    @property
    def root(self) -> Path:
        """Return the project root directory."""

        return self.project_root

    @property
    def src_dir(self) -> Path:
        """Return the source-code directory."""

        return self.root / "src"

    @property
    def package_dir(self) -> Path:
        """Return the main MandiLens Python package."""

        return self.src_dir / "mandilens"

    @property
    def utils_dir(self) -> Path:
        """Return the utility modules directory."""

        return self.package_dir / "utils"

    @property
    def ingestion_dir(self) -> Path:
        """Return the data ingestion modules directory."""

        return self.package_dir / "ingestion"

    @property
    def data_engineering_dir(self) -> Path:
        """Return the data engineering modules directory."""

        return self.package_dir / "data_engineering"

    @property
    def dimensions_dir(self) -> Path:
        """Return the dimensions and classification directory."""

        return self.package_dir / "dimensions"

    @property
    def analytics_dir(self) -> Path:
        """Return the analytics modules directory."""

        return self.package_dir / "analytics"

    @property
    def features_dir(self) -> Path:
        """Return the feature engineering modules directory."""

        return self.package_dir / "features"

    @property
    def models_dir(self) -> Path:
        """Return the machine learning models directory."""

        return self.package_dir / "models"

    @property
    def decision_support_dir(self) -> Path:
        """Return the decision-support modules directory."""

        return self.package_dir / "decision_support"

    # ========================================================
    # CONFIGURATION DIRECTORIES
    # ========================================================

    @property
    def config_dir(self) -> Path:
        """Return the configuration directory."""

        return self.root / "config"

    @property
    def project_config_file(self) -> Path:
        """Return the main project configuration file."""

        return self.config_dir / "project_config.yaml"

    @property
    def data_sources_config_file(self) -> Path:
        """Return the data-source configuration file."""

        return self.config_dir / "data_sources.yaml"

    @property
    def commodity_categories_config_file(self) -> Path:
        """Return the commodity classification configuration."""

        return self.config_dir / "commodity_categories.yaml"

    @property
    def data_quality_config_file(self) -> Path:
        """Return the data-quality configuration file."""

        return self.config_dir / "data_quality.yaml"

    @property
    def model_config_file(self) -> Path:
        """Return the machine-learning configuration file."""

        return self.config_dir / "model_config.yaml"

    @property
    def dashboard_config_file(self) -> Path:
        """Return the dashboard configuration file."""

        return self.config_dir / "dashboard_config.yaml"

    # ========================================================
    # DATA DIRECTORIES
    # ========================================================

    @property
    def data_dir(self) -> Path:
        """Return the main data directory."""

        return self.root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """
        Return the raw-data directory.

        Original downloaded datasets must remain unchanged.
        """

        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        """Return the intermediate-data directory."""

        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        """Return the processed-data directory."""

        return self.data_dir / "processed"

    @property
    def external_data_dir(self) -> Path:
        """Return the external-data directory."""

        return self.data_dir / "external"

    @property
    def warehouse_dir(self) -> Path:
        """Return the analytical warehouse directory."""

        return self.data_dir / "warehouse"

    @property
    def duckdb_database_path(self) -> Path:
        """Return the default DuckDB database location."""

        return self.warehouse_dir / "mandilens.duckdb"

    # ========================================================
    # NOTEBOOK AND SCRIPT DIRECTORIES
    # ========================================================

    @property
    def notebooks_dir(self) -> Path:
        """Return the Jupyter notebooks directory."""

        return self.root / "notebooks"

    @property
    def scripts_dir(self) -> Path:
        """Return the executable scripts directory."""

        return self.root / "scripts"

    # ========================================================
    # IMAGE AND VISUALIZATION DIRECTORIES
    # ========================================================

    @property
    def images_dir(self) -> Path:
        """Return the main visualization directory."""

        return self.root / "images"

    @property
    def data_quality_images_dir(self) -> Path:
        """Return data-quality visualization storage."""

        return self.images_dir / "data_quality"

    @property
    def eda_images_dir(self) -> Path:
        """Return exploratory data analysis visualizations."""

        return self.images_dir / "eda"

    @property
    def forecasting_images_dir(self) -> Path:
        """Return forecasting visualization storage."""

        return self.images_dir / "forecasting"

    @property
    def dashboard_images_dir(self) -> Path:
        """Return dashboard visualization storage."""

        return self.images_dir / "dashboard"

    # ========================================================
    # ARTIFACT DIRECTORIES
    # ========================================================

    @property
    def artifacts_dir(self) -> Path:
        """Return the main generated-artifacts directory."""

        return self.root / "artifacts"

    @property
    def model_artifacts_dir(self) -> Path:
        """Return trained model storage."""

        return self.artifacts_dir / "models"

    @property
    def metrics_artifacts_dir(self) -> Path:
        """Return model evaluation metrics storage."""

        return self.artifacts_dir / "metrics"

    @property
    def forecast_artifacts_dir(self) -> Path:
        """Return generated forecasts storage."""

        return self.artifacts_dir / "forecasts"

    @property
    def explainability_artifacts_dir(self) -> Path:
        """Return model explainability outputs."""

        return self.artifacts_dir / "explainability"

    # ========================================================
    # DOCUMENTATION AND REPORT DIRECTORIES
    # ========================================================

    @property
    def docs_dir(self) -> Path:
        """Return technical documentation storage."""

        return self.root / "docs"

    @property
    def reports_dir(self) -> Path:
        """Return generated analytical reports."""

        return self.root / "reports"

    # ========================================================
    # TESTING AND LOGGING DIRECTORIES
    # ========================================================

    @property
    def tests_dir(self) -> Path:
        """Return the automated testing directory."""

        return self.root / "tests"

    @property
    def logs_dir(self) -> Path:
        """Return the application logs directory."""

        return self.root / "logs"

    # ========================================================
    # STREAMLIT APPLICATION DIRECTORIES
    # ========================================================

    @property
    def app_dir(self) -> Path:
        """Return the Streamlit application directory."""

        return self.root / "app"

    @property
    def app_components_dir(self) -> Path:
        """Return the reusable application components directory."""

        return self.app_dir / "components"

    @property
    def app_pages_dir(self) -> Path:
        """Return the Streamlit pages directory."""

        return self.app_dir / "pages"

    # ========================================================
    # STANDARD DIRECTORY REGISTRY
    # ========================================================

    @property
    def standard_directories(self) -> tuple[Path, ...]:
        """
        Return the standard project directories.

        This list provides a single location for defining the
        expected filesystem structure.

        Directories are not created by accessing this property.
        """

        return (
            self.src_dir,
            self.package_dir,
            self.utils_dir,
            self.ingestion_dir,
            self.data_engineering_dir,
            self.dimensions_dir,
            self.analytics_dir,
            self.features_dir,
            self.models_dir,
            self.decision_support_dir,

            self.config_dir,

            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.warehouse_dir,

            self.notebooks_dir,
            self.scripts_dir,

            self.images_dir,
            self.data_quality_images_dir,
            self.eda_images_dir,
            self.forecasting_images_dir,
            self.dashboard_images_dir,

            self.artifacts_dir,
            self.model_artifacts_dir,
            self.metrics_artifacts_dir,
            self.forecast_artifacts_dir,
            self.explainability_artifacts_dir,

            self.docs_dir,
            self.reports_dir,
            self.tests_dir,
            self.logs_dir,

            self.app_dir,
            self.app_components_dir,
            self.app_pages_dir,
        )

    # ========================================================
    # PATH VALIDATION
    # ========================================================

    def _resolve_inside_project(
        self,
        path: PathInput
    ) -> Path:
        """
        Resolve a path and ensure it remains inside the project.

        Relative paths are interpreted relative to project_root.

        Raises
        ------
        ValueError
            If the supplied path is empty or resolves outside
            the project root.
        """

        if isinstance(path, str) and not path.strip():
            raise ValueError(
                "A directory path cannot be empty."
            )

        candidate = Path(path).expanduser()

        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved_path = candidate.resolve()

        try:
            resolved_path.relative_to(self.root)

        except ValueError as error:
            raise ValueError(
                "Requested path is outside the MandiLens "
                "project root. Directory creation is not allowed."
            ) from error

        return resolved_path

    # ========================================================
    # DIRECTORY CREATION
    # ========================================================

    def ensure_directory(
        self,
        directory: PathInput
    ) -> Path:
        """
        Create one project directory when necessary.

        Parameters
        ----------
        directory:
            Absolute path inside the project or relative path
            interpreted from the project root.

        Returns
        -------
        Path
            Resolved directory path.

        Raises
        ------
        ValueError
            If the directory is outside the project root.
        OSError
            If the operating system cannot create the directory.
        """

        return self.ensure_directories(directory)[0]

    def ensure_directories(
        self,
        *directories: PathInput
    ) -> tuple[Path, ...]:
        """
        Create selected project directories.

        If no directory arguments are supplied, all standard
        project directories are created.

        Directory creation is explicit to avoid unexpected
        filesystem modifications during module imports.

        Parameters
        ----------
        *directories:
            Directories to create.

        Returns
        -------
        tuple[Path, ...]
            Resolved paths of the created or existing directories.
        """

        targets = (
            directories
            if directories
            else self.standard_directories
        )

        created_directories: list[Path] = []

        for directory in targets:

            resolved_directory = (
                self._resolve_inside_project(directory)
            )

            resolved_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            created_directories.append(
                resolved_directory
            )

        return tuple(created_directories)

    def ensure_standard_directories(
        self
    ) -> tuple[Path, ...]:
        """
        Create the complete standard project directory structure.

        Returns
        -------
        tuple[Path, ...]
            Created or already-existing standard directories.
        """

        return self.ensure_directories()

    # ========================================================
    # RELATIVE PATH UTILITIES
    # ========================================================

    def relative_path(
        self,
        path: PathInput
    ) -> Path:
        """
        Convert a project path into a project-relative path.

        Useful for generating readable logs and notebook outputs
        without exposing unnecessary absolute filesystem paths.

        Parameters
        ----------
        path:
            Project-relative or absolute path inside the project.

        Returns
        -------
        Path
            Relative path from the project root.
        """

        resolved_path = self._resolve_inside_project(path)

        return resolved_path.relative_to(
            self.root
        )


# ============================================================
# DEFAULT PROJECT PATH REGISTRY
# ============================================================

# The default registry is initialized at import time.
#
# IMPORTANT:
# Initializing PATHS does not create any directories.
# Only paths are resolved and registered.

PATHS = ProjectPaths.discover()


# Convenient project-root reference.

PROJECT_ROOT = PATHS.project_root


# ============================================================
# PUBLIC MODULE INTERFACE
# ============================================================

__all__ = (
    "ProjectPaths",
    "PATHS",
    "PROJECT_ROOT",
    "PathInput",
)

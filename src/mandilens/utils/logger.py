
"""
MandiLens India — Centralized Logging Infrastructure
=====================================================

Purpose
-------
Provide a reusable, configurable, and production-oriented logging
infrastructure for the MandiLens India application.

Responsibilities
----------------
1. Configure centralized application logging.
2. Support console and file-based logging.
3. Support configurable logging levels and formats.
4. Implement automatic log rotation.
5. Prevent duplicate handlers.
6. Integrate with centralized project path management.
7. Load logging settings from project_config.yaml.
8. Support custom project roots and testing environments.
9. Keep generated log files inside the project directory.
10. Avoid unexpected filesystem operations during module import.

Configuration Source
--------------------
config/project_config.yaml

Expected configuration structure:

logging:
    level: "INFO"
    console_logging: true
    file_logging: true
    log_directory: "logs"
    log_filename: "mandilens.log"
    use_timestamp: true
    include_log_level: true
    include_module_name: true
    include_process_id: false
    rotate_log_files: true
    max_file_size_mb: 10
    backup_count: 5

Dependencies
------------
Python standard library:
    logging
    logging.handlers
    pathlib
    threading
    dataclasses

Internal dependencies:
    mandilens.utils.paths
    mandilens.utils.config

Important Design Principles
----------------------------
- No log files or directories are created during module import.
- The root Python logger is not configured or modified.
- Application loggers use the "mandilens" namespace.
- Repeated logger requests do not create duplicate handlers.
- File logging uses UTF-8 encoding.
- Log directories must remain inside the project root.
- Logging configuration failures are reported explicitly.
- Credentials and secrets must never be included in log messages.

Project:
    MandiLens India

Package:
    mandilens.utils.logger
"""

from __future__ import annotations

import logging
import math
import os
import threading

from collections.abc import Mapping
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path, PureWindowsPath
from typing import Any

from .config import ConfigurationError, load_config
from .paths import PATHS, ProjectPaths


# ============================================================
# MODULE CONSTANTS
# ============================================================

MANDILENS_LOGGER_NAME = "mandilens"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIRECTORY = "logs"
DEFAULT_LOG_FILENAME = "mandilens.log"

DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_BACKUP_COUNT = 5

DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_CONFIGURATION = {
    "level": DEFAULT_LOG_LEVEL,
    "console_logging": True,
    "file_logging": True,
    "log_directory": DEFAULT_LOG_DIRECTORY,
    "log_filename": DEFAULT_LOG_FILENAME,
    "use_timestamp": True,
    "include_log_level": True,
    "include_module_name": True,
    "include_process_id": False,
    "rotate_log_files": True,
    "max_file_size_mb": DEFAULT_MAX_FILE_SIZE_MB,
    "backup_count": DEFAULT_BACKUP_COUNT,
}

# Internal marker used to identify handlers managed by this module.
_MANAGED_HANDLER_ATTRIBUTE = "_mandilens_managed_handler"

# Global configuration state.
_LOGGING_LOCK = threading.RLock()

_LOGGING_CONFIGURED = False

_LOGGING_SIGNATURE: tuple[Any, ...] | None = None


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class LoggingConfigurationError(RuntimeError):
    """
    Raised when logging configuration cannot be loaded or applied.

    Examples
    --------
    - Invalid logging level.
    - Invalid logging configuration values.
    - Invalid log filename.
    - Log directory outside the project root.
    - Failure to initialize a file handler.
    """


# ============================================================
# INTERNAL CONFIGURATION MODEL
# ============================================================

@dataclass(frozen=True, slots=True)
class _LoggingSettings:
    """
    Immutable representation of validated logging settings.

    Attributes
    ----------
    level:
        Numeric Python logging level.

    console_logging:
        Whether console logging is enabled.

    file_logging:
        Whether file logging is enabled.

    log_directory:
        Project-relative or project-contained log directory.

    log_filename:
        Name of the primary log file.

    use_timestamp:
        Whether timestamps should be included.

    include_log_level:
        Whether severity levels should be included.

    include_module_name:
        Whether logger/module names should be included.

    include_process_id:
        Whether process identifiers should be included.

    rotate_log_files:
        Whether rotating file logging should be enabled.

    max_file_size_mb:
        Maximum size of an individual log file.

    backup_count:
        Number of backup log files to retain.
    """

    level: int

    console_logging: bool
    file_logging: bool

    log_directory: str
    log_filename: str

    use_timestamp: bool
    include_log_level: bool
    include_module_name: bool
    include_process_id: bool

    rotate_log_files: bool

    max_file_size_mb: float
    backup_count: int


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def _validate_logging_level(value: Any) -> int:
    """
    Convert a logging level into its numeric representation.

    Accepted examples
    -----------------
    "DEBUG"
    "INFO"
    "WARNING"
    "ERROR"
    "CRITICAL"
    logging.INFO
    """

    if isinstance(value, bool):
        raise LoggingConfigurationError(
            "Logging level cannot be a boolean value."
        )

    if isinstance(value, int):
        if value < 0:
            raise LoggingConfigurationError(
                "Logging level cannot be negative."
            )

        return value

    if isinstance(value, str):

        normalized_level = value.strip().upper()

        if not normalized_level:
            raise LoggingConfigurationError(
                "Logging level cannot be empty."
            )

        numeric_level = logging.getLevelName(
            normalized_level
        )

        if isinstance(numeric_level, int):
            return numeric_level

        raise LoggingConfigurationError(
            f"Unsupported logging level: {value!r}. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    raise LoggingConfigurationError(
        "Logging level must be a string or integer."
    )


def _validate_boolean(
    configuration: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    """
    Validate a boolean configuration option.

    YAML boolean values must be actual Python booleans.
    Strings such as "true" and "false" are rejected.
    """

    value = configuration.get(key, default)

    if not isinstance(value, bool):
        raise LoggingConfigurationError(
            f"Logging option {key!r} must be a boolean. "
            f"Received: {value!r}"
        )

    return value


def _validate_log_directory(value: Any) -> str:
    """
    Validate the configured log directory.

    Absolute paths are permitted as input, but the path manager
    will subsequently enforce that the resolved directory remains
    inside the project root.
    """

    if isinstance(value, os.PathLike):
        value = os.fspath(value)

    if not isinstance(value, str):
        raise LoggingConfigurationError(
            "The log directory must be a string or path-like value."
        )

    normalized_directory = value.strip()

    if not normalized_directory:
        raise LoggingConfigurationError(
            "The log directory cannot be empty."
        )

    return normalized_directory


def _validate_log_filename(value: Any) -> str:
    """
    Validate the log filename and prevent path traversal.

    Only a filename is accepted. Directory components must be
    supplied through the log_directory configuration option.
    """

    if not isinstance(value, str):
        raise LoggingConfigurationError(
            "The log filename must be a string."
        )

    filename = value.strip()

    if not filename:
        raise LoggingConfigurationError(
            "The log filename cannot be empty."
        )

    if filename in {".", ".."}:
        raise LoggingConfigurationError(
            "Invalid log filename."
        )

    if "\x00" in filename:
        raise LoggingConfigurationError(
            "Log filename contains an invalid null character."
        )

    # Protect against both POSIX and Windows-style paths.
    if (
        Path(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise LoggingConfigurationError(
            "Log filename must not contain directory components. "
            "Configure the directory separately."
        )

    return filename


def _validate_file_size(value: Any) -> float:
    """
    Validate the maximum log file size in megabytes.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise LoggingConfigurationError(
            "max_file_size_mb must be a positive number."
        )

    size = float(value)

    if not math.isfinite(size) or size <= 0:
        raise LoggingConfigurationError(
            "max_file_size_mb must be greater than zero."
        )

    if int(size * 1024 * 1024) < 1:
        raise LoggingConfigurationError(
            "Configured maximum log file size is too small."
        )

    return size


def _validate_backup_count(value: Any) -> int:
    """
    Validate the number of backup log files.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise LoggingConfigurationError(
            "backup_count must be a non-negative integer."
        )

    if value < 0:
        raise LoggingConfigurationError(
            "backup_count cannot be negative."
        )

    return value


def _build_logging_settings(
    project_config: Mapping[str, Any],
) -> _LoggingSettings:
    """
    Extract and validate logging settings from project configuration.

    Parameters
    ----------
    project_config:
        Complete configuration dictionary loaded from
        config/project_config.yaml.

    Returns
    -------
    _LoggingSettings
        Validated immutable logging settings.

    Raises
    ------
    LoggingConfigurationError
        If the configuration structure or values are invalid.
    """

    if not isinstance(project_config, Mapping):
        raise LoggingConfigurationError(
            "Project configuration must be a mapping."
        )

    logging_config = project_config.get("logging")

    if not isinstance(logging_config, Mapping):
        raise LoggingConfigurationError(
            "The project configuration must contain a valid "
            "'logging' mapping."
        )

    # Merge defaults with the project's configured values.
    configuration = {
        **DEFAULT_LOG_CONFIGURATION,
        **dict(logging_config),
    }

    return _LoggingSettings(
        level=_validate_logging_level(
            configuration["level"]
        ),

        console_logging=_validate_boolean(
            configuration,
            "console_logging",
            True,
        ),

        file_logging=_validate_boolean(
            configuration,
            "file_logging",
            True,
        ),

        log_directory=_validate_log_directory(
            configuration["log_directory"]
        ),

        log_filename=_validate_log_filename(
            configuration["log_filename"]
        ),

        use_timestamp=_validate_boolean(
            configuration,
            "use_timestamp",
            True,
        ),

        include_log_level=_validate_boolean(
            configuration,
            "include_log_level",
            True,
        ),

        include_module_name=_validate_boolean(
            configuration,
            "include_module_name",
            True,
        ),

        include_process_id=_validate_boolean(
            configuration,
            "include_process_id",
            False,
        ),

        rotate_log_files=_validate_boolean(
            configuration,
            "rotate_log_files",
            True,
        ),

        max_file_size_mb=_validate_file_size(
            configuration["max_file_size_mb"]
        ),

        backup_count=_validate_backup_count(
            configuration["backup_count"]
        ),
    )


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def _load_project_configuration() -> Mapping[str, Any]:
    """
    Load the central project configuration.

    The configuration module manages configuration caching and
    YAML validation.

    Returns
    -------
    Mapping[str, Any]
        Complete project configuration.
    """

    try:
        configuration = load_config("project")

    except ConfigurationError as error:
        raise LoggingConfigurationError(
            "Unable to load project logging configuration. "
            "Check config/project_config.yaml."
        ) from error

    if not isinstance(configuration, Mapping):
        raise LoggingConfigurationError(
            "Loaded project configuration is not a valid mapping."
        )

    return configuration


# ============================================================
# FORMATTER CREATION
# ============================================================

def _build_log_format(
    settings: _LoggingSettings,
) -> str:
    """
    Construct the log message format dynamically.

    Example
    -------
    2026-11-15 10:30:00 | INFO | mandilens.data | Loading dataset

    The exact fields depend on project configuration.
    """

    format_components: list[str] = []

    if settings.use_timestamp:
        format_components.append("%(asctime)s")

    if settings.include_log_level:
        format_components.append("%(levelname)s")

    if settings.include_module_name:
        format_components.append("%(name)s")

    if settings.include_process_id:
        format_components.append("PID=%(process)d")

    format_components.append("%(message)s")

    return " | ".join(format_components)


def _build_formatter(
    settings: _LoggingSettings,
) -> logging.Formatter:
    """
    Create the shared formatter for all logging handlers.
    """

    return logging.Formatter(
        fmt=_build_log_format(settings),
        datefmt=DEFAULT_TIMESTAMP_FORMAT,
        style="%",
    )


# ============================================================
# HANDLER MANAGEMENT
# ============================================================

def _mark_managed_handler(
    handler: logging.Handler,
) -> logging.Handler:
    """
    Mark a handler as managed by MandiLens logging infrastructure.

    This allows safe removal and replacement of project-owned
    handlers without disturbing handlers installed by other code.
    """

    setattr(
        handler,
        _MANAGED_HANDLER_ATTRIBUTE,
        True,
    )

    return handler


def _is_managed_handler(
    handler: logging.Handler,
) -> bool:
    """
    Determine whether a handler belongs to this module.
    """

    return bool(
        getattr(
            handler,
            _MANAGED_HANDLER_ATTRIBUTE,
            False,
        )
    )


def _close_handlers(
    handlers: list[logging.Handler],
) -> None:
    """
    Close handlers that have been created but not installed.
    """

    for handler in handlers:
        handler.close()


def _remove_managed_handlers(
    logger: logging.Logger,
) -> None:
    """
    Remove and close only MandiLens-managed handlers.

    Third-party handlers attached to the same logger are preserved.
    """

    for handler in logger.handlers[:]:

        if _is_managed_handler(handler):

            logger.removeHandler(handler)

            handler.close()


def _create_console_handler(
    formatter: logging.Formatter,
    level: int,
) -> logging.Handler:
    """
    Create a console logging handler.

    StreamHandler uses stderr by default, which is appropriate
    for application diagnostics and notebook execution.
    """

    handler = logging.StreamHandler()

    handler.setLevel(level)

    handler.setFormatter(formatter)

    return _mark_managed_handler(handler)


def _create_file_handler(
    settings: _LoggingSettings,
    paths: ProjectPaths,
    formatter: logging.Formatter,
) -> logging.Handler:
    """
    Create a project-contained file logging handler.

    Directory creation is performed explicitly when logging is
    configured, not when this module is imported.

    The project path manager prevents the resolved log directory
    from escaping the project root.
    """

    log_directory = paths.ensure_directory(
        settings.log_directory
    )

    log_file = log_directory / settings.log_filename

    if settings.rotate_log_files:

        handler: logging.Handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=int(
                settings.max_file_size_mb * 1024 * 1024
            ),
            backupCount=settings.backup_count,
            encoding="utf-8",
            delay=True,
        )

    else:

        handler = logging.FileHandler(
            filename=log_file,
            mode="a",
            encoding="utf-8",
            delay=True,
        )

    handler.setLevel(settings.level)

    handler.setFormatter(formatter)

    return _mark_managed_handler(handler)


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

def configure_logging(
    project_config: Mapping[str, Any] | None = None,
    *,
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure the centralized MandiLens application logger.

    This function initializes the named "mandilens" logger and
    attaches the handlers specified in project_config.yaml.

    Parameters
    ----------
    project_config:
        Optional complete project configuration mapping.

        When omitted, the configuration is loaded through
        mandilens.utils.config.load_config("project").

    paths:
        Optional ProjectPaths instance.

        Useful for custom project roots and isolated tests.
        Defaults to the global PATHS registry.

    force:
        Force handler reconfiguration even when the current
        configuration signature has not changed.

    Returns
    -------
    logging.Logger
        Configured root logger for the MandiLens namespace.

    Raises
    ------
    LoggingConfigurationError
        If settings are invalid or logging cannot be initialized.

    Notes
    -----
    - Repeated calls with unchanged settings are idempotent.
    - The Python root logger is never configured.
    - Existing non-MandiLens handlers are preserved.
    """

    global _LOGGING_CONFIGURED
    global _LOGGING_SIGNATURE

    selected_paths = paths or PATHS

    if not isinstance(selected_paths, ProjectPaths):
        raise TypeError(
            "paths must be an instance of ProjectPaths."
        )

    with _LOGGING_LOCK:

        # ----------------------------------------------------
        # 1. Load project configuration
        # ----------------------------------------------------

        if project_config is None:
            project_config = _load_project_configuration()

        # ----------------------------------------------------
        # 2. Validate settings
        # ----------------------------------------------------

        settings = _build_logging_settings(
            project_config
        )

        # Include project root so switching to another project
        # does not accidentally reuse the previous configuration.

        configuration_signature = (
            settings,
            str(selected_paths.project_root),
        )

        logger = logging.getLogger(
            MANDILENS_LOGGER_NAME
        )

        # ----------------------------------------------------
        # 3. Avoid duplicate handler creation
        # ----------------------------------------------------

        existing_managed_handlers = [
            handler
            for handler in logger.handlers
            if _is_managed_handler(handler)
        ]

        if (
            _LOGGING_CONFIGURED
            and _LOGGING_SIGNATURE == configuration_signature
            and existing_managed_handlers
            and not force
        ):
            return logger

        # ----------------------------------------------------
        # 4. Create formatter
        # ----------------------------------------------------

        formatter = _build_formatter(settings)

        new_handlers: list[logging.Handler] = []

        try:

            # -----------------------------------------------
            # Console logging
            # -----------------------------------------------

            if settings.console_logging:

                console_handler = _create_console_handler(
                    formatter=formatter,
                    level=settings.level,
                )

                new_handlers.append(
                    console_handler
                )

            # -----------------------------------------------
            # File logging
            # -----------------------------------------------

            if settings.file_logging:

                file_handler = _create_file_handler(
                    settings=settings,
                    paths=selected_paths,
                    formatter=formatter,
                )

                new_handlers.append(
                    file_handler
                )

            # -----------------------------------------------
            # Disabled logging fallback
            # -----------------------------------------------

            if not new_handlers:

                # Prevent logging's last-resort handler from
                # unexpectedly printing messages to stderr.
                null_handler = logging.NullHandler()

                new_handlers.append(
                    _mark_managed_handler(null_handler)
                )

        except (OSError, ValueError) as error:

            _close_handlers(new_handlers)

            raise LoggingConfigurationError(
                "Unable to initialize MandiLens log handlers. "
                "Check log directory permissions, filename, "
                "and project-root configuration."
            ) from error

        except Exception:

            _close_handlers(new_handlers)

            raise

        # ----------------------------------------------------
        # 5. Replace previous MandiLens handlers
        # ----------------------------------------------------

        _remove_managed_handlers(logger)

        for handler in new_handlers:
            logger.addHandler(handler)

        # ----------------------------------------------------
        # 6. Apply logger configuration
        # ----------------------------------------------------

        logger.setLevel(settings.level)

        # Prevent duplicate messages through the Python root logger.
        logger.propagate = False

        logger.disabled = False

        # ----------------------------------------------------
        # 7. Save configuration state
        # ----------------------------------------------------

        _LOGGING_SIGNATURE = configuration_signature

        _LOGGING_CONFIGURED = True

        return logger


# ============================================================
# LOGGER ACCESS
# ============================================================

def get_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Retrieve a configured logger from the MandiLens namespace.

    Parameters
    ----------
    name:
        Logger name, normally the calling module's __name__.

        Examples
        --------
        get_logger()

        get_logger(__name__)

        get_logger("data_engineering.validation")

        get_logger("mandilens.analytics.forecasting")

    Returns
    -------
    logging.Logger
        Configured application logger.

    Notes
    -----
    Logging is initialized lazily on the first call.

    Recommended usage inside project modules:

        logger = get_logger(__name__)

    Logger hierarchy:

        mandilens
            mandilens.utils
            mandilens.data_engineering
            mandilens.analytics
            mandilens.models
            mandilens.dashboard
    """

    with _LOGGING_LOCK:

        if not _LOGGING_CONFIGURED:
            configure_logging()

    if name is None:
        return logging.getLogger(
            MANDILENS_LOGGER_NAME
        )

    if not isinstance(name, str):
        raise TypeError(
            "Logger name must be a string or None."
        )

    normalized_name = name.strip()

    if not normalized_name:
        raise ValueError(
            "Logger name cannot be empty."
        )

    if (
        normalized_name == MANDILENS_LOGGER_NAME
        or normalized_name.startswith(
            f"{MANDILENS_LOGGER_NAME}."
        )
    ):
        full_logger_name = normalized_name

    else:
        full_logger_name = (
            f"{MANDILENS_LOGGER_NAME}.{normalized_name}"
        )

    return logging.getLogger(
        full_logger_name
    )


# ============================================================
# LOGGING SHUTDOWN
# ============================================================

def shutdown_logging() -> None:
    """
    Close and remove all MandiLens-managed logging handlers.

    This function is primarily useful for:
    - Automated testing.
    - Application shutdown.
    - Reinitialization with a different project configuration.
    - Releasing open file handles.

    Handlers installed by external libraries are preserved.

    After shutdown, the next get_logger() call initializes
    logging again using the current project configuration.
    """

    global _LOGGING_CONFIGURED
    global _LOGGING_SIGNATURE

    with _LOGGING_LOCK:

        logger = logging.getLogger(
            MANDILENS_LOGGER_NAME
        )

        _remove_managed_handlers(logger)

        _LOGGING_CONFIGURED = False

        _LOGGING_SIGNATURE = None


# ============================================================
# PUBLIC MODULE INTERFACE
# ============================================================

__all__ = (
    "MANDILENS_LOGGER_NAME",
    "LoggingConfigurationError",
    "configure_logging",
    "get_logger",
    "shutdown_logging",
)

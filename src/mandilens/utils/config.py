
"""
MandiLens India — Centralized Configuration Management
=======================================================

Module:
    src/mandilens/utils/config.py

Purpose:
    Provide a centralized, secure, validated and reusable
    configuration management system for MandiLens India.

Responsibilities:
    1. Load YAML configuration files safely.
    2. Reuse the centralized ProjectPaths implementation.
    3. Support all six project configuration files.
    4. Validate YAML document structure.
    5. Detect duplicate YAML mapping keys.
    6. Handle missing and malformed configuration files.
    7. Cache loaded configurations.
    8. Automatically invalidate stale cache entries.
    9. Provide nested configuration access.
    10. Support explicit configuration reloading.
    11. Protect cached data against accidental mutation.
    12. Support concurrent access safely.

Configuration files:
    - config/project_config.yaml
    - config/data_sources.yaml
    - config/commodity_categories.yaml
    - config/data_quality.yaml
    - config/model_config.yaml
    - config/dashboard_config.yaml

Dependencies:
    PyYAML >= 6.0, < 7.0
    mandilens.utils.paths

Design principles:
    - Safe YAML parsing.
    - No configuration file modification.
    - No automatic directory creation.
    - No hardcoded machine-specific paths.
    - No silent configuration failures.
    - No unnecessary external dependencies.

Project:
    MandiLens India

Package:
    mandilens.utils.config
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Final

import yaml

from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .paths import PATHS, ProjectPaths


# =============================================================================
# 1. CONFIGURATION REGISTRY
# =============================================================================

_CONFIG_PATH_ATTRIBUTES: Final[dict[str, str]] = {
    "project": "project_config_file",
    "data_sources": "data_sources_config_file",
    "commodity_categories": "commodity_categories_config_file",
    "data_quality": "data_quality_config_file",
    "model": "model_config_file",
    "dashboard": "dashboard_config_file",
}


_CONFIG_FILE_NAMES: Final[dict[str, str]] = {
    "project": "project_config.yaml",
    "data_sources": "data_sources.yaml",
    "commodity_categories": "commodity_categories.yaml",
    "data_quality": "data_quality.yaml",
    "model": "model_config.yaml",
    "dashboard": "dashboard_config.yaml",
}


# Required structural sections for each configuration.
#
# These checks intentionally validate only the essential structure.
# They do not impose a complete application-specific YAML schema.

_REQUIRED_TOP_LEVEL_SECTIONS: Final[dict[str, tuple[str, ...]]] = {
    "project": (
        "project",
        "environment",
        "paths",
    ),
    "data_sources": (
        "metadata",
        "sources",
    ),
    "commodity_categories": (
        "metadata",
        "categories",
    ),
    "data_quality": (
        "metadata",
        "dataset",
    ),
    "model": (
        "metadata",
        "task",
        "target",
    ),
    "dashboard": (
        "metadata",
        "application",
    ),
}


# Build supported aliases without introducing separate path logic.

_CONFIG_ALIASES: Final[dict[str, str]] = {}

for _canonical_name, _file_name in _CONFIG_FILE_NAMES.items():

    _aliases = (
        _canonical_name,
        f"{_canonical_name}_config",
        _file_name,
        f"config/{_file_name}",
    )

    for _alias in _aliases:
        _CONFIG_ALIASES[_alias] = _canonical_name


# Sentinel used to distinguish a missing default from default=None.

_MISSING = object()


# =============================================================================
# 2. CUSTOM EXCEPTIONS
# =============================================================================


class ConfigurationError(Exception):
    """Base exception for MandiLens configuration operations."""


class ConfigurationNameError(ConfigurationError):
    """Raised when an unsupported configuration name is supplied."""


class ConfigurationFileNotFoundError(ConfigurationError):
    """Raised when a required configuration file does not exist."""


class ConfigurationFileReadError(ConfigurationError):
    """Raised when a configuration file cannot be read."""


class ConfigurationParseError(ConfigurationError):
    """Raised when a YAML configuration cannot be parsed."""


class ConfigurationStructureError(ConfigurationError):
    """Raised when a configuration has an invalid structure."""


class ConfigurationLookupError(ConfigurationError):
    """Raised when a requested configuration key is unavailable."""


# =============================================================================
# 3. DUPLICATE-KEY-SAFE YAML LOADER
# =============================================================================


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """
    Safe YAML loader that detects duplicate explicitly defined keys.

    PyYAML's normal safe loader may silently overwrite earlier values
    when a mapping contains duplicate keys.

    This loader rejects duplicate explicit keys while preserving
    SafeLoader's normal support for YAML values and safe constructors.

    YAML merge keys are left to the standard loader's merge handling.
    """

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:

        if not isinstance(node, MappingNode):
            return super().construct_mapping(node, deep=deep)

        explicitly_defined_keys: set[Any] = set()

        for key_node, _ in node.value:

            # YAML merge keys are handled by the parent SafeLoader.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue

            key = self.construct_object(
                key_node,
                deep=deep,
            )

            try:
                duplicate = key in explicitly_defined_keys

            except TypeError as error:
                raise ConstructorError(
                    "While constructing a YAML mapping",
                    node.start_mark,
                    "Found an unhashable mapping key.",
                    key_node.start_mark,
                ) from error

            if duplicate:
                raise ConstructorError(
                    "While constructing a YAML mapping",
                    node.start_mark,
                    f"Duplicate mapping key: {key!r}",
                    key_node.start_mark,
                )

            explicitly_defined_keys.add(key)

        return super().construct_mapping(
            node,
            deep=deep,
        )


# =============================================================================
# 4. CONFIGURATION MANAGER
# =============================================================================


class ConfigManager:
    """
    Central configuration manager for MandiLens India.

    Features:
        - Centralized YAML configuration loading.
        - Safe YAML parsing.
        - Configuration structure validation.
        - Duplicate key detection.
        - Thread-safe caching.
        - Automatic cache invalidation.
        - Nested configuration access.
        - Configuration reload support.

    Parameters
    ----------
    paths:
        ProjectPaths instance used to resolve configuration locations.

        Defaults to the existing global PATHS registry.

    Notes
    -----
    The manager does not modify YAML files or create directories.

    Returned configuration dictionaries are deep copies.
    Modifying a returned dictionary therefore cannot silently
    modify the manager's internal cache.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
    ) -> None:

        self.paths = paths or PATHS

        # Cache structure:
        #
        # {
        #     "project": (
        #         (file_modified_time_ns, file_size),
        #         configuration_dictionary
        #     )
        # }

        self._cache: dict[
            str,
            tuple[
                tuple[int, int],
                dict[str, Any],
            ],
        ] = {}

        # Protect concurrent cache reads and modifications.

        self._lock = RLock()

    # =========================================================================
    # 4.1 AVAILABLE CONFIGURATIONS
    # =========================================================================

    @property
    def available_configs(self) -> tuple[str, ...]:
        """
        Return the supported canonical configuration names.
        """

        return tuple(
            _CONFIG_PATH_ATTRIBUTES.keys()
        )

    # =========================================================================
    # 4.2 CONFIGURATION NAME NORMALIZATION
    # =========================================================================

    def _normalize_config_name(
        self,
        config_name: str,
    ) -> str:
        """
        Normalize a configuration name to its canonical identifier.

        Supported examples:
            project
            project_config
            project_config.yaml
            config/project_config.yaml

        Raises
        ------
        ConfigurationNameError
            If the supplied name is unsupported.
        """

        if not isinstance(config_name, str) or not config_name.strip():
            raise ConfigurationNameError(
                "Configuration name must be a non-empty string."
            )

        normalized_name = (
            config_name.strip()
            .lower()
            .replace("\\", "/")
        )

        if normalized_name.startswith("./"):
            normalized_name = normalized_name[2:]

        canonical_name = _CONFIG_ALIASES.get(
            normalized_name
        )

        if canonical_name is None:

            supported = ", ".join(
                self.available_configs
            )

            raise ConfigurationNameError(
                f"Unsupported configuration: {config_name!r}. "
                f"Available configurations: {supported}."
            )

        return canonical_name

    # =========================================================================
    # 4.3 RESOLVE CONFIGURATION FILE
    # =========================================================================

    def _get_config_path(
        self,
        config_name: str,
    ) -> Path:
        """
        Resolve a configuration file through ProjectPaths.

        No filesystem paths are independently constructed here.
        """

        canonical_name = self._normalize_config_name(
            config_name
        )

        path_attribute = _CONFIG_PATH_ATTRIBUTES[
            canonical_name
        ]

        config_path = getattr(
            self.paths,
            path_attribute,
            None,
        )

        if not isinstance(config_path, Path):
            raise ConfigurationStructureError(
                f"ProjectPaths does not provide a valid path "
                f"for configuration {canonical_name!r}."
            )

        return config_path

    # =========================================================================
    # 4.4 READABLE CONFIGURATION PATH
    # =========================================================================

    def _get_relative_path(
        self,
        path: Path,
    ) -> str:
        """
        Return a readable project-relative configuration path.
        """

        try:
            return self.paths.relative_path(path).as_posix()

        except (ValueError, OSError):
            return path.name

    # =========================================================================
    # 4.5 CONFIGURATION STRUCTURE VALIDATION
    # =========================================================================

    def _validate_structure(
        self,
        config_name: str,
        configuration: Any,
        path: Path,
    ) -> dict[str, Any]:
        """
        Validate the basic structure of a loaded configuration.

        Validation checks:
            1. YAML root must be a mapping.
            2. Root must not be empty.
            3. Top-level keys must be strings.
            4. Required sections must exist.
            5. Required sections must themselves be mappings.

        This intentionally does not impose a complete schema on
        every individual configuration section.
        """

        relative_path = self._get_relative_path(
            path
        )

        if not isinstance(configuration, dict):

            raise ConfigurationStructureError(
                f"Invalid configuration structure in "
                f"{relative_path}: YAML root must be a mapping."
            )

        if not configuration:

            raise ConfigurationStructureError(
                f"Configuration file is empty: {relative_path}."
            )

        invalid_keys = [
            key
            for key in configuration
            if not isinstance(key, str)
        ]

        if invalid_keys:

            raise ConfigurationStructureError(
                f"Invalid top-level configuration keys in "
                f"{relative_path}: all top-level keys must be strings."
            )

        required_sections = _REQUIRED_TOP_LEVEL_SECTIONS[
            config_name
        ]

        for section in required_sections:

            if section not in configuration:

                raise ConfigurationStructureError(
                    f"Missing required section {section!r} "
                    f"in {relative_path}."
                )

            if not isinstance(
                configuration[section],
                Mapping,
            ):

                raise ConfigurationStructureError(
                    f"Required section {section!r} in "
                    f"{relative_path} must be a YAML mapping."
                )

        return configuration

    # =========================================================================
    # 4.6 YAML ERROR FORMATTING
    # =========================================================================

    @staticmethod
    def _format_yaml_error(
        error: yaml.YAMLError,
    ) -> str:
        """
        Produce a readable YAML parsing error.
        """

        problem = (
            getattr(error, "problem", None)
            or str(error)
        )

        position = getattr(
            error,
            "problem_mark",
            None,
        )

        if position is not None:

            return (
                f"{problem} "
                f"(line {position.line + 1}, "
                f"column {position.column + 1})"
            )

        return problem

    # =========================================================================
    # 4.7 LOAD SINGLE CONFIGURATION
    # =========================================================================

    def load_config(
        self,
        config_name: str,
        *,
        reload: bool = False,
    ) -> dict[str, Any]:
        """
        Load one YAML configuration file.

        Parameters
        ----------
        config_name:
            Canonical name, configuration filename or supported alias.

        reload:
            If True, bypass the cache and read the file again.

        Returns
        -------
        dict
            Deep copy of the validated configuration.

        Raises
        ------
        ConfigurationNameError
            Unsupported configuration name.

        ConfigurationFileNotFoundError
            File does not exist.

        ConfigurationFileReadError
            File cannot be read.

        ConfigurationParseError
            Invalid YAML syntax or duplicate mapping keys.

        ConfigurationStructureError
            Invalid configuration structure.
        """

        canonical_name = self._normalize_config_name(
            config_name
        )

        path = self._get_config_path(
            canonical_name
        )

        relative_path = self._get_relative_path(
            path
        )

        with self._lock:

            # -------------------------------------------------------------
            # Verify file availability and obtain modification metadata.
            # -------------------------------------------------------------

            try:

                file_stats = path.stat()

                if not path.is_file():

                    raise ConfigurationFileNotFoundError(
                        f"Configuration is not a regular file: "
                        f"{relative_path}."
                    )

            except FileNotFoundError as error:

                raise ConfigurationFileNotFoundError(
                    f"Configuration file not found: {relative_path}."
                ) from error

            except OSError as error:

                raise ConfigurationFileReadError(
                    f"Unable to access configuration file "
                    f"{relative_path}: {error}."
                ) from error

            signature = (
                file_stats.st_mtime_ns,
                file_stats.st_size,
            )

            # -------------------------------------------------------------
            # Return a valid cached configuration when available.
            # -------------------------------------------------------------

            cached_entry = self._cache.get(
                canonical_name
            )

            if (
                not reload
                and cached_entry is not None
                and cached_entry[0] == signature
            ):

                return deepcopy(
                    cached_entry[1]
                )

            # -------------------------------------------------------------
            # Read and parse the YAML document.
            # -------------------------------------------------------------

            try:

                with path.open(
                    mode="r",
                    encoding="utf-8",
                ) as file:

                    configuration = yaml.load(
                        file,
                        Loader=_UniqueKeySafeLoader,
                    )

            except FileNotFoundError as error:

                self._cache.pop(
                    canonical_name,
                    None,
                )

                raise ConfigurationFileNotFoundError(
                    f"Configuration file disappeared during reading: "
                    f"{relative_path}."
                ) from error

            except UnicodeDecodeError as error:

                raise ConfigurationFileReadError(
                    f"Configuration file is not valid UTF-8: "
                    f"{relative_path}."
                ) from error

            except yaml.YAMLError as error:

                raise ConfigurationParseError(
                    f"Unable to parse YAML configuration "
                    f"{relative_path}: "
                    f"{self._format_yaml_error(error)}."
                ) from error

            except OSError as error:

                raise ConfigurationFileReadError(
                    f"Unable to read configuration "
                    f"{relative_path}: {error}."
                ) from error

            # -------------------------------------------------------------
            # Validate the parsed document.
            # -------------------------------------------------------------

            validated_configuration = self._validate_structure(
                canonical_name,
                configuration,
                path,
            )

            # -------------------------------------------------------------
            # Store the validated configuration in the cache.
            # -------------------------------------------------------------

            self._cache[canonical_name] = (
                signature,
                deepcopy(validated_configuration),
            )

            return deepcopy(
                validated_configuration
            )

    # =========================================================================
    # 4.8 LOAD ALL CONFIGURATIONS
    # =========================================================================

    def load_all(
        self,
        *,
        reload: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Load all six project configurations.

        Parameters
        ----------
        reload:
            Force all six files to be read again.

        Returns
        -------
        dict
            Dictionary containing all validated configurations.

        Example
        -------
        configs = CONFIG_MANAGER.load_all()

        project = configs["project"]
        sources = configs["data_sources"]
        """

        configurations: dict[str, dict[str, Any]] = {}

        for config_name in self.available_configs:

            configurations[config_name] = self.load_config(
                config_name,
                reload=reload,
            )

        return configurations

    # =========================================================================
    # 4.9 NESTED CONFIGURATION VALUE ACCESS
    # =========================================================================

    def get_value(
        self,
        config_name: str,
        key_path: str,
        *,
        default: Any = _MISSING,
    ) -> Any:
        """
        Retrieve a nested configuration value using dot notation.

        Parameters
        ----------
        config_name:
            Name of the configuration file.

        key_path:
            Dot-separated mapping keys.

        default:
            Optional fallback when the key does not exist.

        Returns
        -------
        Any
            Deep copy of the requested value.

        Examples
        --------
        project_name = CONFIG_MANAGER.get_value(
            "project",
            "project.name",
        )

        download_strategy = CONFIG_MANAGER.get_value(
            "data_sources",
            "sources.mandi_huggingface.acquisition.download_strategy",
        )

        random_seed = CONFIG_MANAGER.get_value(
            "project",
            "reproducibility.random_seed",
        )

        """

        if not isinstance(key_path, str) or not key_path.strip():

            raise ConfigurationLookupError(
                "Configuration key path must be a non-empty string."
            )

        keys = tuple(
            segment.strip()
            for segment in key_path.split(".")
        )

        if any(not key for key in keys):

            raise ConfigurationLookupError(
                f"Invalid configuration key path: {key_path!r}."
            )

        configuration = self.load_config(
            config_name
        )

        current_value: Any = configuration

        for key in keys:

            if (
                not isinstance(current_value, Mapping)
                or key not in current_value
            ):

                if default is not _MISSING:

                    return deepcopy(default)

                raise ConfigurationLookupError(
                    f"Configuration key {key_path!r} "
                    f"was not found in {config_name!r}."
                )

            current_value = current_value[key]

        return deepcopy(
            current_value
        )

    # =========================================================================
    # 4.10 CLEAR CONFIGURATION CACHE
    # =========================================================================

    def clear_cache(
        self,
        config_name: str | None = None,
    ) -> None:
        """
        Clear the configuration cache.

        Parameters
        ----------
        config_name:
            Optional configuration name.

            If supplied, only that configuration is removed.
            If omitted, the complete cache is cleared.
        """

        with self._lock:

            if config_name is None:

                self._cache.clear()

            else:

                canonical_name = self._normalize_config_name(
                    config_name
                )

                self._cache.pop(
                    canonical_name,
                    None,
                )


# =============================================================================
# 5. DEFAULT CONFIGURATION MANAGER
# =============================================================================

# One shared configuration manager for the entire project.
#
# This creates no directories and does not load YAML files
# during module import.

CONFIG_MANAGER = ConfigManager(
    paths=PATHS
)


# =============================================================================
# 6. PUBLIC CONVENIENCE FUNCTIONS
# =============================================================================


def load_config(
    config_name: str,
    *,
    reload: bool = False,
) -> dict[str, Any]:
    """
    Load one project configuration through the default manager.
    """

    return CONFIG_MANAGER.load_config(
        config_name,
        reload=reload,
    )


def load_all_configs(
    *,
    reload: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Load all six project configurations.
    """

    return CONFIG_MANAGER.load_all(
        reload=reload,
    )


def get_config_value(
    config_name: str,
    key_path: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """
    Retrieve a nested value through the default manager.
    """

    return CONFIG_MANAGER.get_value(
        config_name,
        key_path,
        default=default,
    )


def clear_config_cache(
    config_name: str | None = None,
) -> None:
    """
    Clear the shared configuration cache.
    """

    CONFIG_MANAGER.clear_cache(
        config_name=config_name
    )


# =============================================================================
# 7. PUBLIC MODULE INTERFACE
# =============================================================================

__all__ = (
    "ConfigurationError",
    "ConfigurationNameError",
    "ConfigurationFileNotFoundError",
    "ConfigurationFileReadError",
    "ConfigurationParseError",
    "ConfigurationStructureError",
    "ConfigurationLookupError",
    "ConfigManager",
    "CONFIG_MANAGER",
    "load_config",
    "load_all_configs",
    "get_config_value",
    "clear_config_cache",
)

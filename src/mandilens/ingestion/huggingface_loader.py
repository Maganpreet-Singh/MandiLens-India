"""Hugging Face dataset ingestion for MandiLens India.

This module discovers every configured Parquet file in a Hugging Face Dataset
repository, pins discovery and download to an immutable commit, and downloads
the full Parquet snapshot without loading any rows into Pandas.

The source repository, revision, extension policy, and raw-data directory are
read from ``config/data_sources.yaml``. ``HF_TOKEN`` is optional for the
project's public dataset.

Typical use::

    from mandilens.ingestion.huggingface_loader import HuggingFaceDatasetLoader

    loader = HuggingFaceDatasetLoader()
    inventory = loader.discover()
    print(inventory.parquet_file_count)

    # Call only when the complete download is intended.
    result = loader.download_dataset()
    print(result.file_count)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from huggingface_hub import HfApi, snapshot_download

from mandilens.utils.config import load_config
from mandilens.utils.logger import get_logger
from mandilens.utils.paths import PATHS, ProjectPaths

LOGGER = get_logger(__name__)
_T = TypeVar("_T")


class HuggingFaceLoaderError(RuntimeError):
    """Base exception for MandiLens Hugging Face ingestion errors."""


class RepositoryDiscoveryError(HuggingFaceLoaderError):
    """Raised when repository metadata or the remote inventory is invalid."""


class DatasetDownloadError(HuggingFaceLoaderError):
    """Raised when a repository snapshot cannot be downloaded."""


class DatasetIntegrityError(HuggingFaceLoaderError):
    """Raised when downloaded files do not match the remote inventory."""


@dataclass(frozen=True, slots=True)
class RemoteFileInfo:
    """Metadata for one file in the remote repository."""

    path: str
    size_bytes: int | None = None
    blob_id: str | None = None
    lfs_sha256: str | None = None

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.path).suffix.lower()


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """Complete Parquet inventory for one immutable repository revision."""

    repo_id: str
    repo_type: str
    requested_revision: str
    commit_sha: str
    discovered_at_utc: str
    is_private: bool
    is_gated: bool
    last_modified: str | None
    repository_file_count: int
    parquet_files: tuple[RemoteFileInfo, ...]
    expected_parquet_file_count: int | None = None

    @property
    def parquet_file_count(self) -> int:
        return len(self.parquet_files)

    @property
    def total_known_size_bytes(self) -> int | None:
        if any(file.size_bytes is None for file in self.parquet_files):
            return None
        return sum(file.size_bytes or 0 for file in self.parquet_files)

    @property
    def expected_file_count_matches(self) -> bool | None:
        if self.expected_parquet_file_count is None:
            return None
        return self.parquet_file_count == self.expected_parquet_file_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable inventory summary."""
        payload = asdict(self)
        payload.update(
            parquet_file_count=self.parquet_file_count,
            total_known_size_bytes=self.total_known_size_bytes,
            expected_file_count_matches=self.expected_file_count_matches,
        )
        return payload


@dataclass(frozen=True, slots=True)
class DatasetDownloadResult:
    """Result of a downloaded and inventory-checked snapshot."""

    repo_id: str
    commit_sha: str
    snapshot_directory: Path
    downloaded_files: tuple[Path, ...]
    total_size_bytes: int

    @property
    def file_count(self) -> int:
        return len(self.downloaded_files)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attribute(value: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a mapping or an SDK response object."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise HuggingFaceLoaderError(
            f"Expected an integer for configuration/metadata field {field_name!r}."
        ) from error


def _environment_int(
    environment_name: str,
    configured_value: Any,
    *,
    default: int,
    minimum: int,
) -> int:
    raw_value = os.getenv(environment_name)
    candidate = configured_value if raw_value is None else raw_value
    if candidate is None:
        candidate = default
    try:
        parsed = int(candidate)
    except (TypeError, ValueError) as error:
        raise HuggingFaceLoaderError(
            f"{environment_name} must contain an integer."
        ) from error
    if isinstance(candidate, bool) or parsed < minimum:
        raise HuggingFaceLoaderError(
            f"{environment_name} must be at least {minimum}."
        )
    return parsed


def _safe_remote_path(path: str) -> str:
    """Reject absolute paths and traversal before mapping remote files locally."""
    if not isinstance(path, str) or not path.strip():
        raise RepositoryDiscoveryError("Hugging Face returned an empty file path.")
    normalized = path.strip()
    parsed = PurePosixPath(normalized)
    if (
        parsed.is_absolute()
        or "\\" in normalized
        or ".." in parsed.parts
    ):
        raise RepositoryDiscoveryError(
            f"Unsafe repository-relative path received from Hugging Face: {normalized!r}."
        )
    return parsed.as_posix()


def _is_transient_error(error: Exception) -> bool:
    """Identify network failures worth retrying; do not retry ordinary 4xx errors."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in {408, 425, 429} or status >= 500

    name = type(error).__name__.lower()
    return any(
        marker in name
        for marker in ("timeout", "connection", "connecterror", "readerror", "temporar", "network")
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class HuggingFaceDatasetLoader:
    """Discover and download the complete configured MandiLens Hub dataset.

    Args:
        source_key: Key under ``sources`` in ``config/data_sources.yaml``.
        token: Optional Hugging Face token; otherwise ``HF_TOKEN`` is read.
        api: Optional HfApi-compatible client, mainly for isolated tests.
        paths: Optional MandiLens ProjectPaths instance.
        snapshot_downloader: Optional replacement for ``snapshot_download``.
    """

    def __init__(
        self,
        source_key: str = "mandi_huggingface",
        *,
        token: str | None = None,
        api: Any | None = None,
        paths: ProjectPaths | None = None,
        snapshot_downloader: Callable[..., str] | None = None,
    ) -> None:
        self.paths = paths or PATHS
        self.source_key = source_key

        try:
            registry = load_config("data_sources")
            self.source_config = registry["sources"][source_key]
            self.repository_config = self.source_config["repository"]
            self.access_config = self.source_config.get("access", {})
            self.acquisition_config = self.source_config.get("acquisition", {})
            self.storage_config = self.source_config.get("storage", {})
            self.download_policy = registry.get("global_policies", {}).get("download", {})
        except (KeyError, TypeError) as error:
            raise HuggingFaceLoaderError(
                f"Missing expected 'sources.{source_key}' configuration in data_sources.yaml."
            ) from error

        self.repo_id = str(self.repository_config.get("repo_id", "")).strip()
        self.repo_type = str(self.repository_config.get("repo_type", "dataset")).strip()
        self.requested_revision = str(
            self.repository_config.get("revision", "main")
        ).strip()
        if not self.repo_id or "/" not in self.repo_id:
            raise HuggingFaceLoaderError(
                f"Invalid Hugging Face repository ID in source {source_key!r}."
            )
        if self.repo_type not in {"model", "dataset", "space"}:
            raise HuggingFaceLoaderError(f"Unsupported repository type: {self.repo_type!r}.")
        if not self.requested_revision:
            raise HuggingFaceLoaderError("The configured repository revision is empty.")

        token_variable = str(
            self.access_config.get("token_environment_variable", "HF_TOKEN")
        ).strip() or "HF_TOKEN"
        self.token = token or os.getenv(token_variable) or None

        includes = self.acquisition_config.get("include_extensions", [".parquet"])
        excludes = self.acquisition_config.get("exclude_extensions", [])
        self.include_extensions = {str(ext).lower() for ext in includes}
        self.exclude_extensions = {str(ext).lower() for ext in excludes}
        if not self.include_extensions:
            raise HuggingFaceLoaderError("No acquisition file extensions are configured.")

        profile = self.source_config.get("published_profile", {})
        self.expected_file_count = _optional_int(
            profile.get("reported_file_count"),
            field_name="published_profile.reported_file_count",
        )

        self._api = api if api is not None else HfApi(token=self.token)
        self._snapshot_downloader = snapshot_downloader or snapshot_download
        self._inventory: DatasetInventory | None = None

    def _retry_settings(self) -> tuple[int, float]:
        retries = _environment_int(
            "MANDILENS_DOWNLOAD_RETRIES",
            self.download_policy.get("maximum_retries"),
            default=5,
            minimum=0,
        )
        try:
            backoff = float(self.download_policy.get("retry_backoff_seconds", 2))
        except (TypeError, ValueError) as error:
            raise HuggingFaceLoaderError(
                "Configured retry_backoff_seconds must be numeric."
            ) from error
        if backoff < 0:
            raise HuggingFaceLoaderError("retry_backoff_seconds cannot be negative.")
        return retries, backoff

    @staticmethod
    def _run_with_retries(
        operation: Callable[[], _T],
        *,
        description: str,
        maximum_retries: int,
        backoff_seconds: float,
    ) -> _T:
        """Run a network operation with bounded exponential backoff."""
        for attempt in range(maximum_retries + 1):
            try:
                return operation()
            except Exception as error:
                if attempt >= maximum_retries or not _is_transient_error(error):
                    raise
                delay = min(backoff_seconds * (2**attempt), 30.0)
                LOGGER.warning(
                    "%s failed temporarily; retry %d/%d in %.1f seconds.",
                    description,
                    attempt + 1,
                    maximum_retries,
                    delay,
                )
                if delay:
                    time.sleep(delay)
        raise RuntimeError("Retry loop terminated unexpectedly.")

    def discover(self, *, refresh: bool = False) -> DatasetInventory:
        """Resolve the configured revision and discover all matching Parquet files.

        The configured branch/tag is resolved to a commit SHA first. The file
        listing is then performed against that SHA so the inventory is tied to
        one immutable repository state. This method downloads metadata only.
        """
        if self._inventory is not None and not refresh:
            return self._inventory

        retries, backoff = self._retry_settings()
        try:
            info = self._run_with_retries(
                lambda: self._api.repo_info(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.requested_revision,
                    files_metadata=True,
                ),
                description="Hugging Face repository metadata lookup",
                maximum_retries=retries,
                backoff_seconds=backoff,
            )
        except Exception as error:
            LOGGER.exception("Unable to resolve Hugging Face repository %s.", self.repo_id)
            raise RepositoryDiscoveryError(
                "Could not retrieve Hugging Face repository metadata. Check the repository "
                "ID, network access, and optional HF_TOKEN."
            ) from error

        commit_sha = str(_attribute(info, "sha", "") or "").strip()
        if not commit_sha:
            raise RepositoryDiscoveryError("Hub metadata did not contain a commit SHA.")

        try:
            remote_paths = self._run_with_retries(
                lambda: self._api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=commit_sha,
                ),
                description="Pinned Hugging Face repository file listing",
                maximum_retries=retries,
                backoff_seconds=backoff,
            )
        except Exception as error:
            LOGGER.exception("Unable to list files for %s at %s.", self.repo_id, commit_sha)
            raise RepositoryDiscoveryError(
                "Could not retrieve the repository file inventory at the resolved commit."
            ) from error

        if not remote_paths:
            raise RepositoryDiscoveryError(f"Repository {self.repo_id!r} contains no files.")

        # repo_info(files_metadata=True) provides sibling sizes and object metadata.
        sibling_by_path: dict[str, Any] = {}
        for sibling in _attribute(info, "siblings", []) or []:
            sibling_path = _attribute(sibling, "rfilename") or _attribute(sibling, "path")
            if sibling_path:
                sibling_by_path[str(sibling_path)] = sibling

        files: list[RemoteFileInfo] = []
        for raw_path in remote_paths:
            path = _safe_remote_path(str(raw_path))
            suffix = PurePosixPath(path).suffix.lower()
            if suffix not in self.include_extensions or suffix in self.exclude_extensions:
                continue

            sibling = sibling_by_path.get(path)
            lfs = _attribute(sibling, "lfs") if sibling is not None else None
            size = _optional_int(
                _attribute(sibling, "size") if sibling is not None else None,
                field_name=f"remote size for {path}",
            )
            if size is None and lfs is not None:
                size = _optional_int(_attribute(lfs, "size"), field_name=f"LFS size for {path}")

            blob_id = _attribute(sibling, "blob_id") if sibling is not None else None
            sha256 = _attribute(lfs, "sha256") if lfs is not None else None
            files.append(
                RemoteFileInfo(
                    path=path,
                    size_bytes=size,
                    blob_id=str(blob_id) if blob_id else None,
                    lfs_sha256=str(sha256) if sha256 else None,
                )
            )

        files.sort(key=lambda item: item.path)
        if not files:
            raise RepositoryDiscoveryError(
                f"No files matched configured extensions {sorted(self.include_extensions)}."
            )
        if self.expected_file_count is not None and len(files) != self.expected_file_count:
            LOGGER.warning(
                "Discovered %d Parquet files; YAML reports %d. The live inventory is "
                "retained as authoritative; review the difference.",
                len(files),
                self.expected_file_count,
            )

        modified = _attribute(info, "last_modified")
        inventory = DatasetInventory(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            requested_revision=self.requested_revision,
            commit_sha=commit_sha,
            discovered_at_utc=datetime.now(timezone.utc).isoformat(),
            is_private=bool(_attribute(info, "private", False)),
            is_gated=bool(_attribute(info, "gated", False)),
            last_modified=str(modified) if modified else None,
            repository_file_count=len(remote_paths),
            parquet_files=tuple(files),
            expected_parquet_file_count=self.expected_file_count,
        )
        self._inventory = inventory
        LOGGER.info(
            "Discovered %d Parquet files in %s at commit %s.",
            inventory.parquet_file_count,
            inventory.repo_id,
            inventory.commit_sha,
        )
        return inventory

    def _raw_data_root(self, destination_root: str | os.PathLike[str] | None) -> Path:
        """Resolve a configured/overridden destination and keep it inside the project."""
        configured = (
            destination_root
            or os.getenv("MANDILENS_RAW_DATA_DIR")
            or self.storage_config.get("raw_directory")
        )
        if not configured:
            raise HuggingFaceLoaderError("No raw-data directory is configured.")

        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = self.paths.root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.paths.root)
        except ValueError as error:
            raise HuggingFaceLoaderError(
                "The dataset destination must remain inside the MandiLens project root."
            ) from error
        return candidate

    def _snapshot_directory(
        self,
        inventory: DatasetInventory,
        destination_root: str | os.PathLike[str] | None,
    ) -> Path:
        """Create a commit-specific directory without replacing older revisions."""
        directory = self._raw_data_root(destination_root) / "snapshots" / inventory.commit_sha
        return self.paths.ensure_directory(directory)

    @staticmethod
    def _local_file_path(snapshot_directory: Path, remote_file: RemoteFileInfo) -> Path:
        remote_path = _safe_remote_path(remote_file.path)
        local_path = snapshot_directory.joinpath(*PurePosixPath(remote_path).parts).resolve()
        try:
            local_path.relative_to(snapshot_directory.resolve())
        except ValueError as error:
            raise DatasetIntegrityError(
                f"Remote path escapes the local snapshot directory: {remote_path!r}."
            ) from error
        return local_path

    def download_dataset(
        self,
        *,
        destination_root: str | os.PathLike[str] | None = None,
        max_workers: int | None = None,
        max_retries: int | None = None,
        force_download: bool = False,
        refresh_inventory: bool = False,
    ) -> DatasetDownloadResult:
        """Download every discovered Parquet file and verify local presence and size.

        Files are stored beneath ``<raw_directory>/snapshots/<commit_sha>/``
        with their remote relative paths preserved. No year, geography,
        commodity, market or variety filtering is applied.

        Full SHA-256 and Parquet footer/row-count verification are deliberately
        left to the dedicated download-verification stage.
        """
        inventory = self.discover(refresh=refresh_inventory)
        snapshot_dir = self._snapshot_directory(inventory, destination_root)

        worker_count = (
            _environment_int(
                "MANDILENS_DOWNLOAD_WORKERS",
                self.download_policy.get("concurrent_downloads"),
                default=4,
                minimum=1,
            )
            if max_workers is None
            else max_workers
        )
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
            raise HuggingFaceLoaderError("max_workers must be a positive integer.")

        configured_retries, backoff = self._retry_settings()
        retries = configured_retries if max_retries is None else max_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise HuggingFaceLoaderError("max_retries must be a non-negative integer.")

        allow_patterns = [file.path for file in inventory.parquet_files]

        def _download() -> str:
            return self._snapshot_downloader(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=inventory.commit_sha,
                local_dir=snapshot_dir,
                allow_patterns=allow_patterns,
                max_workers=worker_count,
                token=self.token,
                force_download=force_download,
            )

        LOGGER.info(
            "Downloading %d Parquet files from %s at commit %s.",
            len(allow_patterns),
            self.repo_id,
            inventory.commit_sha,
        )
        try:
            self._run_with_retries(
                _download,
                description="Hugging Face Parquet snapshot download",
                maximum_retries=retries,
                backoff_seconds=backoff,
            )
        except Exception as error:
            LOGGER.exception("Snapshot download failed for commit %s.", inventory.commit_sha)
            raise DatasetDownloadError(
                "The Parquet snapshot download failed. Check connectivity, disk space and "
                "optional Hugging Face authentication, then retry."
            ) from error

        local_files: list[Path] = []
        missing: list[str] = []
        size_mismatches: list[str] = []
        for remote_file in inventory.parquet_files:
            local_file = self._local_file_path(snapshot_dir, remote_file)
            if not local_file.is_file():
                missing.append(remote_file.path)
                continue
            actual_size = local_file.stat().st_size
            if remote_file.size_bytes is not None and actual_size != remote_file.size_bytes:
                size_mismatches.append(
                    f"{remote_file.path}: expected {remote_file.size_bytes}, got {actual_size}"
                )
                continue
            local_files.append(local_file)

        if missing or size_mismatches:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:10]))
            if size_mismatches:
                details.append("size_mismatch=" + "; ".join(size_mismatches[:10]))
            raise DatasetIntegrityError(
                "Downloaded files failed the remote-inventory check: " + " | ".join(details)
            )

        total_bytes = sum(file.stat().st_size for file in local_files)
        result = DatasetDownloadResult(
            repo_id=self.repo_id,
            commit_sha=inventory.commit_sha,
            snapshot_directory=snapshot_dir,
            downloaded_files=tuple(local_files),
            total_size_bytes=total_bytes,
        )
        LOGGER.info(
            "Snapshot download complete: %d/%d files; %.2f MiB.",
            result.file_count,
            inventory.parquet_file_count,
            total_bytes / (1024**2),
        )
        return result

    def summary(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return a compact JSON-friendly summary of the remote dataset."""
        inventory = self.discover(refresh=refresh)
        return {
            "repo_id": inventory.repo_id,
            "repo_type": inventory.repo_type,
            "requested_revision": inventory.requested_revision,
            "resolved_commit_sha": inventory.commit_sha,
            "repository_file_count": inventory.repository_file_count,
            "parquet_file_count": inventory.parquet_file_count,
            "expected_parquet_file_count": inventory.expected_parquet_file_count,
            "expected_file_count_matches": inventory.expected_file_count_matches,
            "total_known_size_bytes": inventory.total_known_size_bytes,
            "discovered_at_utc": inventory.discovered_at_utc,
            "is_private": inventory.is_private,
            "is_gated": inventory.is_gated,
        }


__all__ = [
    "DatasetDownloadError",
    "DatasetDownloadResult",
    "DatasetIntegrityError",
    "DatasetInventory",
    "HuggingFaceDatasetLoader",
    "HuggingFaceLoaderError",
    "RemoteFileInfo",
    "RepositoryDiscoveryError",
]

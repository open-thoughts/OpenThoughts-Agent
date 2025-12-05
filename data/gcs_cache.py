#!/usr/bin/env python3
"""
GCS Function Caching Decorator

A decorator that caches function results in Google Cloud Storage based on
function input signatures and checksums.
"""

import hashlib
import pickle
import inspect
import functools
import tarfile
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union
from google.cloud import storage
import os
import json
import h5py

logger = logging.getLogger(__name__)


_DEFAULT_CREDENTIALS_PATH: Optional[str] = None


def _resolve_credentials_path(config_value: Optional[Union[str, dict]]) -> Optional[str]:
    """Resolve the credentials path from config value, environment, or defaults."""
    if isinstance(config_value, dict):
        config_dict = config_value
        env_key = config_dict.get("env")
        if env_key:
            candidate = os.environ.get(env_key)
            if candidate:
                config_value = candidate
        if isinstance(config_value, dict):
            path_value = config_dict.get("path")
            if path_value:
                config_value = path_value

    if not config_value:
        config_value = _DEFAULT_CREDENTIALS_PATH

    if not config_value:
        return None

    if isinstance(config_value, str):
        expanded = os.path.expanduser(os.path.expandvars(config_value))
        return expanded or None

    return None


def set_default_credentials_path(path: Optional[str]) -> None:
    """Configure a default credentials path used by gcs_cache decorators."""
    global _DEFAULT_CREDENTIALS_PATH
    _DEFAULT_CREDENTIALS_PATH = path if path else None


class GCSCache:
    """Google Cloud Storage cache client for dcagent with local caching."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None, local_cache_dir: Optional[str] = None):
        """
        Initialize GCS cache client.
        
        Args:
            bucket_name: GCS bucket name for caching
            credentials_path: Path to GCS service account credentials JSON
            local_cache_dir: Directory for local cache (defaults to LOCAL_DATA_CACHE env var or ~/.dcagent_cache)
        """
        self.bucket_name = bucket_name
        
        # Set up local cache directory
        if local_cache_dir is None:
            local_cache_dir = os.environ.get("LOCAL_DATA_CACHE", os.path.expanduser("~/.dcagent_cache"))
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def _compute_directory_hash(self, dir_path: Union[str, Path]) -> str:
        """
        Compute hash of directory contents.

        Args:
            dir_path: Path to directory

        Returns:
            SHA256 hash of directory contents
        """
        dir_path = Path(dir_path)
        hasher = hashlib.sha256()

        # Get all files in sorted order for deterministic hashing
        all_files = sorted(dir_path.rglob('*'))

        for file_path in all_files:
            if file_path.is_file():
                # Add relative path to hash
                rel_path = file_path.relative_to(dir_path)
                hasher.update(str(rel_path).encode())

                # Add file contents to hash
                try:
                    with open(file_path, 'rb') as f:
                        # Read in chunks to handle large files
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                except Exception:
                    # For files we can't read, just hash the path
                    pass

        return hasher.hexdigest()

    def _compute_hdf5_hash(self, hdf5_path: Union[str, Path]) -> str:
        """
        Compute hash of HDF5 file contents.

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            SHA256 hash of HDF5 file contents
        """
        hdf5_path = Path(hdf5_path)
        hasher = hashlib.sha256()

        try:
            with h5py.File(hdf5_path, 'r') as f:
                def visit_item(name, obj):
                    """Visit each item in HDF5 file and update hash."""
                    # Hash the path/name
                    hasher.update(name.encode())

                    if isinstance(obj, h5py.Dataset):
                        # Hash dataset shape and dtype
                        hasher.update(str(obj.shape).encode())
                        hasher.update(str(obj.dtype).encode())

                        # Hash dataset contents
                        # For large datasets, we read in chunks
                        try:
                            data = obj[()]
                            if hasattr(data, 'tobytes'):
                                hasher.update(data.tobytes())
                            else:
                                hasher.update(str(data).encode())
                        except Exception as e:
                            # If we can't read the dataset, hash the error
                            hasher.update(str(e).encode())

                        # Hash attributes
                        for attr_name, attr_value in obj.attrs.items():
                            hasher.update(attr_name.encode())
                            hasher.update(str(attr_value).encode())

                    elif isinstance(obj, h5py.Group):
                        # Hash group attributes
                        for attr_name, attr_value in obj.attrs.items():
                            hasher.update(attr_name.encode())
                            hasher.update(str(attr_value).encode())

                # Visit all items in the HDF5 file
                f.visititems(visit_item)

        except Exception as e:
            logger.warning(f"Error computing HDF5 hash for {hdf5_path}: {e}")
            # Fallback to hashing the file directly
            with open(hdf5_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

        return hasher.hexdigest()
    
    def _process_arg_for_checksum(self, arg: Any) -> Any:
        """
        Process an argument for checksum computation.
        If it's a directory path, return a hash of its contents instead.
        If it's an HDF5 file path, return a hash of its contents instead.

        Args:
            arg: Argument to process

        Returns:
            Processed argument (directory/HDF5 hash or original arg)
        """
        if isinstance(arg, (str, Path)):
            # Only consider strings that look like paths (contain / or \)
            # and are not too long (to avoid non-path strings)
            if isinstance(arg, str) and ('/' in arg or '\\' in arg) and len(arg) < 500:
                try:
                    path = Path(arg)
                    if path.exists():
                        if path.is_dir():
                            # Return a special tuple that includes directory marker and content hash
                            dir_hash = self._compute_directory_hash(path)
                            logger.info(f"Computing content hash for directory {path}: {dir_hash[:8]}...")
                            return ('__DIR_CONTENT_HASH__', dir_hash)
                        elif path.is_file() and (path.suffix.lower() in ['.h5', '.hdf5', '.hdf']):
                            hdf5_hash = self._compute_hdf5_hash(path)
                            logger.info(f"Computing content hash for HDF5 file {path}: {hdf5_hash[:8]}...")
                            return ('__HDF5_CONTENT_HASH__', hdf5_hash)
                except (OSError, ValueError):
                    # Not a valid path, treat as regular string
                    pass
            elif isinstance(arg, Path):
                # For Path objects, check directly
                try:
                    if arg.exists():
                        if arg.is_dir():
                            dir_hash = self._compute_directory_hash(arg)
                            logger.info(f"Computing content hash for directory {arg}: {dir_hash[:8]}...")
                            return ('__DIR_CONTENT_HASH__', dir_hash)
                        elif arg.is_file() and (arg.suffix.lower() in ['.h5', '.hdf5', '.hdf']):
                            hdf5_hash = self._compute_hdf5_hash(arg)
                            logger.info(f"Computing content hash for HDF5 file {arg}: {hdf5_hash[:8]}...")
                            return ('__HDF5_CONTENT_HASH__', hdf5_hash)
                except (OSError, ValueError):
                    pass
        elif isinstance(arg, (list, tuple)):
            # Recursively process list/tuple arguments
            return type(arg)(self._process_arg_for_checksum(item) for item in arg)
        elif isinstance(arg, dict):
            # Recursively process dictionary arguments
            return {k: self._process_arg_for_checksum(v) for k, v in arg.items()}

        return arg
    
    def _compute_checksum(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """
        Compute checksum for function signature and arguments.
        If string arguments are directory paths, hash the directory contents.

        Args:
            func: Function to cache
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            SHA256 checksum string
        """
        # Get function source code
        try:
            func_source = inspect.getsource(func)
        except OSError:
            # Fallback for built-in functions
            func_source = f"{func.__module__}.{func.__name__}"
        
        # Process args and kwargs to replace directory paths with content hashes
        processed_args = tuple(self._process_arg_for_checksum(arg) for arg in args)
        processed_kwargs = {k: self._process_arg_for_checksum(v) for k, v in kwargs.items()}
        
        # Create signature from function name, source, args, and kwargs
        signature_data = {
            'function_name': func.__name__,
            'function_source': func_source,
            'args': processed_args,
            'kwargs': processed_kwargs
        }

        # Serialize and hash
        serialized = pickle.dumps(signature_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()
    
    def _get_cache_key(self, checksum: str, func_name: str) -> str:
        """
        Generate GCS object key for cache entry.
        
        Args:
            checksum: Function signature checksum
            func_name: Function name
            
        Returns:
            GCS object key
        """
        return f"dcagent_cache/{func_name}/{checksum}.pkl"
    
    def _get_local_cache_path(self, checksum: str, func_name: str) -> Path:
        """
        Generate local cache file path for cache entry.
        
        Args:
            checksum: Function signature checksum
            func_name: Function name
            
        Returns:
            Local cache file path
        """
        cache_dir = self.local_cache_dir / func_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{checksum}.pkl"
    
    def _get_from_local_cache(self, checksum: str, func_name: str) -> tuple[bool, Any]:
        """
        Get cached result from local cache if available.
        
        Args:
            checksum: Function signature checksum
            func_name: Function name
            
        Returns:
            Tuple of (cache_hit: bool, result: Any)
        """
        local_cache_path = self._get_local_cache_path(checksum, func_name)
        
        if local_cache_path.exists():
            try:
                with open(local_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if cache_data.get('type') == 'directory':
                    # Extract tar archive to temporary directory
                    logger.info(f"Local cache HIT for {func_name} - extracting directory")
                    temp_dir = self._untar_to_temp_directory(cache_data['tar_data'])
                    logger.info(f"Directory extracted to temporary path: {temp_dir}")
                    return True, temp_dir
                else:
                    # Regular cached result
                    logger.info(f"Local cache HIT for {func_name}")
                    return True, cache_data['result']
            except Exception as e:
                logger.error(f"Local cache error for {func_name}: {e}")
                # Remove corrupted cache file
                local_cache_path.unlink(missing_ok=True)
                return False, None
        
        return False, None
    
    def _put_to_local_cache(self, checksum: str, func_name: str, cache_data: dict) -> None:
        """
        Store cache data in local cache.
        
        Args:
            checksum: Function signature checksum
            func_name: Function name
            cache_data: Cache data to store
        """
        local_cache_path = self._get_local_cache_path(checksum, func_name)
        
        try:
            with open(local_cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved to local cache: {func_name}")
        except Exception as e:
            logger.error(f"Failed to save to local cache for {func_name}: {e}")
    
    def get(self, func: Callable, args: tuple, kwargs: dict) -> tuple[bool, Any]:
        """
        Get cached result if available, checking local cache first, then GCS.
        
        Args:
            func: Function to check cache for
            args: Function positional arguments
            kwargs: Function keyword arguments
            
        Returns:
            Tuple of (cache_hit: bool, result: Any)
        """
        checksum = self._compute_checksum(func, args, kwargs)
        
        # First check local cache
        local_hit, local_result = self._get_from_local_cache(checksum, func.__name__)
        if local_hit:
            return True, local_result
        
        # Local cache miss, try GCS
        cache_key = self._get_cache_key(checksum, func.__name__)
        
        try:
            blob = self.bucket.blob(cache_key)
            if blob.exists():
                # Download and deserialize
                data = blob.download_as_bytes()
                cache_data = pickle.loads(data)
                
                # Save to local cache for next time
                self._put_to_local_cache(checksum, func.__name__, cache_data)

                if cache_data.get('type') == 'directory':
                    # Extract tar archive to temporary directory
                    logger.info(f"GCS cache HIT for {func.__name__} - extracting directory (key: {cache_key})")
                    temp_dir = self._untar_to_temp_directory(cache_data['tar_data'])
                    logger.info(f"Directory extracted to temporary path: {temp_dir}")
                    return True, temp_dir
                else:
                    # Regular cached result
                    logger.info(f"GCS cache HIT for {func.__name__} (key: {cache_key})")
                    return True, cache_data['result']
            else:
                logger.info(f"Cache MISS for {func.__name__} (key: {cache_key})")
                return False, None
        except Exception as e:
            logger.error(f"GCS cache error for {func.__name__}: {e}")
            return False, None
    
    def _is_directory_path(self, result: Any) -> bool:
        """Check if result is a string path pointing to an existing directory."""
        return (isinstance(result, (str, Path)) and 
                Path(result).exists() and 
                Path(result).is_dir())
    
    def _tar_directory(self, dir_path: Union[str, Path]) -> bytes:
        """Create a tar.gz archive of directory and return as bytes."""
        dir_path = Path(dir_path)
        
        with tempfile.NamedTemporaryFile() as temp_file:
            with tarfile.open(temp_file.name, 'w') as tar:
                # Add all contents of the directory recursively
                tar.add(dir_path, arcname='.')
            
            temp_file.seek(0)
            return temp_file.read()
    
    def _untar_to_temp_directory(self, tar_data: bytes) -> str:
        """Extract tar data to temporary directory and return path."""
        temp_dir = tempfile.mkdtemp(prefix='dcagent_cache_')
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(tar_data)
            temp_file.seek(0)
            
            with tarfile.open(temp_file.name, 'r') as tar:
                tar.extractall(temp_dir)
        
        return temp_dir

    def put(self, func: Callable, args: tuple, kwargs: dict, result: Any) -> None:
        """
        Store function result in both GCS and local cache.
        
        Args:
            func: Function that was called
            args: Function positional arguments
            kwargs: Function keyword arguments
            result: Function result to cache
        """
        checksum = self._compute_checksum(func, args, kwargs)
        cache_key = self._get_cache_key(checksum, func.__name__)
        
        try:
            # Check if result is a directory path
            if self._is_directory_path(result):
                logger.info(f"Tarring directory {result} for caching...")
                # Create tar archive of directory
                tar_data = self._tar_directory(result)

                # Store metadata indicating this is a directory
                cache_data = {
                    'type': 'directory',
                    'original_path': str(result),
                    'tar_data': tar_data
                }
                logger.info(f"Cached directory {result} as tar archive (key: {cache_key})")
            else:
                # Regular result - serialize as before
                cache_data = {
                    'type': 'regular',
                    'result': result
                }
                logger.info(f"Cached result for {func.__name__} (key: {cache_key})")

            # Save to local cache first
            self._put_to_local_cache(checksum, func.__name__, cache_data)

            # Upload to GCS
            data = pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)
            blob = self.bucket.blob(cache_key)
            blob.upload_from_string(data, content_type='application/octet-stream')
            logger.info(f"Uploaded to GCS: {cache_key}")

        except Exception as e:
            logger.error(f"Failed to cache result for {func.__name__}: {e}")


def _resolve_credentials_path(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return explicit_path
    if _DEFAULT_CREDENTIALS_PATH:
        return _DEFAULT_CREDENTIALS_PATH
    return os.environ.get("GCS_CREDENTIALS_PATH")


def gcs_cache(bucket_name: str = "dcagent", credentials_path: Optional[str] = None, local_cache_dir: Optional[str] = None):
    """
    Decorator that caches function results in Google Cloud Storage with local caching.
    
    Args:
        bucket_name: GCS bucket name for caching (defaults to "dcagent")
        credentials_path: Path to GCS service account credentials JSON
        local_cache_dir: Directory for local cache (defaults to LOCAL_DATA_CACHE env var or ~/.dcagent_cache)
        
    Returns:
        Decorated function with GCS and local caching
        
    Example:
        @gcs_cache()  # Uses "dcagent" bucket and LOCAL_DATA_CACHE or ~/.dcagent_cache
        def expensive_function(x, y):
            # Some expensive computation
            return x * y + complex_calculation()
    """
    credentials_config = credentials_path
    sentinel = object()
    state = {
        "cache": None,
        "disabled": False,
        "last_credentials": sentinel,
    }

    def _ensure_cache(func_name: str) -> Optional[GCSCache]:
        resolved_credentials = _resolve_credentials_path(credentials_config)
        if state["disabled"] and state["last_credentials"] != resolved_credentials:
            state["disabled"] = False
            state["cache"] = None

        if state["disabled"]:
            return None

        # Recreate cache if credentials path changed
        if state["cache"] is not None and state["last_credentials"] != resolved_credentials:
            state["cache"] = None
        # When no cache yet, attempt to create
        if state["cache"] is None:
            try:
                cache_obj = GCSCache(bucket_name, resolved_credentials, local_cache_dir)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"GCS cache disabled for {func_name}; falling back to no-op caching: {exc}")
                state["disabled"] = True
                state["cache"] = None
                state["last_credentials"] = resolved_credentials
                return None
            state["cache"] = cache_obj
            state["last_credentials"] = resolved_credentials

        return state["cache"]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_obj = _ensure_cache(func.__name__)
            wrapper._cache = state["cache"]
            if cache_obj is None:
                return func(*args, **kwargs)
            # Try to get from cache first
            try:
                cache_hit, cached_result = cache_obj.get(func, args, kwargs)
            except Exception as exc:
                print(f"GCS cache error during get for {func.__name__}; disabling cache: {exc}")
                state["cache"] = None
                state["disabled"] = True
                wrapper._cache = None
                state["last_credentials"] = _resolve_credentials_path(credentials_config)
                return func(*args, **kwargs)
            
            if cache_hit:
                return cached_result
            
            # Cache miss - run function and store result
            result = func(*args, **kwargs)
            
            try:
                cache_obj.put(func, args, kwargs, result)
            except Exception as exc:
                print(f"GCS cache error during put for {func.__name__}; disabling cache: {exc}")
                state["cache"] = None
                state["disabled"] = True
                wrapper._cache = None
                state["last_credentials"] = _resolve_credentials_path(credentials_config)
            
            return result
        
        # Add cache management methods to wrapped function
        wrapper._cache = state["cache"]
        wrapper._original_func = func
        
        return wrapper
    
    return decorator


_initial_credentials = os.environ.get("GCS_CREDENTIALS_PATH")
if _initial_credentials:
    set_default_credentials_path(_initial_credentials)


class GCSCacheHDF5:
    """Google Cloud Storage cache client for HDF5 files with local caching."""

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None, local_cache_dir: Optional[str] = None):
        """
        Initialize GCS cache client for HDF5 files.

        Args:
            bucket_name: GCS bucket name for caching
            credentials_path: Path to GCS service account credentials JSON
            local_cache_dir: Directory for local cache (defaults to LOCAL_DATA_CACHE env var or ~/.dcagent_cache)
        """
        self.bucket_name = bucket_name

        # Set up local cache directory
        if local_cache_dir is None:
            local_cache_dir = os.environ.get("LOCAL_DATA_CACHE", os.path.expanduser("~/.dcagent_cache"))
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def _compute_directory_hash(self, dir_path: Union[str, Path]) -> str:
        """Compute hash of directory contents."""
        dir_path = Path(dir_path)
        hasher = hashlib.sha256()

        all_files = sorted(dir_path.rglob('*'))

        for file_path in all_files:
            if file_path.is_file():
                rel_path = file_path.relative_to(dir_path)
                hasher.update(str(rel_path).encode())

                try:
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                except Exception:
                    pass

        return hasher.hexdigest()

    def _compute_hdf5_hash(self, hdf5_path: Union[str, Path]) -> str:
        """
        Compute hash of HDF5 file contents.

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            SHA256 hash of HDF5 file contents
        """
        hdf5_path = Path(hdf5_path)
        hasher = hashlib.sha256()

        try:
            with h5py.File(hdf5_path, 'r') as f:
                def visit_item(name, obj):
                    """Visit each item in HDF5 file and update hash."""
                    # Hash the path/name
                    hasher.update(name.encode())

                    if isinstance(obj, h5py.Dataset):
                        # Hash dataset shape and dtype
                        hasher.update(str(obj.shape).encode())
                        hasher.update(str(obj.dtype).encode())

                        # Hash dataset contents
                        # For large datasets, we read in chunks
                        try:
                            data = obj[()]
                            if hasattr(data, 'tobytes'):
                                hasher.update(data.tobytes())
                            else:
                                hasher.update(str(data).encode())
                        except Exception as e:
                            # If we can't read the dataset, hash the error
                            hasher.update(str(e).encode())

                        # Hash attributes
                        for attr_name, attr_value in obj.attrs.items():
                            hasher.update(attr_name.encode())
                            hasher.update(str(attr_value).encode())

                    elif isinstance(obj, h5py.Group):
                        # Hash group attributes
                        for attr_name, attr_value in obj.attrs.items():
                            hasher.update(attr_name.encode())
                            hasher.update(str(attr_value).encode())

                # Visit all items in the HDF5 file
                f.visititems(visit_item)

        except Exception as e:
            logger.warning(f"Error computing HDF5 hash for {hdf5_path}: {e}")
            # Fallback to hashing the file directly
            with open(hdf5_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

        return hasher.hexdigest()

    def _process_arg_for_checksum(self, arg: Any) -> Any:
        """Process an argument for checksum computation."""
        if isinstance(arg, (str, Path)):
            if isinstance(arg, str) and ('/' in arg or '\\' in arg) and len(arg) < 500:
                try:
                    path = Path(arg)
                    if path.exists():
                        if path.is_dir():
                            dir_hash = self._compute_directory_hash(path)
                            logger.info(f"Computing content hash for directory {path}: {dir_hash[:8]}...")
                            return ('__DIR_CONTENT_HASH__', dir_hash)
                        elif path.is_file() and (path.suffix.lower() in ['.h5', '.hdf5', '.hdf']):
                            hdf5_hash = self._compute_hdf5_hash(path)
                            logger.info(f"Computing content hash for HDF5 file {path}: {hdf5_hash[:8]}...")
                            return ('__HDF5_CONTENT_HASH__', hdf5_hash)
                except (OSError, ValueError):
                    pass
            elif isinstance(arg, Path):
                try:
                    if arg.exists():
                        if arg.is_dir():
                            dir_hash = self._compute_directory_hash(arg)
                            logger.info(f"Computing content hash for directory {arg}: {dir_hash[:8]}...")
                            return ('__DIR_CONTENT_HASH__', dir_hash)
                        elif arg.is_file() and (arg.suffix.lower() in ['.h5', '.hdf5', '.hdf']):
                            hdf5_hash = self._compute_hdf5_hash(arg)
                            logger.info(f"Computing content hash for HDF5 file {arg}: {hdf5_hash[:8]}...")
                            return ('__HDF5_CONTENT_HASH__', hdf5_hash)
                except (OSError, ValueError):
                    pass
        elif isinstance(arg, (list, tuple)):
            return type(arg)(self._process_arg_for_checksum(item) for item in arg)
        elif isinstance(arg, dict):
            return {k: self._process_arg_for_checksum(v) for k, v in arg.items()}

        return arg

    def _compute_checksum(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Compute checksum for function signature and arguments."""
        try:
            func_source = inspect.getsource(func)
        except OSError:
            func_source = f"{func.__module__}.{func.__name__}"

        processed_args = tuple(self._process_arg_for_checksum(arg) for arg in args)
        processed_kwargs = {k: self._process_arg_for_checksum(v) for k, v in kwargs.items()}

        signature_data = {
            'function_name': func.__name__,
            'function_source': func_source,
            'args': processed_args,
            'kwargs': processed_kwargs
        }

        serialized = pickle.dumps(signature_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()

    def _get_cache_key(self, checksum: str, func_name: str) -> str:
        """Generate GCS object key for HDF5 cache entry."""
        return f"dcagent_cache_hdf5/{func_name}/{checksum}.h5"

    def _get_local_cache_path(self, checksum: str, func_name: str) -> Path:
        """Generate local cache file path for HDF5 cache entry."""
        cache_dir = self.local_cache_dir / f"{func_name}_hdf5"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{checksum}.h5"

    def _get_from_local_cache(self, checksum: str, func_name: str) -> tuple[bool, Any]:
        """Get cached HDF5 file from local cache if available."""
        local_cache_path = self._get_local_cache_path(checksum, func_name)

        if local_cache_path.exists():
            logger.info(f"Local cache HIT for {func_name} - HDF5 file at {local_cache_path}")
            return True, str(local_cache_path)

        return False, None

    def get(self, func: Callable, args: tuple, kwargs: dict) -> tuple[bool, Any]:
        """
        Get cached HDF5 file if available, checking local cache first, then GCS.

        Returns:
            Tuple of (cache_hit: bool, hdf5_path: str)
        """
        checksum = self._compute_checksum(func, args, kwargs)

        # First check local cache
        local_hit, local_result = self._get_from_local_cache(checksum, func.__name__)
        if local_hit:
            return True, local_result

        # Local cache miss, try GCS
        cache_key = self._get_cache_key(checksum, func.__name__)

        try:
            blob = self.bucket.blob(cache_key)
            if blob.exists():
                # Download HDF5 file to local cache
                local_cache_path = self._get_local_cache_path(checksum, func.__name__)
                blob.download_to_filename(str(local_cache_path))

                logger.info(f"GCS cache HIT for {func.__name__} - downloaded HDF5 to {local_cache_path}")
                return True, str(local_cache_path)
            else:
                logger.info(f"Cache MISS for {func.__name__} (key: {cache_key})")
                return False, None
        except Exception as e:
            logger.error(f"GCS cache error for {func.__name__}: {e}")
            return False, None

    def put(self, func: Callable, args: tuple, kwargs: dict, result: Any) -> None:
        """
        Store HDF5 file in both GCS and local cache.

        Args:
            func: Function that was called
            args: Function positional arguments
            kwargs: Function keyword arguments
            result: Path to HDF5 file to cache
        """
        checksum = self._compute_checksum(func, args, kwargs)
        cache_key = self._get_cache_key(checksum, func.__name__)

        try:
            # Validate that result is a path to an HDF5 file
            if not isinstance(result, (str, Path)):
                logger.error(f"Result is not a file path for {func.__name__}, skipping HDF5 cache")
                return

            result_path = Path(result)
            if not result_path.exists():
                logger.error(f"Result path does not exist: {result_path}")
                return

            if not result_path.is_file():
                logger.error(f"Result path is not a file: {result_path}")
                return

            # Copy to local cache
            local_cache_path = self._get_local_cache_path(checksum, func.__name__)
            shutil.copy2(result_path, local_cache_path)
            logger.info(f"Saved HDF5 to local cache: {local_cache_path}")

            # Upload to GCS
            blob = self.bucket.blob(cache_key)
            blob.upload_from_filename(str(result_path))
            logger.info(f"Uploaded HDF5 to GCS: {cache_key}")

        except Exception as e:
            logger.error(f"Failed to cache HDF5 for {func.__name__}: {e}")


def gcs_cache_hdf5(bucket_name: str = "dcagent", credentials_path: Optional[str] = None, local_cache_dir: Optional[str] = None):
    """
    Decorator that caches HDF5 file results in Google Cloud Storage with local caching.

    The decorated function must return a path (str or Path) to an HDF5 file.

    Args:
        bucket_name: GCS bucket name for caching (defaults to "dcagent")
        credentials_path: Path to GCS service account credentials JSON
        local_cache_dir: Directory for local cache (defaults to LOCAL_DATA_CACHE env var or ~/.dcagent_cache)

    Returns:
        Decorated function with GCS and local caching for HDF5 files

    Example:
        @gcs_cache_hdf5()
        def create_hdf5_dataset(data_path):
            # Create HDF5 file
            h5_path = "/tmp/dataset.h5"
            with h5py.File(h5_path, 'w') as f:
                # ... populate HDF5 file
            return h5_path
    """
    credentials_config = credentials_path
    sentinel = object()
    state = {
        "cache": None,
        "disabled": False,
        "last_credentials": sentinel,
    }

    def _ensure_cache(func_name: str) -> Optional[GCSCacheHDF5]:
        resolved_credentials = _resolve_credentials_path(credentials_config)
        if state["disabled"] and state["last_credentials"] != resolved_credentials:
            state["disabled"] = False
            state["cache"] = None

        if state["disabled"]:
            return None

        if state["cache"] is not None and state["last_credentials"] != resolved_credentials:
            state["cache"] = None

        if state["cache"] is None:
            try:
                cache_obj = GCSCacheHDF5(bucket_name, resolved_credentials, local_cache_dir)
            except Exception as exc:
                print(f"GCS HDF5 cache disabled for {func_name}; falling back to no-op caching: {exc}")
                state["disabled"] = True
                state["cache"] = None
                state["last_credentials"] = resolved_credentials
                return None
            state["cache"] = cache_obj
            state["last_credentials"] = resolved_credentials

        return state["cache"]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_obj = _ensure_cache(func.__name__)
            wrapper._cache = state["cache"]
            if cache_obj is None:
                return func(*args, **kwargs)

            # Try to get from cache first
            try:
                cache_hit, cached_result = cache_obj.get(func, args, kwargs)
            except Exception as exc:
                print(f"GCS HDF5 cache error during get for {func.__name__}; disabling cache: {exc}")
                state["cache"] = None
                state["disabled"] = True
                wrapper._cache = None
                state["last_credentials"] = _resolve_credentials_path(credentials_config)
                return func(*args, **kwargs)

            if cache_hit:
                return cached_result

            # Cache miss - run function and store result
            result = func(*args, **kwargs)

            try:
                cache_obj.put(func, args, kwargs, result)
            except Exception as exc:
                print(f"GCS HDF5 cache error during put for {func.__name__}; disabling cache: {exc}")
                state["cache"] = None
                state["disabled"] = True
                wrapper._cache = None
                state["last_credentials"] = _resolve_credentials_path(credentials_config)

            return result

        wrapper._cache = state["cache"]
        wrapper._original_func = func

        return wrapper

    return decorator

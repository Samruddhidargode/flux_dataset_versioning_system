"""File locking utilities for FLUX versioning system.

Provides filesystem-based locking using exclusive file creation
for safe concurrent access to the repository.
"""

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Optional, Union

from flux.exceptions import LockError, LockTimeoutError

logger = logging.getLogger(__name__)


class FileLock:
    """A filesystem-based lock using exclusive file creation.

    Uses atomic file creation (O_CREAT | O_EXCL on POSIX, 'x' mode fallback
    on Windows) to implement mutual exclusion. Lock files contain PID and
    hostname information for debugging.

    Args:
        lock_dir: Path to the locks directory.
        lock_name: Name of the lock (used as filename).
        timeout: Maximum seconds to wait for the lock (default 10).
        retry_interval: Initial retry interval in seconds (default 0.1).
    """

    def __init__(
        self,
        lock_dir: Union[str, Path],
        lock_name: str,
        timeout: float = 10.0,
        retry_interval: float = 0.1,
    ):
        self.lock_dir = Path(lock_dir)
        self.lock_name = lock_name
        self.lock_path = self.lock_dir / f"{lock_name}.lock"
        self.timeout = timeout
        self.retry_interval = retry_interval
        self._acquired = False

    def acquire(self) -> None:
        """Acquire the lock with exponential backoff.

        Raises:
            LockTimeoutError: If the lock cannot be acquired within timeout.
            LockError: If an unexpected error occurs during locking.
        """
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        interval = self.retry_interval

        while True:
            try:
                self._try_create_lock()
                self._acquired = True
                logger.debug("Lock acquired: %s", self.lock_path)
                return
            except FileExistsError:
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    raise LockTimeoutError(
                        f"Could not acquire lock '{self.lock_name}' within "
                        f"{self.timeout}s. Lock file: {self.lock_path}"
                    )
                logger.debug(
                    "Lock '%s' is held, retrying in %.2fs (elapsed: %.2fs)",
                    self.lock_name, interval, elapsed,
                )
                time.sleep(interval)
                interval = min(interval * 2, 1.0)  # Exponential backoff, cap at 1s
            except Exception as e:
                raise LockError(f"Unexpected error acquiring lock '{self.lock_name}': {e}")

    def _try_create_lock(self) -> None:
        """Attempt to create the lock file atomically.

        Raises:
            FileExistsError: If the lock file already exists.
        """
        lock_info = {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "timestamp": time.time(),
        }
        lock_content = json.dumps(lock_info)

        try:
            # Try POSIX atomic creation first
            fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(fd, lock_content.encode("utf-8"))
            os.close(fd)
        except AttributeError:
            # Fallback for systems where O_CREAT/O_EXCL not available
            with open(self.lock_path, "x") as f:
                f.write(lock_content)

    def release(self) -> None:
        """Release the lock by removing the lock file."""
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
                logger.debug("Lock released: %s", self.lock_path)
            except OSError as e:
                logger.warning("Failed to remove lock file %s: %s", self.lock_path, e)
            finally:
                self._acquired = False

    def __enter__(self) -> "FileLock":
        """Context manager entry: acquire the lock."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: release the lock."""
        self.release()

    def __del__(self) -> None:
        """Destructor: release lock if still held."""
        if self._acquired:
            self.release()


def acquire_lock(
    lock_dir: Union[str, Path],
    lock_name: str,
    timeout: float = 10.0,
) -> FileLock:
    """Create and acquire a file lock.

    Convenience function for acquiring a lock outside of a context manager.

    Args:
        lock_dir: Path to the locks directory.
        lock_name: Name of the lock.
        timeout: Maximum seconds to wait.

    Returns:
        An acquired FileLock instance.

    Raises:
        LockTimeoutError: If the lock cannot be acquired within timeout.
    """
    lock = FileLock(lock_dir, lock_name, timeout=timeout)
    lock.acquire()
    return lock

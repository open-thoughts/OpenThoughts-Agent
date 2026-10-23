"""Reclaim disposable pip HTTP downloads before grading a CLI task."""

import json
import shlex

from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import Verifier


PIP_HTTP_CACHE_RECLAIM = r"""
import errno
import json
import os
import shutil
import stat

flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open('/', flags)
result = {'cache': '/root/.cache/pip/http-v2', 'removed': False}
try:
    for part in ('root', '.cache', 'pip'):
        child = os.open(part, flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    entry = os.stat('http-v2', dir_fd=descriptor, follow_symlinks=False)
    if stat.S_ISDIR(entry.st_mode):
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError('Descriptor-based safe cache removal unavailable')
        shutil.rmtree('http-v2', dir_fd=descriptor)
        result['removed'] = True
    else:
        result['skip_reason'] = 'cache is not a real directory'
except OSError as error:
    if error.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
        raise
    result['skip_reason'] = 'cache absent or path contains a symlink'
finally:
    os.close(descriptor)
print(json.dumps(result))
"""


class CliCacheVerifier(Verifier):
    """Run the normal verifier after safe, provider-specific HTTP-cache cleanup."""

    async def verify(self) -> VerifierResult:
        if self.task.name != "pytorch-model-cli":
            raise ValueError("CLI cache reclamation is scoped to pytorch-model-cli")
        if not isinstance(self.environment, DaytonaEnvironment):
            raise TypeError("CLI cache reclamation requires a Daytona sandbox")
        sandbox = self.environment._sandbox
        if sandbox is None:
            raise RuntimeError("CLI sandbox is not running")
        # The direct exec endpoint does not create a persistent exec-session
        # directory, which is precisely what fails when package downloads fill
        # the sandbox. Descriptor-relative removal rejects symlinked ancestors.
        response = await sandbox.process.exec(
            "python3 -c " + shlex.quote(PIP_HTTP_CACHE_RECLAIM), timeout=20
        )
        if response.exit_code != 0:
            raise RuntimeError(f"Pip HTTP-cache reclamation failed: {response.result}")
        receipt = json.loads(response.result)
        (self.trial_paths.verifier_dir / "pip-http-cache-reclamation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return await super().verify()

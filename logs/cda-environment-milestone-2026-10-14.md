# CDA environment/operator milestone — 2026-10-14

Status: **PASS — environment and CUDA operator milestone complete.** Full
CDA-VSR inference was not run. The earlier blocked-session notes below are
retained as history and superseded by these continuation results.

## Approved host continuation — 2026-10-14

- Effective permissions: workspace writes are allowed; host execution requires
  on-request approval. Explicit `require_escalated` requests succeeded for host
  diagnostics, interpreter installation, setup, and the corrected setup retry.
- Host `nvidia-smi` succeeded: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB,
  driver 615.71.09. Existing `.venv` torch 2.14.0+cu130 reported CUDA available.
  `curl -I` against the MMCV cu121/torch2.1 index returned HTTP 200.
- Installed uv 0.13.1 into `/tmp/cda-bootstrap` using pip's `--target`, then
  installed standalone CPython 3.10.22 with uv's `--no-bin` option at
  `/home/ibexe/.local/share/cda-python/cpython-3.10.22-linux-x86_64-gnu`.
  No system Python replacement or PATH change was performed.
- Inspected both scripts before execution. The first setup exited 1: MMCV's
  dependency resolution replaced NumPy 1.26.4 with 2.2.6, and bundled pip 23.0.1
  rejected normalized package names on the PyTorch index, selecting
  typing_extensions 3.10.0.2. Torch import failed because `Self` was absent,
  with an additional NumPy ABI warning. `pip check` alone did not detect these
  runtime incompatibilities. Full output: `cda-setup-2026-10-14.log`.
- Corrected `scripts/setup_cda_env.sh` to install pip 25.0.1 inside `.venv-cda`,
  pin typing_extensions 4.15.0 and OpenCV 4.11.0.86, and explicitly retain
  NumPy 1.26.4 when resolving MMCV dependencies. Binary-only installation remains
  enforced. `scripts/check_cda_operator.py` is unchanged.
- Corrected setup exited **0**, including `pip check` (no broken requirements)
  and the CUDA operator test. Full output: `cda-setup-retry-2026-10-14.log`.
  Package snapshot: `cda-environment-freeze-2026-10-14.txt`.

Verified stack: Python **3.10.22**, torch **2.1.0+cu121**, torchvision
**0.16.0+cu121**, NumPy **1.26.4**, MMCV **2.2.0**, CUDA runtime **12.1**.

Actual operator results on the RTX 3050:

- 64 input/output channels, kernel size 1, deform_groups 4, input 1×64×16×16.
- CUDA forward and backward passed; input, offset, mask, weight and bias
  gradients were present, finite and nonzero.
- Zero offsets/unit masks matched ordinary convolution with the same parameters:
  **maximum absolute error 0.0** (rtol 1e-4, atol 1e-5).
- **Peak allocated GPU memory: 21,619,712 bytes** (PyTorch allocator metric).

Preservation checks: original `.venv` pip-freeze snapshots before/after are
identical. Both checkpoint SHA-256 checks passed. The operator script hash is
unchanged; the setup script hash changed intentionally for the dependency fix.
Both external repositories remain clean at the HEADs recorded below. Root Git
status remains unavailable because its exposed `.git` is not a valid repository.
No commands modified system Python, drivers, system CUDA, or checkpoints.

To rerun the operator in a GPU-accessible host shell:

```bash
.venv-cda/bin/python scripts/check_cda_operator.py
```

Stopped after this operator milestone. This does not validate CDA model
integration or full inference.

## Earlier blocked session (historical)

## Inspection

- No AGENTS.md found in the project or searched parent tree. README.md and
  requirements.txt are empty. .agents and .codex are empty.
- Root `git status --short` fails: the exposed .git directory is empty, so root
  user changes cannot be assessed through Git. Existing files were left intact;
  only the two new scripts and this report were added.
- `git -C external/nanovsr status --short` and the corresponding CDA command
  produce no changes, including after this work. HEADs remain respectively
  b48e2bad89d01e22226eb8613bee25cb45623fae and
  5630821e5df4e110878d0caae39bcc53f658e9a9.
- Both checkpoint files exist: NanoVSR approximately 1.1 MiB, CDA approximately
  24 MiB. Neither was modified.
- Only .venv exists; .venv-cda, uv, conda, Python 3.10 and Python 3.11 were absent.
  System executables include Python 3.12 and 3.14. No matching installation
  assets were found in the inspected /tmp and user cache paths.
- .venv inspection: Python 3.12.14, torch 2.14.0+cu130, torchvision 0.29.0+cu130,
  CUDA runtime 13.0, NumPy 2.5.3, setuptools 84.0.0; MMCV absent.
  `torch.cuda.is_available()` is **False in this session**.
- `nvidia-smi` exits 9: cannot communicate with the NVIDIA driver.
  No /dev/nvidia* entries are exposed here. This does not establish that the
  driver is broken on the host; this execution session lacks working GPU access.
- `curl -I --max-time 15 https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`
  exits 6: could not resolve host. Shell network access is restricted and cannot
  fetch the missing interpreter/packages here.
- Browser access verified that the official index lists the Linux cp310
  MMCV 2.2.0 wheel. Browser access does not provide shell package-download access.

## Added commands and actual results

`bash scripts/setup_cda_env.sh`

Exits 127: `python3.10: command not found`, before creating any environment or
installing packages. The script accepts CDA_PYTHON for an existing Python 3.10,
preserves .venv, pins the candidate stack, requires binary packages, and installs
MMCV from its explicit official wheel URL. It has not been installation-tested.

`.venv/bin/python scripts/check_cda_operator.py`

Prints the existing environment versions, then exits 1 with
`RuntimeError: CUDA unavailable: operator verification cannot run`.
This is a diagnostic execution under the existing environment, not validation
of the proposed CDA stack.

`bash -n scripts/setup_cda_env.sh` and AST parsing of the Python script pass.
These establish syntax only.

## Next steps to finish this milestone

1. Run in a session with GPU device/driver access and package-download access.
   Recheck nvidia-smi and CUDA availability; do not change system drivers here.
2. Obtain a separate user-local Python 3.10 (for example a managed standalone
   interpreter), without replacing system Python or the working .venv.
3. From fish, run `env CDA_PYTHON=/absolute/path/to/python3.10 bash scripts/setup_cda_env.sh`.
   Capture its output and `.venv-cda/bin/python -m pip freeze` on successful setup.
4. The operator script verifies the exact requested versions, CUDA forward and
   backward with 64 input/output channels, kernel_size=1, deform_groups=4, finite
   nonzero input/offset/mask/parameter gradients, and zero offsets/unit masks
   against ordinary convolution using the same weights/bias. It reports maximum
   comparison error and peak allocated GPU memory. All numerical results remain
   unmeasured until this script passes in .venv-cda.
5. Stop after this operator milestone. Full CDA inference and its additional
   repository dependencies must be investigated separately. The CDA alignment
   source doubles its configured input/output channels internally, so this
   requested 64-channel operator case alone does not validate model integration.

References:
- https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
- https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.ModulatedDeformConv2d.html

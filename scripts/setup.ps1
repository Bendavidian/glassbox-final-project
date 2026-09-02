<#
.SYNOPSIS
    Create the virtual environment and install GlassBox Trader from the lockfile.

.DESCRIPTION
    **A documented procedure a human must execute correctly is a note; a script is a
    mechanism** (CLAUDE.md 3). This exists because the author of that principle skipped
    step 3 of his own README while running the GB-59 reproducibility audit on
    23 Aug 2026 - the editable install - and would have audited his own install procedure
    rather than the documented one.

    It runs the three commands the README lists, in order, stopping at the first failure,
    and then **verifies the result** rather than assuming it: that `import glassbox`
    resolves and that the package is installed in editable mode. An install that is
    silently non-editable is the failure this catches - the imports work from the repo
    root because Python puts the working directory on the path, and break the moment
    anything runs from elsewhere.

.PARAMETER VenvPath
    Where to create the environment. Defaults to `.venv` beside this repository.

.PARAMETER SkipVenv
    Install into the interpreter already on PATH instead of creating an environment.
    For CI, which supplies its own.

.EXAMPLE
    .\scripts\setup.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvPath = ".venv",
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# ── 1. the environment ───────────────────────────────────────────────────────
if ($SkipVenv) {
    $python = (Get-Command python).Source
    Write-Host "using the interpreter on PATH: $python"
} else {
    if (-not (Test-Path $VenvPath)) {
        Step "create the virtual environment at $VenvPath" { python -m venv $VenvPath }
    } else {
        Write-Host "reusing the existing environment at $VenvPath"
    }
    $python = Join-Path $repo "$VenvPath\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        Write-Host "FAILED: no interpreter at $python" -ForegroundColor Red
        exit 1
    }
}

# The version gate is here rather than only in the README: the pinned numpy and scipy
# both declare requires_python >= 3.12, so on 3.11 the install fails deep inside pip
# with a resolver message that does not name the cause.
$version = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]"3.12") {
    Write-Host "FAILED: Python $version. This project needs 3.12 or newer - the pinned numpy and scipy both require it." -ForegroundColor Red
    exit 1
}
Write-Host "Python $version"

# ── 2. the three commands the README lists ───────────────────────────────────
Step "upgrade pip" { & $python -m pip install --upgrade pip --quiet }
Step "install pinned dependencies from requirements.lock" {
    & $python -m pip install -r requirements.lock --quiet
}
# `--no-deps`: requirements.lock is the authority on versions, and letting the project
# metadata resolve again could quietly install something the lockfile did not pin.
Step "install glassbox in editable mode" {
    & $python -m pip install -e ".[dev]" --no-deps --quiet
}

# ── 3. verify, rather than assume ────────────────────────────────────────────
Write-Host ""
Write-Host "==> verify" -ForegroundColor Cyan

& $python -c "import glassbox"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: 'import glassbox' does not resolve after install." -ForegroundColor Red
    exit 1
}
Write-Host "  import glassbox                OK"

# An editable install resolves to the repository, not to site-packages. A non-editable
# one still imports from the repo root - Python puts the working directory on the path -
# and breaks the moment anything runs from elsewhere, which is the silent half.
$located = & $python -c "import glassbox, pathlib; print(pathlib.Path(glassbox.__file__).resolve().parent.parent)"
if ($located -ne $repo) {
    Write-Host "FAILED: glassbox imports from '$located', not from '$repo'." -ForegroundColor Red
    Write-Host "        The editable install did not take; a run from any other directory would use the wrong copy." -ForegroundColor Red
    exit 1
}
Write-Host "  editable, resolving to repo   OK"

& $python -c "import pytest, torch, scipy, matplotlib, pandas_market_calendars" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: a pinned dependency is missing after install." -ForegroundColor Red
    exit 1
}
Write-Host "  every heavyweight dependency  OK"

Write-Host ""
Write-Host "Setup complete. Next: $VenvPath\Scripts\pytest.exe" -ForegroundColor Green
exit 0

# Starts the local model server (when it is not already running) and the Moments web app.
#   .\start.ps1               open http://127.0.0.1:8765 in your browser
#   .\start.ps1 -Port 9000 -NoBrowser
param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Create the environment first: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
}

function Test-Ollama {
    try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; return $true } catch { return $false }
}

# Prefer the portable runtime kept in local_data, then a system-wide Ollama install.
$runtime = Join-Path $PSScriptRoot 'local_data\ollama_runtime\ollama.exe'
$portable = Test-Path $runtime
if (-not $portable) {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $command) { throw 'Install Ollama from https://ollama.com/download/windows, then rerun .\start.ps1' }
    $runtime = $command.Source
}
$env:OLLAMA_HOST = '127.0.0.1:11434'

if (-not (Test-Ollama)) {
    if ($portable) { $env:OLLAMA_MODELS = Join-Path $PSScriptRoot 'local_data\ollama_models' }
    $log = Join-Path $PSScriptRoot 'local_data\ollama-server.log'
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    Write-Host 'Starting the local Ollama server...'
    Start-Process -FilePath $runtime -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" | Out-Null
    for ($i = 0; $i -lt 40 -and -not (Test-Ollama); $i++) { Start-Sleep -Milliseconds 500 }
    if (-not (Test-Ollama)) { throw "Ollama did not start. See $log.err" }
}

# First run: download the models the pipeline needs (about 3.7 GB in total).
$required = [ordered]@{
    'qwen3.5:4b'        = 'planning, scene descriptions, and verification'
    'nomic-embed-text'  = 'scene and transcript search'
}
$installed = (Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags').models.name
foreach ($model in $required.Keys) {
    if ($installed -notcontains $model) {
        Write-Host "Downloading $model for $($required[$model])..."
        & $runtime pull $model
    }
}

$appArgs = @('-m', 'webapp', '--port', $Port)
if ($NoBrowser) { $appArgs += '--no-browser' }
& $python @appArgs

$ErrorActionPreference = 'Stop'
try {
    $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2
    Write-Host 'Ollama is already running on localhost:11434. You can launch the Streamlit app.'
    return
} catch {
    # No running server: start the project-local runtime below.
}
$runtime = Join-Path $PSScriptRoot 'local_data/ollama_runtime/ollama.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $command) { throw 'Install Ollama from https://ollama.com/download/windows, then rerun this script.' }
    $runtime = $command.Source
}
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $PSScriptRoot 'local_data/ollama_models'
Write-Host 'Starting local model server. Keep this terminal open.'
Write-Host 'First-time setup in another terminal: .\local_data\ollama_runtime\ollama.exe pull qwen2.5vl:3b'
Write-Host 'Also pull the planning model: .\local_data\ollama_runtime\ollama.exe pull qwen2.5:7b'
& $runtime serve

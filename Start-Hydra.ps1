param(
    [string]$Python = '.venv\Scripts\python.exe',
    [string]$ModelPath = '',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
$previousKey = $env:API_KEY
$previousModel = $env:MODEL_PATH
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Create the API virtual environment and install requirements.txt first, or pass -Python with its full path.'
    }
    if ([string]::IsNullOrEmpty($env:API_KEY)) {
        $privateKey = Read-Host 'Enter Team Hydra assigned API key (hidden)' -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($privateKey)
        try { $env:API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer); $privateKey.Dispose() }
    }
    if ([string]::IsNullOrEmpty($env:API_KEY)) { throw 'The assigned API key must not be empty.' }
    if ($ModelPath) { $env:MODEL_PATH = (Resolve-Path -LiteralPath $ModelPath).Path }
    Write-Host 'Starting Team Hydra API. The key is held in the process environment and is not printed or saved.'
    & $Python -m uvicorn app.main:app --host 0.0.0.0 --port $Port --workers 1 --no-access-log
    if ($LASTEXITCODE -ne 0) { throw "API process exited with code $LASTEXITCODE." }
}
finally {
    $env:API_KEY = $previousKey
    $env:MODEL_PATH = $previousModel
    Pop-Location
}

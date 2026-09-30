#!/usr/bin/env pwsh
<#
.SYNOPSIS
    run_all_pc.ps1 — Khởi động toàn bộ orchestrator-on-edge trên Windows PC

.DESCRIPTION
    Hỗ trợ 2 chế độ chạy:
    1. Host Mode (Mặc định):
       - Car Simulator đóng vai trò STUB CAR (thay thế phần cứng xe / CAN bus)
       - Đầu vào sử dụng MICROPHONE THẬT từ máy tính qua voice_processing
       - Toàn bộ backend chạy trên host Python với stubs phần cứng Jetson

    2. Docker Mode (-Docker):
       - Chạy backend services bằng Docker containers (docker-compose.x86.yml)
       - Car Simulator stub chạy trong container car_control
       - Giao diện mô phỏng xe SVG chạy trên port 8010
       - Voice processing chạy mic thật từ host kết nối tới orchestrator Docker

    Cả 2 chế độ KHÔNG làm thay đổi cách chạy trên Jetson AGX (l4t container/arm64).

.PARAMETER Docker
    Chạy backend bằng Docker containers thay vì chạy trên host Python

.PARAMETER SkipInstall
    Bỏ qua bước install dependencies (host mode)

.PARAMETER SkipVoice
    Không khởi động voice processing

.PARAMETER VoiceStub
    Dùng text input terminal giả lập voice thay vì microphone thật

.PARAMETER SkipUI
    Không khởi động car_control_ui (giao diện mô phỏng xe)

.PARAMETER NoBrowser
    Không tự động mở browser sau khi khởi động

.PARAMETER Reset
    Xóa và tạo lại virtual environment (host mode)

.PARAMETER Down
    Dừng tất cả container và process đang chạy rồi thoát

.EXAMPLE
    .\run_all_pc.ps1                     # Chạy host mode với microphone thật
    .\run_all_pc.ps1 -Docker             # Chạy backend Docker + mic thật trên host
    .\run_all_pc.ps1 -VoiceStub          # Chạy với bàn phím gõ lệnh (không dùng mic)
    .\run_all_pc.ps1 -Down               # Dừng toàn bộ
#>

param(
    [switch]$Docker,
    [switch]$SkipInstall,
    [switch]$SkipVoice,
    [switch]$VoiceStub,
    [switch]$SkipUI,
    [switch]$NoBrowser,
    [switch]$Reset,
    [switch]$Down
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ──────────────────────────────────────────────────────────────────────
$ROOT        = $PSScriptRoot
$STUBS_DIR   = Join-Path $ROOT "stubs"
$AGENTS_DIR  = Join-Path $ROOT "agents"
$ORCH_DIR    = Join-Path $ROOT "orchestrator"
$VOICE_DIR   = Join-Path $ROOT "voice_processing"
$UI_DIR      = Join-Path $ROOT "car_control_ui"
$VENV_BASE   = Join-Path $ROOT ".venvs"
$VENV_PC     = Join-Path $VENV_BASE "pc"
$ENV_FILE    = Join-Path $ROOT ".env.x86"
$COMPOSE_X86 = Join-Path $ROOT "docker-compose.x86.yml"
# Always use the amd64 env file so compose interpolation doesn't pick the
# Jetson .env by accident.
$COMPOSE_X86_ARGS = @("-f", $COMPOSE_X86)
if (Test-Path $ENV_FILE) { $COMPOSE_X86_ARGS += @("--env-file", $ENV_FILE) }

# ── Ports ──────────────────────────────────────────────────────────────────────
$PORT_ORCH   = 8000
$PORT_CC     = 8001
$PORT_CM     = 8002
$PORT_UI     = 8010

# ── Colors & Logging ──────────────────────────────────────────────────────────
function Write-Header { param([string]$msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Step   { param([string]$msg) Write-Host "  -> $msg" -ForegroundColor White }
function Write-Ok     { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn   { param([string]$msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err    { param([string]$msg) Write-Host "  [ERR] $msg" -ForegroundColor Red }

# ── Handle -Down ──────────────────────────────────────────────────────────────
if ($Down) {
    Write-Header "Stopping all services..."
    if (Test-Path $COMPOSE_X86) {
        Write-Step "Stopping Docker containers..."
        docker compose @COMPOSE_X86_ARGS down 2>$null | Out-Null
    }
    Write-Step "Stopping any host service launchers..."
    Get-Process -Name "python*", "pwsh*" -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -match "\[(car_control|car_manual|orchestrator|voice_stub|voice_processing|car_control_ui)\]"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Ok "All services stopped."
    exit 0
}

# ── Print Banner ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
if ($Docker) {
    Write-Host "   Orchestrator-on-Edge -- PC Docker Mode (-Docker)       " -ForegroundColor Yellow
} else {
    Write-Host "   Orchestrator-on-Edge -- PC Host Mode (Stub Car + Mic)  " -ForegroundColor Green
}
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Stub: Car Simulator (Doors, Windows, Trunk, AC)         " -ForegroundColor Gray
Write-Host "  Input: Real Microphone on Host via sounddevice          " -ForegroundColor Gray
Write-Host "  Jetson code unchanged -- compatible with AGX (arm64)`n"   -ForegroundColor Gray

# ── Function to start voice processing on host ────────────────────────────────
function Start-VoiceHost {
    param([string]$PythonExe)

    if ($VoiceStub) {
        Write-Header "Starting voice_stub (keyboard text terminal)..."
        $voiceScript = Join-Path $STUBS_DIR "voice_stub_main.py"
        $cmd = "`$host.UI.RawUI.WindowTitle = '[voice_stub]'; `$env:ORCHESTRATOR_URL = 'http://localhost:$PORT_ORCH'; & '$PythonExe' '$voiceScript'"
        Start-Process -FilePath "pwsh.exe" -ArgumentList "-NoProfile", "-NoExit", "-Command", $cmd
        Write-Ok "Started [voice_stub] in separate window"
    } else {
        Write-Header "Starting voice_processing (real microphone input)..."
        $voiceScript = Join-Path $VOICE_DIR "main.py"
        $launcherFile = Join-Path $VENV_BASE "voice_processing_launcher.ps1"
        $lines = @()
        $lines += "`$host.UI.RawUI.WindowTitle = '[voice_processing - Real Mic]'"
        $lines += "`$env:ORCHESTRATOR_URL = 'http://localhost:$PORT_ORCH'"
        $lines += "`$env:STT_BACKEND = 'whisper'"
        $lines += "`$env:STT_MODEL_TYPE = 'sense_voice'"
        $lines += "`$env:STT_LANGUAGE = 'auto'"
        $lines += "`$env:TTS_LANGUAGE = ''"
        $lines += "`$env:WHISPER_MODEL = 'large-v3'"
        $lines += "`$env:FASTER_WHISPER_DEVICE = 'cuda'"
        $lines += "`$env:FASTER_WHISPER_COMPUTE = 'float16'"
        # Same HF cache the x86 container uses (./cache/voice_processing -> /app/cache)
        # so the prefetched Whisper model is shared (fetch_assets asr-fw).
        $lines += "`$env:HF_HOME = '$ROOT\cache\voice_processing'"
        # cuBLAS/cuDNN for CTranslate2 (faster-whisper GPU) from the nvidia wheels
        $lines += "`$env:PATH = '$VENV_PC\Lib\site-packages\nvidia\cublas\bin;$VENV_PC\Lib\site-packages\nvidia\cudnn\bin;' + `$env:PATH"
        $lines += "`$env:STT_MODEL_DIR = '$VOICE_DIR\agent_assets\models\asr'"
        $lines += "`$env:STT_WHISPER_DIR = '$VOICE_DIR\agent_assets\models\asr_whisper'"
        $lines += "`$env:STT_WHISPER_MODEL = 'small'"
        $lines += "`$env:PIPER_VI_DIR = '$VOICE_DIR\agent_assets\models\tts\vits-piper-vi_VN-vais1000-medium'"
        $lines += "`$env:SPEAKER_ENABLED = '0'"
        $lines += "`$env:SPEAKER_DEVICE = 'cuda'"
        $lines += "`$env:SPEAKER_TSE_ENABLED = '0'"
        $lines += "`$env:TSE_MODEL = 'MossFormer2_SS_16K'"
        $lines += "`$env:WAKE_WORD_BACKEND = 'openwakeword'"
        $lines += "`$env:WAKE_WORD_MODEL_DIR = '$VOICE_DIR\agent_assets\models\kws'"
        $lines += "`$env:AUDIO_SAMPLE_RATE = '16000'"
        $lines += "`$env:AUDIO_CHUNK = '1024'"
        $lines += "`$env:AEC_ENABLED = '1'"
        $lines += "`$env:AEC_DELAY_MS = '60'"
        $lines += "`$env:AEC_NOISE_SUPPRESS = '1'"
        $lines += "`$env:AEC_NS_LEVEL = 'high'"
        $lines += "`$env:AEC_TRANSIENT_SUPPRESS = '1'"
        $lines += "`$env:AEC_AGC_ENABLED = '1'"
        $lines += "`$env:AEC_AGC_MAX_GAIN_DB = '30.0'"
        $lines += "`$env:AEC_AGC_INITIAL_GAIN_DB = '15.0'"
        $lines += "`$env:BARGE_IN_ENABLED = '1'"
        $lines += "`$env:BARGE_IN_MIN_PLAY_SEC = '1.0'"
        $lines += "`$env:SEGMENT_SILENCE = '1.0'"
        $lines += "`$env:TURN_END_SILENCE = '2.5'"
        $lines += "`$env:VAD_FIRST_TIMEOUT = '8.0'"
        $lines += "`$env:FOLLOWUP_LISTEN_SEC = '12.0'"
        $lines += "`$env:FOLLOWUP_GRACE_SEC = '0.2'"
        $lines += "`$env:SPEECH_PREROLL_SEC = '0.8'"
        $lines += "Set-Location '$VOICE_DIR'"
        $lines += "Write-Host '==================================================' -ForegroundColor Cyan"
        $lines += "Write-Host ' Voice Processing with REAL Microphone Active' -ForegroundColor Green"
        $lines += 'Write-Host " Say: \"Hey Dora\" or \"Hey Doh Ra\" to trigger" -ForegroundColor Yellow'
        $lines += "Write-Host '==================================================' -ForegroundColor Cyan"
        $lines += "& '$PythonExe' '$voiceScript'"
        $lines += "Write-Host 'voice_processing exited. Press Enter to close.' -ForegroundColor Yellow"
        $lines += "Read-Host"

        $lines | Out-File -FilePath $launcherFile -Encoding UTF8
        Start-Process -FilePath "pwsh.exe" -ArgumentList "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $launcherFile
        Write-Ok "Started [voice_processing] with real microphone in separate window"
    }
}

# ==============================================================================
# MODE 1: DOCKER MODE
# ==============================================================================
if ($Docker) {
    Write-Header "Checking Docker environment..."
    try {
        $dockerVer = & docker version --format '{{.Server.Version}}' 2>&1
        if ($LASTEXITCODE -ne 0) { throw }
        Write-Ok "Docker Desktop is running (Server v$dockerVer)"
    } catch {
        Write-Err "Docker is not running or unreachable. Please start Docker Desktop first."
        exit 1
    }

    # Clear conflicting host environment variables that might pollute docker-compose
    Remove-Item Env:CAR_CONTROL_BASE_URL -ErrorAction SilentlyContinue
    Remove-Item Env:CAR_MANUAL_BASE_URL -ErrorAction SilentlyContinue

    # Configure agent_list.json for internal Docker network
    $agentListPath = Join-Path $ORCH_DIR "config\agent_list.json"
    $agentListDocker = @{
        "car_control" = @{ url = "http://car_control:$PORT_CC"; enabled = $true }
        "car_manual"  = @{ url = "http://car_manual:$PORT_CM";   enabled = $true }
        "cloud"       = @{ url = "http://cloud:8005";           enabled = $false }
        "navigation"  = @{ url = "http://navigation:8003";      enabled = $false }
        "infotainment"= @{ url = "http://infotainment:8004";    enabled = $false }
    }
    $agentListDocker | ConvertTo-Json -Depth 3 | Set-Content -Path $agentListPath -Encoding UTF8
    Write-Ok "Configured agent_list.json for Docker internal network"

    # Services to start
    $servicesToStart = @("car_control", "car_manual", "orchestrator")
    if (-not $SkipUI) {
        $servicesToStart += "car_control_ui"
    }

    Write-Header "Starting Docker Compose services: $($servicesToStart -join ', ')..."
    docker compose @COMPOSE_X86_ARGS up -d $servicesToStart
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to start Docker Compose services."
        exit 1
    }

    function Wait-ForHealth {
        param([string]$Url, [string]$Name, [int]$MaxRetries = 40, [int]$DelaySec = 3)
        Write-Step "Waiting for $Name at $Url ..."
        for ($i = 0; $i -lt $MaxRetries; $i++) {
            try {
                $r = Invoke-RestMethod -Uri $Url -TimeoutSec 5 -ErrorAction SilentlyContinue
                if ($r.status -eq "ok" -or $null -ne $r) {
                    Write-Ok "$Name is ready"
                    return $true
                }
            } catch {}
            Start-Sleep -Seconds $DelaySec
        }
        Write-Warn "$Name not responding yet, continuing anyway..."
        return $false
    }

    Write-Header "Verifying service health..."
    [void](Wait-ForHealth "http://localhost:$PORT_CC/health" "car_control" 30 2)
    [void](Wait-ForHealth "http://localhost:$PORT_CM/health" "car_manual" 30 2)
    [void](Wait-ForHealth "http://localhost:$PORT_ORCH/health" "orchestrator" 30 2)
    if (-not $SkipUI) {
        [void](Wait-ForHealth "http://localhost:$PORT_UI/health" "car_control_ui" 20 2)
    }

    # Start voice processing on host (connecting to Docker orchestrator on port 8000)
    if (-not $SkipVoice) {
        $pyExe = Join-Path $VENV_PC "Scripts\python.exe"
        if (-not (Test-Path $pyExe)) {
            $pyExe = "python"
        }
        Start-VoiceHost -PythonExe $pyExe
    }

    # Open Browser
    if (-not $SkipUI -and -not $NoBrowser) {
        Start-Sleep -Seconds 1
        Write-Step "Opening Car Control UI: http://localhost:$PORT_UI ..."
        Start-Process "http://localhost:$PORT_UI"
    }

    Write-Header "Docker mode is RUNNING!"
    Write-Host "  * Car Control UI : http://localhost:$PORT_UI" -ForegroundColor Green
    Write-Host "  * Orchestrator   : http://localhost:$PORT_ORCH" -ForegroundColor Green
    Write-Host "  * Car Control    : http://localhost:$PORT_CC" -ForegroundColor Green
    Write-Host "  * Car Manual     : http://localhost:$PORT_CM" -ForegroundColor Green
    Write-Host "`nTo stop all services, run: .\run_all_pc.ps1 -Down" -ForegroundColor Yellow
    exit 0
}

# ==============================================================================
# MODE 2: HOST MODE (Default)
# ==============================================================================

# Ensure old conflicting Docker containers are stopped
Write-Step "Ensuring Docker containers are stopped to free ports..."
if (Test-Path $COMPOSE_X86) {
    docker compose @COMPOSE_X86_ARGS down 2>$null | Out-Null
}

# ── Detect Python ─────────────────────────────────────────────────────────────
Write-Header "Detecting Python on host..."
function Find-Python {
    $candidates = @("python", "python3", "python3.12", "python3.11", "python3.10")
    foreach ($cmd in $candidates) {
        try {
            $verOut = & $cmd --version 2>&1
            if ($verOut -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                    return $cmd, "$major.$minor"
                }
            }
        } catch {}
    }
    return $null, $null
}

$PY_CMD, $PY_VER = Find-Python
if (-not $PY_CMD) {
    Write-Err "Python >= 3.10 is required but not found in PATH."
    exit 1
}
Write-Ok "Found Python: $PY_CMD ($PY_VER)"

# ── Setup Unified PC Virtual Environment ──────────────────────────────────────
Write-Header "Setting up unified PC virtual environment..."

if (-not (Test-Path $VENV_BASE)) {
    New-Item -ItemType Directory -Path $VENV_BASE -Force | Out-Null
}

if ($Reset -and (Test-Path $VENV_PC)) {
    Write-Step "Resetting virtual environment ($VENV_PC)..."
    Remove-Item -Recurse -Force $VENV_PC -ErrorAction SilentlyContinue
}

$pyExe = Join-Path $VENV_PC "Scripts\python.exe"
if (-not (Test-Path $pyExe)) {
    Write-Step "Creating unified PC venv at $VENV_PC..."
    if (Test-Path $VENV_PC) {
        Remove-Item -Recurse -Force $VENV_PC -ErrorAction SilentlyContinue
    }
    & $PY_CMD -m venv $VENV_PC
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $pyExe)) {
        Write-Err "Failed to create virtual environment."
        exit 1
    }
    Write-Ok "Virtual environment created."
} else {
    Write-Ok "Virtual environment already exists ($VENV_PC)"
}

# ── Install All Packages at Once (Single pip command) ─────────────────────────
if (-not $SkipInstall) {
    Write-Header "Installing/verifying packages in one single command..."
    $ALL_PACKAGES = @(
        "a2a-sdk[http-server]==0.3.24",
        "fastapi[standard]",
        "uvicorn[standard]",
        "starlette",
        "httpx",
        "openai",
        "python-dotenv",
        "numpy",
        "pyyaml",
        "tqdm",
        "rank_bm25",
        "requests",
        "websocket-client",
        "jinja2",
        "sounddevice",
        "soundfile",
        "onnxruntime",
        "pydub",
        "kokoro-onnx",
        # Japanese TTS G2P for Kokoro's ja voice; without these it falls back to
        # espeak with the wrong phoneme set. pyopenjtalk-plus ships prebuilt
        # wheels (the plain pyopenjtalk needs a CMake build on Windows).
        "misaki",
        "fugashi",
        "jaconv",
        "mojimoji",
        "pyopenjtalk-plus",
        "unidic-lite",
        "openwakeword",
        "sherpa-onnx",
        "webrtcvad-wheels"
    )

    Write-Step "Running pip install for $($ALL_PACKAGES.Length) packages together..."
    & $pyExe -m pip install --disable-pip-version-check --quiet @ALL_PACKAGES
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "pip install reported warnings/errors, but proceeding..."
    } else {
        Write-Ok "All dependencies are ready"
    }
} else {
    Write-Ok "Skipped package installation (-SkipInstall)"
}

# ── Configure agent_list.json for Localhost ───────────────────────────────────
Write-Header "Configuring agent registry for localhost..."
$agentListPath = Join-Path $ORCH_DIR "config\agent_list.json"
$agentList = @{
    "car_control" = @{ url = "http://localhost:$PORT_CC"; enabled = $true }
    "car_manual"  = @{ url = "http://localhost:$PORT_CM"; enabled = $true }
    "cloud"       = @{ url = "http://localhost:8005";     enabled = $false }
    "navigation"  = @{ url = "http://localhost:8003";     enabled = $false }
    "infotainment"= @{ url = "http://localhost:8004";     enabled = $false }
}
$agentList | ConvertTo-Json -Depth 3 | Set-Content -Path $agentListPath -Encoding UTF8
Write-Ok "agent_list.json updated (localhost:$PORT_CC, localhost:$PORT_CM)"

# ── Service Launcher Helper ───────────────────────────────────────────────────
$procs = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

function Start-ServiceWindow {
    param(
        [string]$Name,
        [string]$WorkDir,
        [string]$ScriptWithArgs,
        [hashtable]$ExtraEnv = @{}
    )

    $launcherFile = Join-Path $VENV_BASE "${Name}_launcher.ps1"
    $lines = @()
    $lines += "`$host.UI.RawUI.WindowTitle = '[$Name]'"
    $lines += ""
    foreach ($k in [System.Environment]::GetEnvironmentVariables("Process").Keys) {
        $v = [System.Environment]::GetEnvironmentVariable($k, "Process")
        if ($null -ne $v) {
            $vEsc = $v -replace "'", "''"
            $lines += "[System.Environment]::SetEnvironmentVariable('$k', '$vEsc', 'Process')"
        }
    }
    $lines += ""
    $lines += "# Service-specific environment"
    $lines += "`$env:PYTHONPATH = '$STUBS_DIR;$WorkDir'"
    foreach ($k in $ExtraEnv.Keys) {
        $vEsc = $ExtraEnv[$k] -replace "'", "''"
        $lines += "`$env:$k = '$vEsc'"
    }
    $lines += ""
    $lines += "Set-Location '$WorkDir'"
    $lines += "Write-Host '[$(Get-Date -Format HH:mm:ss)] Starting $Name...' -ForegroundColor Cyan"
    $lines += "& '$pyExe' $ScriptWithArgs"
    $lines += "Write-Host '[$(Get-Date -Format HH:mm:ss)] $Name exited. Press Enter to close.' -ForegroundColor Yellow"
    $lines += "Read-Host"

    $lines | Out-File -FilePath $launcherFile -Encoding UTF8

    $p = Start-Process -FilePath "pwsh.exe" `
        -ArgumentList "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $launcherFile `
        -WorkingDirectory $WorkDir `
        -PassThru
    $procs.Add($p)
    Write-Ok "Started [$Name] (PID=$($p.Id))"
    return $p
}

function Wait-ForHealthHost {
    param([string]$Url, [string]$Name, [int]$MaxRetries = 30, [int]$DelaySec = 2)
    Write-Step "Waiting for $Name at $Url ..."
    for ($i = 0; $i -lt $MaxRetries; $i++) {
        try {
            $r = Invoke-RestMethod -Uri $Url -TimeoutSec 4 -ErrorAction SilentlyContinue
            if ($r.status -eq "ok" -or $null -ne $r) {
                Write-Ok "$Name is healthy"
                return $true
            }
        } catch {}
        Start-Sleep -Seconds $DelaySec
    }
    Write-Warn "$Name not responding yet, continuing..."
    return $false
}

# ── 1. Start car_control (Stub Car) ───────────────────────────────────────────
Write-Header "Starting car_control (port $PORT_CC)..."
$ccDir = Join-Path $AGENTS_DIR "car_control"
Start-ServiceWindow -Name "car_control" `
    -WorkDir $ccDir `
    -ScriptWithArgs "main_pc.py" `
    -ExtraEnv @{
        CAR_CONTROL_PORT             = "$PORT_CC"
        CAR_CONTROL_BASE_URL         = "http://localhost:$PORT_CC"
        CAR_SIMULATOR_CONNECTED      = "true"
        CAR_CONTROL_ENABLE_STATE_API = "1"
        GRAPHQL_HOST                 = ""
    }

[void](Wait-ForHealthHost "http://localhost:$PORT_CC/health" "car_control" 20 2)

# ── 2. Start car_manual ───────────────────────────────────────────────────────
Write-Header "Starting car_manual (port $PORT_CM)..."
$cmDir = Join-Path $AGENTS_DIR "car_manual"
Start-ServiceWindow -Name "car_manual" `
    -WorkDir $cmDir `
    -ScriptWithArgs "main.py" `
    -ExtraEnv @{
        CAR_MANUAL_PORT            = "$PORT_CM"
        CAR_MANUAL_BASE_URL        = "http://localhost:$PORT_CM"
        CAR_MANUAL_BRAND           = "mmc"
        CAR_MANUAL_SEARCH_MODE     = "bm25"
        CAR_MANUAL_SCORE_THRESHOLD = "0.2"
        HF_HOME                    = (Join-Path $cmDir "cache")
    }

[void](Wait-ForHealthHost "http://localhost:$PORT_CM/health" "car_manual" 25 2)

# ── 3. Start orchestrator ─────────────────────────────────────────────────────
Write-Header "Starting orchestrator (port $PORT_ORCH)..."
Start-ServiceWindow -Name "orchestrator" `
    -WorkDir $ORCH_DIR `
    -ScriptWithArgs "main.py" `
    -ExtraEnv @{
        ORCHESTRATOR_PORT  = "$PORT_ORCH"
        HF_HOME            = (Join-Path $ORCH_DIR "cache")
        LOCAL_LLM_URL      = "http://localhost:8081/v1"
        LOCAL_LLM_MODEL    = if ($env:LOCAL_LLM_MODEL) { $env:LOCAL_LLM_MODEL } else { "google/gemma-4-e2b" }
        GRAPHQL_HOST       = ""
    }

[void](Wait-ForHealthHost "http://localhost:$PORT_ORCH/health" "orchestrator" 25 2)

# ── 4. Start voice processing (Real Mic) or voice stub ────────────────────────
if (-not $SkipVoice) {
    Start-VoiceHost -PythonExe $pyExe
}

# ── 5. Start car_control_ui ───────────────────────────────────────────────────
if (-not $SkipUI) {
    Write-Header "Starting car_control_ui (port $PORT_UI)..."
    Start-ServiceWindow -Name "car_control_ui" `
        -WorkDir $UI_DIR `
        -ScriptWithArgs "-m uvicorn app:app --host 0.0.0.0 --port $PORT_UI" `
        -ExtraEnv @{
            CAR_CONTROL_URL      = "http://localhost:$PORT_CC"
            ORCHESTRATOR_URL     = "http://localhost:$PORT_ORCH"
            CAR_CONTROL_UI_PORT  = "$PORT_UI"
        }

    [void](Wait-ForHealthHost "http://localhost:$PORT_UI/health" "car_control_ui" 20 2)
}

# ── 6. Open Browser ───────────────────────────────────────────────────────────
if (-not $SkipUI -and -not $NoBrowser) {
    Start-Sleep -Seconds 1
    Write-Step "Opening Car Control UI: http://localhost:$PORT_UI ..."
    Start-Process "http://localhost:$PORT_UI"
}

Write-Header "All services are RUNNING!"
Write-Host "  * Car Control UI : http://localhost:$PORT_UI" -ForegroundColor Green
Write-Host "  * Orchestrator   : http://localhost:$PORT_ORCH" -ForegroundColor Green
Write-Host "  * Car Control    : http://localhost:$PORT_CC" -ForegroundColor Green
Write-Host "  * Car Manual     : http://localhost:$PORT_CM" -ForegroundColor Green
if (-not $SkipVoice) {
    if ($VoiceStub) {
        Write-Host "  * Voice Input    : Terminal window [voice_stub]" -ForegroundColor Green
    } else {
        Write-Host "  * Voice Input    : Real Microphone [voice_processing]" -ForegroundColor Green
    }
}
Write-Host "`nTo stop all services, run: .\run_all_pc.ps1 -Down" -ForegroundColor Yellow

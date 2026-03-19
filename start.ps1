# Containerised and application launch script to run from Windows OS

# Write-Host "Building and starting containers..."
# docker-compose up --build -d

Write-Host "Waiting for backend to be ready..."
$timeout = 360
$elapsed = 0
while ($elapsed -lt $timeout) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "Backend is ready!"
            break
        }
    } catch {
        Write-Host "  Backend not ready yet, retrying in 3 seconds..."
        Start-Sleep -Seconds 3
        $elapsed += 3
    }
}

Write-Host "Waiting for frontend to be ready..."
$elapsed = 0
while ($elapsed -lt $timeout) {
    try {
        # changing "http://localhost:8501" to "http://localhost" is for nginx
        $response = Invoke-WebRequest -Uri "http://localhost" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "Frontend is ready!"
            break
        }
    } catch {
        Write-Host "  Frontend not ready yet, retrying in 3 seconds..."
        Start-Sleep -Seconds 3
        $elapsed += 3
    }
}

Write-Host "Checking FastAPI endpoints..."
$endpoints = @(
    @{ Name = "Health Check";   Url = "http://localhost:8000/health" },
    @{ Name = "API Docs";       Url = "http://localhost:8000/docs" },
    @{ Name = "Redoc";          Url = "http://localhost:8000/redoc" },
    @{ Name = "OpenAPI Schema"; Url = "http://localhost:8000/openapi.json" }
)

foreach ($ep in $endpoints) {
    try {
        $r = Invoke-WebRequest -Uri $ep.Url -UseBasicParsing -TimeoutSec 5
        Write-Host "  [OK]   $($ep.Name) -> $($ep.Url) (HTTP $($r.StatusCode))"
    } catch {
        Write-Host "  [FAIL] $($ep.Name) -> $($ep.Url)"
    }
}

Write-Host "Opening browser tabs..."
# Docker setting:
#Start-Process "http://localhost:8501"
# Nginx setting
Start-Process "http://localhost"
Start-Process "http://localhost:8000/docs"
Start-Process "http://localhost:8000/redoc"
Start-Process "http://localhost:5000"
Start-Process "http://localhost:6006"

Write-Host ""
Write-Host "================================================"
Write-Host "  All services are running!"
Write-Host ""
Write-Host "  FRONTEND"
# Docker setting
#Write-Host "    Streamlit       : http://localhost:8501"
# Nginx setting: Streamlit via nging port 80
Write-Host "    Streamlit       : http://localhost"
Write-Host ""
Write-Host "  BACKEND"
Write-Host "    Health Check   : http://localhost:8000/health"
Write-Host "    API Docs       : http://localhost:8000/docs"
Write-Host "    Redoc          : http://localhost:8000/redoc"
Write-Host "    OpenAPI Schema : http://localhost:8000/openapi.json"
Write-Host ""
Write-Host "  MONITORING"
# Docker setting
# Write-Host "    MLflow         : http://localhost:5000"
# Nginx setting
Write-Host "    MLflow         : http://localhost:5000 (user: admin)"
Write-Host "    TensorBoard    : http://localhost:6006"
Write-Host ""
Write-Host "  To stop: docker-compose down"
Write-Host "================================================"
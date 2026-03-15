# ═══════════════════════════════════════════════════════════════
# deploy.ps1 — One command deploys everything to Google Cloud
# Usage: .\deploy.ps1
# ═══════════════════════════════════════════════════════════════

$PROJECT_ID  = "ai-sales-analyst"
$REGION      = "asia-south1"
$SERVICE     = "sales-api"
$IMAGE       = "gcr.io/$PROJECT_ID/$SERVICE"

Write-Host ""
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  AI Sales Analyst — Cloud Deploy" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 1 - Build and push Docker image
Write-Host "[ 1/4 ] Building Docker image..." -ForegroundColor Yellow
gcloud builds submit --tag $IMAGE --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }
Write-Host "  ✅ Image built" -ForegroundColor Green

# Step 2 - Read Gemini key from .env
Write-Host "[ 2/4 ] Reading API key from .env..." -ForegroundColor Yellow
$GEMINI_KEY = ""
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^GEMINI_API_KEY=(.+)$") {
            $GEMINI_KEY = $matches[1].Trim()
        }
    }
}
if ($GEMINI_KEY -eq "" -or $GEMINI_KEY -eq "paste_your_gemini_key_here") {
    Write-Host "  ❌ No Gemini API key found in .env!" -ForegroundColor Red
    Write-Host "     Open .env and set: GEMINI_API_KEY=your_key" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ API key loaded" -ForegroundColor Green

# Step 3 - Deploy to Cloud Run
Write-Host "[ 3/4 ] Deploying to Cloud Run ($REGION)..." -ForegroundColor Yellow
$DEPLOY_OUTPUT = gcloud run deploy $SERVICE `
    --image $IMAGE `
    --region $REGION `
    --platform managed `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --timeout 120 `
    --set-env-vars "GEMINI_API_KEY=$GEMINI_KEY" `
    --project $PROJECT_ID `
    --format "value(status.url)" 2>&1

$SERVICE_URL = gcloud run services describe $SERVICE `
    --region $REGION `
    --project $PROJECT_ID `
    --format "value(status.url)"

if ($LASTEXITCODE -ne 0) { Write-Host "Deploy failed!" -ForegroundColor Red; exit 1 }
Write-Host "  ✅ Backend deployed: $SERVICE_URL" -ForegroundColor Green

# Step 4 - Update frontend with Cloud Run URL and deploy to Firebase
Write-Host "[ 4/4 ] Updating frontend and deploying to Firebase..." -ForegroundColor Yellow
$HTML = Get-Content "frontend\index.html" -Raw
$HTML = $HTML -replace "const API='[^']*'", "const API='$SERVICE_URL'"
Set-Content "frontend\index.html" $HTML -NoNewline
Write-Host "  ✅ Frontend updated with: $SERVICE_URL" -ForegroundColor Green

firebase deploy --only hosting --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Host "Firebase deploy failed!" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "═══════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend:  $SERVICE_URL" -ForegroundColor Cyan
Write-Host "  Frontend: https://$PROJECT_ID.web.app" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Open your app:" -ForegroundColor White
Write-Host "  https://$PROJECT_ID.web.app" -ForegroundColor Yellow
Write-Host ""

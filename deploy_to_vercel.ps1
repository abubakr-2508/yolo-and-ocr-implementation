# PowerShell script to deploy to Vercel
Write-Host "Deploying to Vercel..." -ForegroundColor Green
Set-Location "c:\Users\HP OMEN\OneDrive\Desktop\yolo"

# Check if Vercel CLI is installed
try {
    $vercelVersion = vercel --version
    Write-Host "Vercel CLI is installed: $vercelVersion" -ForegroundColor Green
} catch {
    Write-Host "Vercel CLI is not installed. Installing now..." -ForegroundColor Yellow
    npm install -g vercel
}

# Deploy to Vercel
Write-Host "Starting deployment..." -ForegroundColor Yellow
vercel --prod

Write-Host "Deployment completed!" -ForegroundColor Green
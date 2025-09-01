# PowerShell script to initialize Git repository
Write-Host "Setting up Git repository..." -ForegroundColor Green
Set-Location "c:\Users\HP OMEN\OneDrive\Desktop\yolo"
git init
git add .
git commit -m "Initial commit: YOLO Object Detection with OCR"
Write-Host "Git repository initialized and files committed." -ForegroundColor Green
Write-Host ""
Write-Host "To push to GitHub, you'll need to:" -ForegroundColor Yellow
Write-Host "1. Create a repository on GitHub" -ForegroundColor Yellow
Write-Host "2. Add the remote origin with: git remote add origin <repository-url>" -ForegroundColor Yellow
Write-Host "3. Push with: git push -u origin main" -ForegroundColor Yellow
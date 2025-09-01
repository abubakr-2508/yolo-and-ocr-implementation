# PowerShell script to push to your GitHub repository
Write-Host "Setting up remote origin and pushing to GitHub..." -ForegroundColor Green
Set-Location "c:\Users\HP OMEN\OneDrive\Desktop\yolo"

# Add the remote origin
git remote add origin https://github.com/abubakr-2508/yolo-and-ocr-implementation.git

# Push to GitHub
git branch -M main
git push -u origin main

Write-Host "Repository pushed to GitHub successfully!" -ForegroundColor Green
Write-Host "Your project is now available at: https://github.com/abubakr-2508/yolo-and-ocr-implementation" -ForegroundColor Yellow
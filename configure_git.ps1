# PowerShell script to configure Git and complete setup
Write-Host "Configuring Git user identity..." -ForegroundColor Green
Set-Location "c:\Users\HP OMEN\OneDrive\Desktop\yolo"

# Set user identity (you should replace these with your actual GitHub email and name)
git config user.email "your_email@example.com"
git config user.name "Your Name"

# Add all files and commit
git add .
git commit -m "Initial commit: YOLO Object Detection with OCR"

Write-Host "Git repository configured and files committed." -ForegroundColor Green
Write-Host ""
Write-Host "To push to GitHub, follow these steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub (https://github.com/new)" -ForegroundColor Yellow
Write-Host "2. Add the remote origin with: git remote add origin <repository-url>" -ForegroundColor Yellow
Write-Host "3. Push with: git push -u origin main" -ForegroundColor Yellow
@echo off
echo Setting up Git repository...
cd /d "c:\Users\HP OMEN\OneDrive\Desktop\yolo"
git init
git add .
git commit -m "Initial commit: YOLO Object Detection with OCR"
echo Git repository initialized and files committed.
echo.
echo To push to GitHub, you'll need to:
echo 1. Create a repository on GitHub
echo 2. Add the remote origin with: git remote add origin ^<repository-url^>
echo 3. Push with: git push -u origin main
pause
@echo off
echo YOLOv5 Project Management Integration System - GitHub Upload
echo =========================================================
echo.

echo IMPORTANT: Before proceeding, make sure you have:
echo 1. A GitHub account
echo 2. A Personal Access Token (PAT) for GitHub
echo.
echo If you don't have a Personal Access Token, please read the GITHUB_UPLOAD_GUIDE.md file
echo for instructions on how to create one.
echo.

set /p CONTINUE=Do you want to continue with the upload? (Y/N):
if /i "%CONTINUE%" neq "Y" goto :end

echo.
echo Initializing Git repository...
git init

echo.
echo Adding files to Git...
git add .

echo.
echo Committing files...
git commit -m "Initial commit of YOLOv5 Project Management Integration System"

echo.
echo Adding GitHub remote...
git remote add origin https://github.com/ai-in-pm/YOLOv5-PMIS.git

echo.
echo Pushing to GitHub...
echo When prompted for credentials:
echo - For username, enter your GitHub username
echo - For password, enter your Personal Access Token (not your GitHub password)
echo.
echo Press any key to continue...
pause > nul

git push -u origin master

echo.
if %ERRORLEVEL% EQU 0 (
    echo Upload successful! Check your GitHub repository at:
    echo https://github.com/ai-in-pm/YOLOv5-PMIS
) else (
    echo Upload failed. Please check the error message above.
    echo For troubleshooting, refer to the GITHUB_UPLOAD_GUIDE.md file.
)

:end
echo.
echo Press any key to exit...
pause > nul

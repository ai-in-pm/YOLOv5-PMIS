# GitHub Upload Guide

This guide will help you upload the YOLOv5 Project Management Integration System to your GitHub repository.

## Prerequisites

1. Git installed on your computer
2. GitHub account
3. Personal Access Token (PAT) for GitHub

## Creating a Personal Access Token (PAT)

1. Go to GitHub.com and log in to your account
2. Click on your profile picture in the top-right corner and select "Settings"
3. Scroll down and click on "Developer settings" in the left sidebar
4. Click on "Personal access tokens" and then "Tokens (classic)"
5. Click "Generate new token" and then "Generate new token (classic)"
6. Give your token a descriptive name (e.g., "YOLOv5-PMIS Upload")
7. Select the following scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (if you plan to use GitHub Actions)
8. Click "Generate token"
9. **IMPORTANT**: Copy the token immediately and save it somewhere secure. You won't be able to see it again!

## Uploading to GitHub

### Option 1: Using the Command Line

1. Open a command prompt or PowerShell window
2. Navigate to the yolo_projects directory:
   ```
   cd C:\CC-WorkingDir\yolov5-master\yolo_projects
   ```
3. Initialize a Git repository:
   ```
   git init
   ```
4. Add all files to Git:
   ```
   git add .
   ```
5. Commit the files:
   ```
   git commit -m "Initial commit of YOLOv5 Project Management Integration System"
   ```
6. Add the GitHub remote:
   ```
   git remote add origin https://github.com/ai-in-pm/YOLOv5-PMIS.git
   ```
7. Push to GitHub (you'll be prompted for your username and password/token):
   ```
   git push -u origin master
   ```
   - For username, enter your GitHub username
   - For password, enter your Personal Access Token (not your GitHub password)

### Option 2: Using the Batch File

1. Run the `upload_to_github.bat` file
2. When prompted for credentials:
   - For username, enter your GitHub username
   - For password, enter your Personal Access Token (not your GitHub password)

## Troubleshooting

If you encounter any issues:

1. Make sure Git is installed and configured correctly
2. Verify that your Personal Access Token has the correct permissions
3. Check that the repository exists on GitHub
4. Ensure you're using the correct username and token

## After Uploading

Once the upload is complete, you can verify that your files are on GitHub by visiting:
https://github.com/ai-in-pm/YOLOv5-PMIS

# Hailo Chat Launcher — run this to start the chat app fresh
# Re-initializes the device then starts Streamlit (do not kill Streamlit while using it)

Write-Host "Initializing UGen300 device..." -ForegroundColor Cyan
Start-Process msiexec -ArgumentList "/i","C:\Users\Test\Downloads\HailoRT_5.3.2_windows_installer.msi","/quiet","/norestart" -Verb RunAs -Wait
Write-Host "Waiting for device to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

Write-Host "Starting Hailo Chat..." -ForegroundColor Green
& "C:\Users\Test\hailo_ai\hailo_venv\Scripts\streamlit.exe" run "C:\Users\Test\hailo_ai\hailo_chat.py"

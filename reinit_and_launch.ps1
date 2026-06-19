Write-Host "Step 1: Rebinding USB driver..." -ForegroundColor Cyan
pnputil.exe /add-driver "C:\Program Files\HailoRT\driver\hailo10h_usb\hailokm_usb.inf" /install
Start-Sleep -Seconds 3

Write-Host "Step 2: Loading firmware..." -ForegroundColor Cyan
& "C:\Program Files\HailoRT\bin\hailo_usb_loader.exe" fw-update 2>&1
Start-Sleep -Seconds 8

Write-Host "Step 3: Starting chat app..." -ForegroundColor Green
Start-Process -FilePath "C:\Users\Test\hailo_ai\hailo_venv\Scripts\streamlit.exe" -ArgumentList "run","C:\Users\Test\hailo_ai\hailo_chat.py","--server.headless","false"
Write-Host "Chat app launched. Open http://localhost:8501" -ForegroundColor Green

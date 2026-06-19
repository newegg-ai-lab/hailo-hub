Stop-Service hailo_usb_service -ErrorAction SilentlyContinue
Set-Service hailo_usb_service -StartupType Disabled -ErrorAction SilentlyContinue
$result = & 'C:\Users\Test\hailo_ai\hailo_venv\Scripts\python.exe' -c "from hailo_platform import VDevice; v = VDevice(); print('VDevice OK'); v.release()" 2>&1
$result | Out-File "C:\Users\Test\hailo_ai\vdev_test_result.txt" -Force

# V.A.T.S. — Environment Setup Guide

Step-by-step instructions to replicate this environment on a fresh machine.
Tested on Windows 11 + WSL2 Ubuntu.

---

## 1. WSL2 Ubuntu

If WSL isn't installed yet:

```powershell
# PowerShell (Admin)
wsl --install -d Ubuntu
```

Reboot if prompted. Then open Ubuntu from the Start menu and set up your
username/password.

---

## 2. Python + GPU in WSL

WSL2 automatically passes through the NVIDIA GPU. Verify:

```bash
# Inside WSL
nvidia-smi
```

You should see your GPU listed. If not, make sure you have the latest
NVIDIA Game Ready or Studio driver installed on **Windows** (not inside WSL).

Install Python and venv if not present:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## 3. Project Setup

```bash
# Clone the repo (or copy the folder)
cd ~/repos
git clone <your-repo-url> VATS
cd VATS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install ultralytics opencv-python-headless numpy
```

Verify GPU is visible to PyTorch:

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
```

---

## 4. USB Webcam Passthrough (usbipd)

WSL doesn't natively see USB devices. We use `usbipd-win` to attach the
webcam from Windows into WSL.

### 4a. Install usbipd on Windows

```powershell
# PowerShell (Admin)
winget install dorssel.usbipd-win --accept-source-agreements --accept-package-agreements
```

Close and reopen your terminal after installing so `usbipd` is on PATH.

### 4b. Install USB tools in WSL

```bash
# Inside WSL
sudo apt update
sudo apt install -y linux-tools-generic hwdata usbutils
```

You also need the usbip tool. On some Ubuntu versions the binary lands at
a kernel-version-specific path. Create a symlink if needed:

```bash
# Check if usbip exists
which usbip || sudo ln -s /usr/lib/linux-tools/*/usbip /usr/local/bin/usbip
```

### 4c. Attach the webcam

```powershell
# PowerShell (Admin) — list all USB devices
usbipd list
```

You'll see output like:

```
BUSID  VID:PID    DEVICE                          STATE
5-4    046d:08e5  HD Pro Webcam C920               Not shared
2-5    8087:0029  Intel Bluetooth                  Not shared
```

Find your webcam's BUSID (e.g., `5-4`), then:

```powershell
# Bind it (one-time, makes it available for sharing)
usbipd bind --busid 5-4

# Attach it to WSL
usbipd attach --wsl --busid 5-4
```

### 4d. Verify in WSL

```bash
# Inside WSL — check the camera appeared
ls /dev/video*
# Should show /dev/video0 (or similar)

# Quick test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'FAILED'); cap.release()"
```

### 4e. Re-attaching after reboot

The `bind` persists across reboots, but the `attach` does not. After each
reboot (or after unplugging/replugging the webcam):

```powershell
# PowerShell (Admin)
usbipd attach --wsl --busid 5-4
```

---

## 5. Display / GUI from WSL

To show OpenCV windows from WSL, you need an X server or WSLg.

**Windows 11 (22H2+) with WSLg** — works out of the box. OpenCV windows
just appear on your Windows desktop.

**Older Windows 11 or Windows 10** — install an X server:

```powershell
# Option: VcXsrv
winget install marha.VcXsrv
```

Then in WSL, set the display:

```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

To test:

```bash
sudo apt install -y x11-apps
xclock   # Should show a clock window on your desktop
```

---

## 6. Run V.A.T.S.

```bash
cd ~/repos/VATS
source .venv/bin/activate

# Test model against webcam
python3 test_model.py models/yolov8x_drone.pt

# Run full pipeline
python3 vats.py 0
```

---

## Troubleshooting

**"Cannot open webcam"**
- Is the webcam attached? Run `usbipd list` — state should say "Attached"
- Check `ls /dev/video*` in WSL
- Try camera index 1 or 2 instead of 0

**"CUDA not available"**
- Update your Windows NVIDIA driver (not the WSL one)
- Run `nvidia-smi` in WSL to verify GPU passthrough
- Make sure you installed PyTorch with CUDA support (pip should auto-detect)

**No GUI window appears**
- On Win11 22H2+: WSLg should handle this. Try `wsl --update`
- On older systems: install VcXsrv and set DISPLAY variable

**usbipd attach fails**
- Run PowerShell as Administrator
- Make sure the webcam isn't in use by another app (Zoom, Teams, etc.)
- Try `usbipd unbind` then `usbipd bind` again

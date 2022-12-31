# AutoPTZ - Automating Your Production

Our application is to be used mainly with PTZ cameras and have them physically move automatically by tracking a face on screen. While it can be connected to non-PTZ cameras, it does not add much value for individuals to use. The only difference would be that a PTZ camera can move but a normal camera will not move because of its nature. We have implemented facial recognition techniques to specifically select a face to track if there are multiple people on screen. Our application works on any PTZ cameras and is designed to be compatible with any standardized video connections via USB, NewTek NDI®, or RTSP IP. 



## Disclaimer

- ⚠️ The project is under active development.
- ⚠️ Expect bugs and inaccuracies.
- ⚠️ Do not use the app during your live productions yet!
## Features

- Sources for NewTek NDI® and USB are supported, IP (RTSP) is under development
- Live camera feeds
- Accurate Facial Recognition and Motion Tracking
- Automated PTZ VISCA Movement for Network and USB
- Cross platform


## Technology  Stack

1. **Python** - Backend of the application
2. **Pyside6 (Qt)** - Frontend of the application
3. **OpenCV** - Camera Video Feeds and Facial Detection
4. **Dlib** - Facial Recognition and Motion Tracking
5. **Facial Recognition** - Powered by Dlib, Provided by ageitgey (https://github.com/ageitgey/face_recognition)
6. **NewTek NDI Wrapper**  - Provided by buresu (https://github.com/buresu/ndi-python)
7. **IP VISCA Controller** - PTZ Movement Controller, Provided by misterhay (https://github.com/misterhay/VISCA-IP-Controller)


## Installation

### Requirements

- Python 3.7+
- Windows or macOS (Linux is not officially supported, but should work)

### Installation Options:
Clone the project
```bash
  git clone https://github.com/AutoPTZ/autoptz.git
```

Then instal cmake to build a copy a dlib for your system.
```bash
  pip install cmake
  pip install dlib
```

After you successfully install cmake and dlib, you can install the rest of the required libraries.
```bash
  pip install -r requirements.txt
```

Then you can finally run the program.
```bash
  python startup.py
```
    

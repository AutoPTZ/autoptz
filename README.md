# AutoPTZ Copy

Our application is to be used mainly with PTZ cameras and have them physically move automatically by tracking a face on screen. While it can be connected to non-PTZ cameras, it does not add much value for individuals to use. The only difference would be that a PTZ camera can move but a normal camera will not move because of its nature. We have implemented facial recognition techniques to specifically select a face to track if there are multiple people on screen. Our application works on any PTZ cameras and is designed to be compatible with any standardized video connections via USB, NewTek NDI®, or RTSP IP. 



# Installation

## Requirements

- Python 3.7+
- Windows or macOS (Linux is not officially supported, but should work)

## Installation Options:
Clone the project
```bash
  git clone https://github.com/AutoPTZ/autoptz-copy.git
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
    
# Features

- Sources for IP (RTSP), NewTek NDI®, and USB are supported
- Live camera feeds
- Accurate Facial Recognition and Motion Tracking
- Automated PTZ VISCA Movement for Network and USB
- Cross platform



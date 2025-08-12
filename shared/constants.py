import os

try:
    import cv2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
except Exception:  # pragma: no cover - optional dependency
    FONT = None

# Determine the project root based on this file's location rather than the
# current working directory.  This allows the code to be executed from any
# path without breaking the relative resource references.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINER_PATH = os.path.join(ROOT_DIR, "logic", "image_processing", "models")
ENCODINGS_PATH = os.path.join(TRAINER_PATH, "encodings.pickle")
CAFFEMODEL_PATH = os.path.join(TRAINER_PATH, "MobileNetSSD_deploy.caffemodel")
PROTOTXT_PATH = os.path.join(TRAINER_PATH, "MobileNetSSD_deploy.prototxt")
CAMERA_STYLESHEET = """
                    QLabel[active="false"]{
                        border: 2.5px solid slategray;
                        border-radius: 3px;}

                    QLabel::hover {
                        border: 2.5px solid crimson;
                        border-radius: 3px;}

                    QLabel[active="true"]{
                        border: 2.5px solid dodgerblue;
                        border-radius: 3px;}
                    """
CURRENT_ACTIVE_CAM_WIDGET = None
CURRENT_ACTIVE_PTZ_DEVICE = None
IN_USE_USB_PTZ_DEVICES = []
ASSIGNED_USB_PTZ_CAMERA_WIDGETS = []
RUNNING_HARDWARE_CAMERA_WIDGETS = []
ICON_PNG = os.path.join(ROOT_DIR, 'shared', 'AutoPTZLogo.png')
NDI_SOURCE_LIST = []

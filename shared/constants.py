import cv2
import os

ROOT_DIR = os.path.abspath(os.curdir)
TRAINER_PATH = ROOT_DIR + "/logic/image_processing/models/"
ENCODINGS_PATH = ROOT_DIR + '/logic/image_processing/models/encodings.pickle'
CAFFEMODEL_PATH = ROOT_DIR + \
    '/logic/image_processing/models/MobileNetSSD_deploy.caffemodel'
PROTOTXT_PATH = ROOT_DIR + \
    '/logic/image_processing/models/MobileNetSSD_deploy.prototxt'
FONT = cv2.FONT_HERSHEY_SIMPLEX
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
ICON_PNG = ROOT_DIR + '/shared/AutoPTZLogo.png'
NDI_SOURCE_LIST = []

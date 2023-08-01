import os
from logic.facial_recognition.dialogs.add_face import AddFaceDlg
from logic.facial_recognition.dialogs.remove_face import RemoveFaceDlg
from logic.facial_recognition.dialogs.reset_database import ResetDatabaseDlg
from shared.message_prompts import show_info_messagebox
import shared.constants as constants


class ShowDialog:
    """
    Simplified Add/Train/Remove/Reset Dialog methods into one place
    """

    def __init__(self):
        super(ShowDialog, self).__init__()

    @staticmethod
    def add_face():
        """Launch the Add Face dialog based on the currently selected camera."""
        if constants.CURRENT_ACTIVE_CAM_WIDGET is None:
            show_info_messagebox("Please add and select a camera.")
        else:
            print("Opening Face Dialog")
            dlg = AddFaceDlg(camera=constants.CURRENT_ACTIVE_CAM_WIDGET)
            dlg.exec()

    @staticmethod
    def remove_face():
        """Launch the Remove Face dialog based on the currently selected camera."""
        if not os.path.exists(constants.ENCODINGS_PATH):
            show_info_messagebox("No Faces to remove.")
        else:
            print("Opening Face Dialog")
            dlg = RemoveFaceDlg()
            dlg.exec()

    @staticmethod
    def reset_database():
        """Launch the Remove Face dialog based on the currently selected camera."""
        print("Opening Face Dialog")
        dlg = ResetDatabaseDlg()
        dlg.exec()

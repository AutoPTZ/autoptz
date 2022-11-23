import os
import shutil

from logic.facial_tracking.dialogs.add_face import AddFaceDlg
from logic.facial_tracking.dialogs.remove_face import RemoveFaceDlg
from logic.facial_tracking.dialogs.reset_database import ResetDatabaseDlg
from logic.facial_tracking.dialogs.train_face import TrainerDlg
from shared.message_prompts import show_info_messagebox
import shared.constants as constants


class ShowDialog:
    """
    Simplified Add/Train/Remove/Reset Dialog methods into one place
    """
    def __init__(self):
        super(ShowDialog, self).__init__()

    @staticmethod
    def add_face(update_face_selection):
        """Launch the Add Face dialog based on the currently selected camera."""
        if constants.CURRENT_ACTIVE_CAM_WIDGET is None:
            show_info_messagebox("Please add and select a camera.")
        else:
            print("Opening Face Dialog")
            dlg = AddFaceDlg(camera=constants.CURRENT_ACTIVE_CAM_WIDGET)
            dlg.closeEvent = update_face_selection
            dlg.show()

    @staticmethod
    def remove_face(update_face_selection):
        """Launch the Remove Face dialog based on the currently selected camera."""
        if not os.path.isdir(constants.IMAGE_PATH) or not os.listdir(constants.IMAGE_PATH):
            show_info_messagebox("No Faces to remove.")
        else:
            current_len = len(os.listdir(constants.IMAGE_PATH))
            print("Opening Face Dialog")
            dlg = RemoveFaceDlg()
            dlg.closeEvent = update_face_selection
            dlg.show()
            if not os.listdir(constants.IMAGE_PATH):
                if os.path.exists(constants.IMAGE_PATH):
                    shutil.rmtree(constants.IMAGE_PATH)
                if os.path.exists(constants.ENCODINGS_PATH):
                    os.remove(constants.ENCODINGS_PATH)
            elif current_len is not len(os.listdir(constants.IMAGE_PATH)):
                dlg = TrainerDlg()
                dlg.show()

    @staticmethod
    def retrain_face():
        """ Launch the Retrain Model dialog. """
        if not os.path.isdir(constants.IMAGE_PATH) or not os.listdir(constants.IMAGE_PATH):
            show_info_messagebox("No Faces to train.")
        else:
            dlg = TrainerDlg()
            dlg.show()

    @staticmethod
    def reset_database(update_face_selection):
        """Launch the Remove Face dialog based on the currently selected camera."""
        print("Opening Face Dialog")
        dlg = ResetDatabaseDlg()
        dlg.closeEvent = update_face_selection
        dlg.exec()

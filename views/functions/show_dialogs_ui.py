from logic.facial_tracking.dialogs.add_face import AddFaceDlg
from shared.message_prompts import show_info_messagebox
import shared.constants as constants


class ShowDialog:

    def __init__(self):
        super(ShowDialog, self).__init__()

    def add_face(self):
        """Launch the Add Face dialog based on the currently selected camera."""
        if constants.CURRENT_ACTIVE_CAM_WIDGET is None:
            show_info_messagebox("Please add and select a camera.")
        else:
            print("Opening Face Dialog")
            dlg = AddFaceDlg(camera=constants.CURRENT_ACTIVE_CAM_WIDGET)
            dlg.exec()

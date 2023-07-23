from PySide6.QtCore import QThread, Signal
from views.widgets.camera_widget import CameraWidget


class CameraSetupThread(QThread):
    setup_finished = Signal(CameraWidget)

    def __init__(self, source, width, height, isNDI, lock):
        super().__init__()
        print(f'init camera setup thread for Source {source}')
        self.source = source
        self.width = width
        self.height = height
        self.isNDI = isNDI
        self.lock = lock

    def run(self):
        print(f'run camera setup thread for Source {self.source}')
        camera_widget = CameraWidget(source=self.source, width=self.width, height=self.height,
                                     isNDI=self.isNDI, lock=self.lock)
        print(f'created camera widget for Source {self.source}')
        self.setup_finished.emit(camera_widget)

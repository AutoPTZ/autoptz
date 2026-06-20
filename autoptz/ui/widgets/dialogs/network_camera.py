"""NetworkCameraDialog — add an RTSP/ONVIF camera (single native title).

One title (the window title), a protocol toggle, connection fields, a live URI
preview, and validation.  On accept it calls ``EngineClient.addCamera(uri, name)``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

_DEFAULT_PORT = {"RTSP": 554, "ONVIF": 80}


class NetworkCameraDialog(QDialog):
    """Build an RTSP/ONVIF stream URI and add it as a camera."""

    def __init__(self, client: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("Add Network Camera")
        self.setModal(True)
        self.setMinimumWidth(420)

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(10)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)

        self._protocol = QComboBox()
        self._protocol.addItems(["RTSP", "ONVIF"])
        self._protocol.currentTextChanged.connect(self._on_protocol)
        form.addRow("Protocol", self._protocol)

        self._ip = QLineEdit()
        self._ip.setPlaceholderText("192.168.1.50")
        form.addRow("IP address", self._ip)
        self._port = QLineEdit(str(_DEFAULT_PORT["RTSP"]))
        form.addRow("Port", self._port)
        self._path = QLineEdit()
        self._path.setPlaceholderText("/stream1")
        form.addRow("Stream path", self._path)
        self._user = QLineEdit()
        form.addRow("Username", self._user)
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password", self._password)
        self._name = QLineEdit()
        self._name.setPlaceholderText("optional")
        form.addRow("Display name", self._name)
        col.addLayout(form)

        self._preview = QLabel("—")
        self._preview.setStyleSheet(
            f"color: {T.CURRENT.subtext}; font-family: Menlo; padding: 6px;"
            f" border: 1px solid {T.CURRENT.border}; border-radius: 6px;"
        )
        self._preview.setWordWrap(True)
        col.addWidget(self._preview)

        self._error = QLabel("")
        self._error.setStyleSheet(f"color: {T.ERROR};")
        col.addWidget(self._error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

        for w in (self._ip, self._port, self._path, self._user, self._password):
            w.textChanged.connect(self._update_preview)
        self._update_preview()

    def _on_protocol(self, proto: str) -> None:
        self._port.setText(str(_DEFAULT_PORT.get(proto, 554)))
        is_rtsp = proto == "RTSP"
        self._path.setEnabled(is_rtsp)
        self._update_preview()

    def _build_uri(self) -> str:
        proto = self._protocol.currentText().lower()
        ip = self._ip.text().strip()
        port = self._port.text().strip()
        user = self._user.text().strip()
        pw = self._password.text()
        auth = f"{user}:{pw}@" if user else ""
        scheme = "rtsp" if proto == "rtsp" else "onvif"
        path = self._path.text().strip() if scheme == "rtsp" else ""
        if path and not path.startswith("/"):
            path = "/" + path
        portpart = f":{port}" if port else ""
        return f"{scheme}://{auth}{ip}{portpart}{path}"

    def _update_preview(self) -> None:
        # show credentials masked in the preview
        proto = self._protocol.currentText().lower()
        ip = self._ip.text().strip() or "<ip>"
        port = self._port.text().strip()
        user = self._user.text().strip()
        scheme = "rtsp" if proto == "rtsp" else "onvif"
        path = self._path.text().strip() if scheme == "rtsp" else ""
        if path and not path.startswith("/"):
            path = "/" + path
        auth = f"{user}:••••@" if user else ""
        self._preview.setText(f"{scheme}://{auth}{ip}{(':' + port) if port else ''}{path}")

    def _accept(self) -> None:
        ip = self._ip.text().strip()
        if not ip or " " in ip:
            self._error.setText("Enter a valid IP/host (no spaces).")
            return
        port = self._port.text().strip()
        if port and (not port.isdigit() or not (1 <= int(port) <= 65535)):
            self._error.setText("Port must be 1–65535.")
            return
        uri = self._build_uri()
        try:
            self._client.addCamera(uri, self._name.text().strip())
        except Exception:  # noqa: BLE001
            self._error.setText("Could not add the camera.")
            return
        self.accept()

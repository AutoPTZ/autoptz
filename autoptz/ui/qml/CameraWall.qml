import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    title: "AutoPTZ v2"
    visible: true
    minimumWidth: 800
    minimumHeight: 560
    width: 1280
    height: 720

    // ── palette ───────────────────────────────────────────────────────────────
    readonly property color clrBg:        "#0d0d1a"
    readonly property color clrSurface:   "#13132b"
    readonly property color clrBorder:    "#252545"
    readonly property color clrAccent:    "#e94560"
    readonly property color clrText:      "#e8e8f0"
    readonly property color clrSubtext:   "#7070a0"
    readonly property color clrTracking:  "#00cc66"
    readonly property color clrWarning:   "#ffcc00"

    background: Rectangle { color: root.clrBg }

    // ── top bar ───────────────────────────────────────────────────────────────
    header: Rectangle {
        height: 48
        color: root.clrSurface
        border.color: root.clrBorder
        border.width: 1

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12

            Text {
                text: "AutoPTZ"
                font.pixelSize: 18
                font.bold: true
                color: root.clrAccent
            }

            Item { Layout.fillWidth: true }

            // Active camera count
            Text {
                text: engineClient.cameraModel.rowCount > 0
                      ? engineClient.cameraModel.rowCount + " camera"
                        + (engineClient.cameraModel.rowCount > 1 ? "s" : "")
                      : ""
                color: root.clrSubtext
                font.pixelSize: 13
            }

            // Add Camera button
            Button {
                text: "+ Add Camera"
                highlighted: true
                onClicked: addDialog.open()

                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font.pixelSize: 13
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: parent.pressed ? "#c73750" : (parent.hovered ? "#f05070" : root.clrAccent)
                    radius: 6
                }
            }
        }
    }

    // ── main layout: left rail + camera wall ──────────────────────────────────
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Left rail (collapsible source list)
        Rectangle {
            id: leftRail
            Layout.preferredWidth: railVisible ? 200 : 0
            Layout.fillHeight: true
            color: root.clrSurface
            border.color: root.clrBorder
            border.width: 1
            clip: true
            visible: railVisible

            property bool railVisible: false  // collapsed by default in MVP

            Column {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 12
                spacing: 8
                topPadding: 12

                Text {
                    text: "Sources"
                    color: root.clrSubtext
                    font.pixelSize: 11
                    font.capitalization: Font.AllUppercase
                    font.letterSpacing: 1
                }

                Repeater {
                    model: engineClient.cameraModel
                    Text {
                        text: model.displayName
                        color: root.clrText
                        font.pixelSize: 13
                        elide: Text.ElideRight
                        width: parent.width
                    }
                }
            }
        }

        // Camera wall
        Item {
            id: wallArea
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Empty state
            Column {
                anchors.centerIn: parent
                spacing: 16
                visible: cameraGrid.count === 0

                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "No cameras yet"
                    color: root.clrSubtext
                    font.pixelSize: 22
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "Add a USB, RTSP, NDI, or ONVIF source to get started."
                    color: root.clrSubtext
                    font.pixelSize: 14
                }
                Button {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "+ Add Camera"
                    onClicked: addDialog.open()
                    background: Rectangle {
                        color: parent.pressed ? "#c73750" : root.clrAccent
                        radius: 6
                    }
                    contentItem: Text {
                        text: parent.text
                        color: "white"
                        font.pixelSize: 14
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    implicitWidth: 160
                    implicitHeight: 40
                }
            }

            // ── camera grid ───────────────────────────────────────────────────
            GridView {
                id: cameraGrid
                anchors.fill: parent
                anchors.margins: 8
                model: engineClient.cameraModel
                clip: true

                // Auto columns: ~square layout
                readonly property int cols: Math.max(1, Math.ceil(Math.sqrt(count)))
                cellWidth:  Math.floor(width  / cols)
                cellHeight: Math.floor(cellWidth * 9 / 16)

                displaced: Transition {
                    NumberAnimation {
                        properties: "x,y"
                        duration: 200
                        easing.type: Easing.InOutQuad
                    }
                }

                delegate: Item {
                    id: delegateRoot
                    width: cameraGrid.cellWidth
                    height: cameraGrid.cellHeight

                    // Cache model data so it stays accessible during drag state
                    readonly property string camId:        model.cameraId
                    readonly property string camName:      model.displayName
                    readonly property bool   camTracking:  model.trackingEnabled
                    readonly property var    camTracks:    model.tracks
                    readonly property real   camFps:       model.fps
                    readonly property string camHealth:    model.health
                    readonly property string camShmName:   model.shmName

                    // ── drag support ──────────────────────────────────────────
                    Drag.active:    dragMouse.drag.active
                    Drag.source:    delegateRoot
                    Drag.hotSpot.x: width / 2
                    Drag.hotSpot.y: height / 2

                    states: State {
                        when: delegateRoot.Drag.active
                        ParentChange {
                            target: delegateRoot
                            parent: cameraGrid
                        }
                        AnchorChanges {
                            target: delegateRoot
                            anchors.horizontalCenter: undefined
                            anchors.verticalCenter:   undefined
                        }
                    }

                    DropArea {
                        anchors { fill: parent; margins: 6 }
                        onEntered: function(drag) {
                            if (drag.source !== delegateRoot) {
                                engineClient.cameraModel.swapCameras(
                                    drag.source.camId, delegateRoot.camId)
                            }
                        }
                    }

                    // Global drag mouse area (covers the whole tile)
                    MouseArea {
                        id: dragMouse
                        anchors.fill: parent
                        drag.target: delegateRoot
                        // forward single-click to the tile for selection
                        onClicked: tile.selected = !tile.selected
                        onDoubleClicked: tile.maximized = !tile.maximized
                    }

                    CameraTile {
                        id: tile
                        anchors.fill: parent
                        cameraId:        delegateRoot.camId
                        displayName:     delegateRoot.camName
                        trackingEnabled: delegateRoot.camTracking
                        tracks:          delegateRoot.camTracks
                        fps:             delegateRoot.camFps
                        health:          delegateRoot.camHealth
                    }
                }
            }
        }
    }

    // ── Add Camera dialog ─────────────────────────────────────────────────────
    Dialog {
        id: addDialog
        title: "Add Camera"
        anchors.centerIn: parent
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 420

        background: Rectangle {
            color: root.clrSurface
            border.color: root.clrBorder
            border.width: 1
            radius: 8
        }

        contentItem: Column {
            spacing: 16
            padding: 8

            Text {
                text: "Source URI"
                color: root.clrSubtext
                font.pixelSize: 12
            }
            TextField {
                id: uriField
                width: 380
                placeholderText: "usb://0  |  rtsp://...  |  ndi://Name  |  onvif://..."
                color: root.clrText
                background: Rectangle {
                    color: root.clrBg
                    border.color: uriField.activeFocus ? root.clrAccent : root.clrBorder
                    border.width: 1
                    radius: 4
                }
            }
            Text {
                text: "Display name (optional)"
                color: root.clrSubtext
                font.pixelSize: 12
            }
            TextField {
                id: nameField
                width: 380
                placeholderText: "Camera 1"
                color: root.clrText
                background: Rectangle {
                    color: root.clrBg
                    border.color: nameField.activeFocus ? root.clrAccent : root.clrBorder
                    border.width: 1
                    radius: 4
                }
            }
        }

        onAccepted: {
            if (uriField.text.trim().length > 0) {
                engineClient.addCamera(uriField.text.trim(), nameField.text.trim())
                uriField.clear()
                nameField.clear()
            }
        }
    }

    // ── keyboard shortcuts ────────────────────────────────────────────────────
    // (Space = toggle tracking on selected camera — wired in CameraTile via
    //  a global handler; arrow keys for PTZ nudge are in CameraTile too.)
}

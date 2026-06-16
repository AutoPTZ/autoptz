/**
 * CameraTile.qml — live preview + telemetry overlays for one camera.
 *
 * Overlays (all drawn in QML, never touching the engine thread):
 *   • Person bounding boxes (Repeater from tracks model data)
 *   • Target highlight (thicker green border + target label)
 *   • Identity name + confidence chip
 *   • Dead-zone ellipse (when tracking is active)
 *   • Centre reticle crosshair (when tracking)
 *   • FPS / health chip (top-right)
 *   • Tracking state banner (LOST / RECONNECTING / ERROR)
 *
 * Controls:
 *   • Click a bounding box → SetTarget
 *   • Space key (when selected) → toggle tracking
 *   • Arrow keys (when selected) → PTZ nudge
 *   • Track toggle button (bottom-right)
 *   • Remove button (top-left hover)
 */
import QtQuick
import QtQuick.Controls

Item {
    id: tile

    // ── required properties (set by CameraWall delegate) ─────────────────────
    required property string cameraId
    required property string displayName
    required property bool   trackingEnabled
    required property var    tracks          // list<{track_id, bbox, identity, is_target, confidence}>
    required property real   fps
    required property string health          // "ok" | "reconnecting" | "error" | "stopped"

    // ── state ─────────────────────────────────────────────────────────────────
    property bool selected:  false
    property bool maximized: false

    // preview refresh counter — incremented by a Timer to bust the image cache
    property int _tick: 0

    // ── palette (mirrors CameraWall) ──────────────────────────────────────────
    readonly property color clrBg:       "#0d0d1a"
    readonly property color clrBorder:   "#252545"
    readonly property color clrAccent:   "#e94560"
    readonly property color clrText:     "#e8e8f0"
    readonly property color clrSubtext:  "#7070a0"
    readonly property color clrTracking: "#00cc66"
    readonly property color clrTarget:   "#00ff88"
    readonly property color clrBbox:     "#ffdd00"
    readonly property color clrLost:     "#ff6644"

    // ── root border: selection / tracking state ───────────────────────────────
    Rectangle {
        anchors.fill: parent
        color: tile.clrBg
        border.color: tile.selected     ? tile.clrAccent
                     : tile.trackingEnabled ? tile.clrTracking
                     : tile.clrBorder
        border.width: tile.selected ? 2 : 1
        radius: 4
    }

    // ── video preview ─────────────────────────────────────────────────────────
    Image {
        id: preview
        anchors {
            fill: parent
            margins: tile.selected ? 2 : 1
        }
        source: "image://frame/" + tile.cameraId + "?r=" + tile._tick
        fillMode: Image.PreserveAspectFit
        asynchronous: true
        cache: false

        // Calculated painted area (letterboxed inside the Image element)
        readonly property real scaleX: paintedWidth  / sourceSize.width  || 1.0
        readonly property real scaleY: paintedHeight / sourceSize.height || 1.0
        readonly property real offsetX: (width  - paintedWidth)  / 2
        readonly property real offsetY: (height - paintedHeight) / 2
    }

    // ── person bounding-box overlays ──────────────────────────────────────────
    Repeater {
        id: trackRepeater
        model: tile.tracks

        Item {
            // bbox values are normalized [0,1] relative to the source frame
            x: modelData.bbox.x1 * preview.paintedWidth  + preview.offsetX
            y: modelData.bbox.y1 * preview.paintedHeight + preview.offsetY
            width:  (modelData.bbox.x2 - modelData.bbox.x1) * preview.paintedWidth
            height: (modelData.bbox.y2 - modelData.bbox.y1) * preview.paintedHeight

            // box outline
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: modelData.is_target ? tile.clrTarget : tile.clrBbox
                border.width: modelData.is_target ? 2 : 1
                radius: 2
            }

            // identity / ID label
            Rectangle {
                anchors.bottom: parent.top
                anchors.left:   parent.left
                height: idLabel.implicitHeight + 4
                width:  idLabel.implicitWidth  + 8
                color: modelData.is_target ? "#cc00ff88" : "#cc333300"
                radius: 3
                visible: parent.width > 30

                Text {
                    id: idLabel
                    anchors.centerIn: parent
                    text: modelData.identity !== "" ? modelData.identity
                          : "ID " + modelData.track_id
                    color: "white"
                    font.pixelSize: 10
                }
            }

            // click-to-target hit area
            MouseArea {
                anchors.fill: parent
                onClicked: function(mouse) {
                    mouse.accepted = true
                    engineClient.setTarget(tile.cameraId, modelData.track_id)
                }
                cursorShape: Qt.PointingHandCursor
            }
        }
    }

    // ── centre reticle (when tracking) ───────────────────────────────────────
    Canvas {
        id: reticle
        anchors.centerIn: preview
        width:  22
        height: 22
        visible: tile.trackingEnabled
        opacity: 0.8

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            ctx.strokeStyle = "#00ff88"
            ctx.lineWidth = 1.5
            var cx = width / 2, cy = height / 2, r = 7, g = 4
            ctx.beginPath()
            ctx.arc(cx, cy, r, 0, Math.PI * 2)
            ctx.stroke()
            ctx.beginPath()
            ctx.moveTo(cx - r - g, cy); ctx.lineTo(cx + r + g, cy)
            ctx.moveTo(cx, cy - r - g); ctx.lineTo(cx, cy + r + g)
            ctx.stroke()
        }
    }

    // ── dead-zone ellipse (when tracking) ────────────────────────────────────
    Canvas {
        id: deadZone
        anchors.centerIn: preview
        // ~10 % of painted frame dimensions — cosmetic only in Phase 7
        width:  preview.paintedWidth  * 0.10
        height: preview.paintedHeight * 0.10
        visible: tile.trackingEnabled && width > 8

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            ctx.strokeStyle = "rgba(255,220,0,0.35)"
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.ellipse(0, 0, width, height)
            ctx.stroke()
        }

        onWidthChanged:  requestPaint()
        onHeightChanged: requestPaint()
        Component.onCompleted: requestPaint()
    }

    // ── FPS / health chip (top right) ─────────────────────────────────────────
    Rectangle {
        anchors.top:    parent.top
        anchors.right:  parent.right
        anchors.margins: 6
        width:  chipText.implicitWidth + 10
        height: chipText.implicitHeight + 4
        color: "#b3000000"
        radius: 4
        visible: tile.fps > 0 || tile.health !== "ok"

        Text {
            id: chipText
            anchors.centerIn: parent
            text: tile.health !== "ok"
                  ? tile.health.toUpperCase()
                  : Math.round(tile.fps) + " fps"
            color: tile.health === "error"        ? "#ff4444"
                 : tile.health === "reconnecting" ? tile.clrLost
                 : tile.fps > 20                  ? "#00ff88"
                 : tile.fps > 10                  ? "#ffcc00"
                 : "#ff6644"
            font.pixelSize: 10
            font.bold: true
        }
    }

    // ── camera name chip (top left) ────────────────────────────────────────────
    Rectangle {
        anchors.top:    parent.top
        anchors.left:   parent.left
        anchors.margins: 6
        width:  nameChip.implicitWidth + 10
        height: nameChip.implicitHeight + 4
        color: "#b3000000"
        radius: 4

        Text {
            id: nameChip
            anchors.centerIn: parent
            text: tile.displayName
            color: tile.clrText
            font.pixelSize: 10
            elide: Text.ElideRight
            maximumLineCount: 1
        }
    }

    // ── tracking state banner (LOST / ERROR) ──────────────────────────────────
    Rectangle {
        anchors.centerIn: parent
        width:  stateBanner.implicitWidth + 24
        height: stateBanner.implicitHeight + 10
        color: "#cc000000"
        radius: 6
        visible: tile.health !== "ok" || (tile.trackingEnabled && tile.fps === 0)

        Text {
            id: stateBanner
            anchors.centerIn: parent
            text: tile.health === "reconnecting" ? "● RECONNECTING"
                : tile.health === "error"         ? "⚠ ERROR"
                : tile.health === "stopped"        ? "⏹ STOPPED"
                : tile.trackingEnabled && tile.fps === 0 ? "◎ SEARCHING"
                : ""
            color: tile.health === "error" ? "#ff4444" : tile.clrLost
            font.pixelSize: 13
            font.bold: true
        }
    }

    // ── bottom bar: display name + tracking toggle ─────────────────────────────
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.left:   parent.left
        anchors.right:  parent.right
        anchors.margins: 1
        height: 32
        color: "#cc000000"
        radius: 3

        Row {
            anchors.fill: parent
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            spacing: 6

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: tile.trackingEnabled ? "▶ TRACKING" : "⏸ IDLE"
                color: tile.trackingEnabled ? tile.clrTracking : tile.clrSubtext
                font.pixelSize: 10
                font.bold: tile.trackingEnabled
            }

            Item { width: 1; height: 1; Layout.fillWidth: true }

            // Tracking toggle switch
            Switch {
                id: trackSwitch
                anchors.verticalCenter: parent.verticalCenter
                checked: tile.trackingEnabled
                onToggled: engineClient.enableTracking(tile.cameraId, checked)

                indicator: Rectangle {
                    implicitWidth: 36
                    implicitHeight: 18
                    radius: 9
                    color: trackSwitch.checked ? tile.clrTracking : "#444466"
                    border.color: trackSwitch.checked ? tile.clrTracking : tile.clrBorder

                    Rectangle {
                        x: trackSwitch.checked ? parent.width - width - 2 : 2
                        anchors.verticalCenter: parent.verticalCenter
                        width: 14; height: 14
                        radius: 7
                        color: "white"
                        Behavior on x { NumberAnimation { duration: 120 } }
                    }
                }
            }
        }
    }

    // ── frame refresh timer (10 Hz; paused when off-screen) ──────────────────
    Timer {
        interval: 100
        running:  tile.visible
        repeat:   true
        onTriggered: tile._tick++
    }

    // ── keyboard shortcuts (active when tile is selected) ────────────────────
    Keys.onPressed: function(event) {
        if (!tile.selected) return
        var nudge = 0.3
        switch (event.key) {
        case Qt.Key_Left:  engineClient.ptzNudge(tile.cameraId, -nudge, 0, 0); event.accepted = true; break
        case Qt.Key_Right: engineClient.ptzNudge(tile.cameraId,  nudge, 0, 0); event.accepted = true; break
        case Qt.Key_Up:    engineClient.ptzNudge(tile.cameraId, 0,  nudge, 0); event.accepted = true; break
        case Qt.Key_Down:  engineClient.ptzNudge(tile.cameraId, 0, -nudge, 0); event.accepted = true; break
        case Qt.Key_Space:
            engineClient.enableTracking(tile.cameraId, !tile.trackingEnabled)
            event.accepted = true
            break
        }
    }
    focus: tile.selected
}

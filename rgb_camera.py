cd ~/depthai-python/examples

cat <<'PY' > rgb_camera.py
import depthai as dai
import cv2

p = dai.Pipeline()

cam = p.createColorCamera()
# Use CAM_A if available (avoids RGB deprecation); else fall back to RGB
if hasattr(dai.CameraBoardSocket, "CAM_A"):
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
else:
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)

# XLinkOut (compatible with DepthAI 2.25)
xout = p.createXLinkOut() if hasattr(p, "createXLinkOut") else p.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(p) as dev:
    q = dev.getOutputQueue("video", 4, False)
    while True:
        frame = q.get().getCvFrame()
        cv2.imshow("RGB Preview", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cv2.destroyAllWindows()
PY

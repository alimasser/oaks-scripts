cat <<'PY' > depth_preview.py
import depthai as dai
import cv2, numpy as np

# Build pipeline
p = dai.Pipeline()

# ----- RGB (small preview) -----
cam = p.createColorCamera()
sock = dai.CameraBoardSocket.CAM_A if hasattr(dai.CameraBoardSocket, "CAM_A") else dai.CameraBoardSocket.RGB
cam.setBoardSocket(sock)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(640, 360)
cam.setInterleaved(False)

xoutRgb = p.createXLinkOut() if hasattr(p, "createXLinkOut") else p.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
cam.preview.link(xoutRgb.input)

# ----- Stereo depth -----
monoL = p.createMonoCamera(); monoR = p.createMonoCamera()
# Use new socket names if available
if hasattr(dai.CameraBoardSocket, "CAM_B") and hasattr(dai.CameraBoardSocket, "CAM_C"):
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
else:
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = p.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StudioDepth.PresetMode.HIGH_DENSITY) if hasattr(dai.node, "StudioDepth") else stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setConfidenceThreshold(150)  # lower than default to get more valid points
stereo.setDepthAlign(sock)          # align depth to RGB view

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xoutDepth = p.createXLinkOut() if hasattr(p, "createXLinkOut") else p.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# ----- Run -----
with dai.Device(p) as dev:
    qRgb   = dev.getOutputQueue("rgb",   4, False)
    qDepth = dev.getOutputQueue("depth", 4, False)

    near, far = 200.0, 8000.0  # mm display range ~0.2–8.0 m
    i = 0

    while True:
        # RGB (non-blocking)
        inRgb = qRgb.tryGet()
        if inRgb is not None:
            cv2.imshow("RGB", inRgb.getCvFrame())

        # Depth (blocking)
        raw = qDepth.get().getFrame().astype(np.float32)  # mm
        valid = raw > 0

        clamped = np.clip(raw, near, far)
        norm = np.zeros_like(clamped, dtype=np.uint8)
        if valid.any():
            norm[valid] = ((clamped[valid] - near) / (far - near) * 255.0).astype(np.uint8)

        vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[~valid] = (40, 40, 40)  # show invalid as dark gray
        cv2.imshow("Depth (0.2–8.0 m)", vis)

        # Diagnostics every ~30 frames
        i += 1
        if i % 30 == 0:
            if valid.any():
                dmin = raw[valid].min() / 1000.0
                dmax = raw[valid].max() / 1000.0
                print(f"Depth valid px: {valid.sum()}, range ~{dmin:.2f}-{dmax:.2f} m")
            else:
                print("No valid depth pixels yet")

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
PY

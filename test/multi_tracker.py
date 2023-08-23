from motrackers import CentroidTracker
from motrackers.detectors import TF_SSDMobileNetV2
from motrackers.utils import draw_tracks
import cv2
import numpy as np

model = TF_SSDMobileNetV2(
        weights_path="tensorflow_weights/frozen_inference_graph.pb",
        configfile_path="tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
        labels_path="tensorflow_weights/ssd_mobilenet_v2_coco_names.json",
        confidence_threshold=0.5,
        nms_threshold=0.5,
        draw_bboxes=True,
        use_gpu=False
    )
tracker = CentroidTracker(max_lost=5, tracker_output_format='mot_challenge') # or IOUTracker(...), CentroidKF_Tracker(...), SORT(...)
cap = cv2.VideoCapture(0)
while True:
    ok, image = cap.read()

    if not ok:
        print("Cannot read the video feed.")
        break

    image = cv2.resize(image, (700, 500))

    bboxes, confidences, class_ids = model.detect(image)

    # Filter for "person" detections (class_id == 1)
    people_indices = np.where(np.array(class_ids) == 1)[0]
    people_bboxes = np.array(bboxes)[people_indices]
    people_confidences = np.array(confidences)[people_indices]
    people_class_ids = np.array(class_ids)[people_indices]

    tracks = tracker.update(people_bboxes, people_confidences, people_class_ids)
    updated_image = model.draw_bboxes(image.copy(), people_bboxes, people_confidences, people_class_ids)
    updated_image = draw_tracks(updated_image, tracks)

    cv2.imshow("image", updated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
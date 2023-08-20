import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def estimate_pose(self, frame):
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the pose landmarks
        results = self.pose.process(rgb_image)

        # If landmarks are found, draw them
        if results.pose_landmarks:
            annotated_image = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            return annotated_image, results.pose_landmarks
        return frame, None

from typing import List

import norfair
import numpy as np
from norfair import Detection
from norfair.camera_motion import MotionEstimator

from inference import Converter, YoloV5
from soccer import Ball, Match


def get_ball_detections(
    ball_detector, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Uses a YOLO detector to get ball predictions and converts them to Norfair.Detection list.

    Parameters
    ----------
    ball_detector : YOLO
        YOLO detector for balls
    frame : np.ndarray
        Frame to get the ball detections from

    Returns
    -------
    List[norfair.Detection]
        List of ball detections
    """
    results = ball_detector(frame)[0]  # With ultralytics YOLOv8/v11 style
    boxes = results.boxes

    if boxes is None or boxes.xyxy.shape[0] == 0:
        #print('_______________________________')
        return []

    # Extract detection data
    xyxy = boxes.xyxy.cpu().numpy()       # (x1, y1, x2, y2)
    conf = boxes.conf.cpu().numpy()       # confidence scores
    clses = boxes.cls.cpu().numpy()       # class indices

    # Filter by confidence > 0.3
    valid_mask = conf > 0.3
    xyxy = xyxy[valid_mask]
    conf = conf[valid_mask]
    clses = clses[valid_mask]

    if xyxy.shape[0] == 0:
        return []

    # Build DataFrame for compatibility with Converter
    import pandas as pd
    ball_df = pd.DataFrame({
        "xmin": xyxy[:, 0],
        "ymin": xyxy[:, 1],
        "xmax": xyxy[:, 2],
        "ymax": xyxy[:, 3],
        "confidence": conf,
        "class": clses,
        "name": ["ball"] * len(clses)
    })

    return Converter.DataFrame_to_Detections(ball_df)


def get_player_detections(
    person_detector, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Detects players in the frame using a YOLO model and returns Norfair Detections.

    Parameters
    ----------
    person_detector : YOLO
        Trained YOLO model for detecting people.
    frame : np.ndarray
        Video frame.

    Returns
    -------
    List[norfair.Detection]
        List of detected players.
    """
    results = person_detector.predict(frame)[0]
    boxes = results.boxes

    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy()

    valid_mask = conf > 0.3
    xyxy = xyxy[valid_mask]
    conf = conf[valid_mask]
    clses = clses[valid_mask]

    import pandas as pd
    player_df = pd.DataFrame({
        "xmin": xyxy[:, 0],
        "ymin": xyxy[:, 1],
        "xmax": xyxy[:, 2],
        "ymax": xyxy[:, 3],
        "confidence": conf,
        "class": clses,
        "name": ["person"] * len(clses),  # You could use class_map if available
    })

    return Converter.DataFrame_to_Detections(player_df)



def create_mask(frame: np.ndarray, detections: List[norfair.Detection]) -> np.ndarray:
    """

    Creates mask in order to hide detections and goal counter for motion estimation

    Parameters
    ----------
    frame : np.ndarray
        Frame to create mask for.
    detections : List[norfair.Detection]
        Detections to hide.

    Returns
    -------
    np.ndarray
        Mask.
    """

    if not detections:
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    else:
        detections_df = Converter.Detections_to_DataFrame(detections)
        mask = YoloV5.generate_predictions_mask(detections_df, frame, margin=40)

    # remove goal counter
    mask[69:200, 160:510] = 0

    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an img

    Parameters
    ----------
    img : np.ndarray
        Image to apply the mask to
    mask : np.ndarray
        Mask to apply

    Returns
    -------
    np.ndarray
        img with mask applied
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img


def update_motion_estimator(
    motion_estimator: MotionEstimator,
    detections: List[Detection],
    frame: np.ndarray,
) -> "CoordinatesTransformation":
    """

    Update coordinate transformations every frame

    Parameters
    ----------
    motion_estimator : MotionEstimator
        Norfair motion estimator class
    detections : List[Detection]
        List of detections to hide in the mask
    frame : np.ndarray
        Current frame

    Returns
    -------
    CoordinatesTransformation
        Coordinate transformation for the current frame
    """

    mask = create_mask(frame=frame, detections=detections)
    coord_transformations = motion_estimator.update(frame, mask=mask)
    return coord_transformations

def update_with_sanity_check(self, frame_length, frame_height, new_detection: norfair.Detection=None):
    """
    Updates ball detection, rejecting unrealistic jumps.
    If jump is unrealistic, estimate new center using velocity.
    """
    if new_detection == None:
        print('new_detection=none')
        return
    new_center = self.get_center(new_detection.absolute_points)

    dy = frame_length * 0.05
    dx = frame_height * 0.07
    max_speed_px_per_frame=15
    #check if ball is outside of a normal range of movement (bc soccer is a continuous game hardcoding risk boxes should be fine
    #this goes in run_utils, where it is used as a check after detections are retrieved
    if (self.center_abs[0] - dx) > self.prev_center_abs[0] or (self.center_abs[1] - dy) > self.prev_center_abs[1]:
        dx1 = new_center[0] - self.center_abs[0]
        dy1 = new_center[1] - self.center_abs[1]
        distance = np.sqrt(dx1**2 + dy1**2)

        if distance > max_speed_px_per_frame:
            # Predict next position
            predicted_x = self.center_abs[0] + self.velocity[0]
            predicted_y = self.center_abs[1] + self.velocity[1]

            # Update detection absolute_points to predicted location
            corrected_points = np.array([
                [predicted_x - 5, predicted_y - 5],
                [predicted_x + 5, predicted_y + 5]
            ])
            new_detection.absolute_points = corrected_points

            # Recalculate center after correction
            new_center = self.get_center(corrected_points)

        # Update state
        if self.center_abs is not None:
            self.velocity = (new_center[0] - self.center_abs[0], new_center[1] - self.center_abs[1])

        self.prev_center_abs = self.center_abs
        self.detection = new_detection
def get_main_ball(ball : Ball, match: Match = None) -> Ball:
    """
    Gets the main ball from a list of balls detection

    The match is used in order to set the color of the ball to
    the color of the team in possession of the ball.

    Parameters
    ----------
    detections : List[Detection]
        List of detections
    match : Match, optional
        Match object, by default None

    Returns
    -------
    Ball
        Main ball
    """
    #ball = Ball(detection=None)

    if match:
        ball.set_color(match)

    # if detections:
    #     ball.detection = detections[0]

    return ball

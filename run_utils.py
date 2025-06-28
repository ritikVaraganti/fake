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


def get_main_ball(
    detections: List[Detection],
    frame_length,
    frame_height,
    prev_center: Tuple[float, float],
    velocity: Tuple[float, float],
    match: Match = None
) -> Tuple[Ball, Tuple[float, float], Tuple[float, float]]:
    """
    Gets the main ball from a list of detections with sanity check applied.

    Parameters
    ----------
    detections : List[Detection]
        List of ball detections
    frame_length : int
        Frame height
    frame_height : int
        Frame width
    prev_center : Tuple[float, float]
        Previous ball center for sanity check
    velocity : Tuple[float, float]
        Previous velocity for sanity check
    match : Match, optional
        Match object for team color

    Returns
    -------
    Ball
        Main ball after sanity check
    Tuple[float, float]
        Updated center
    Tuple[float, float]
        Updated velocity
    """

    ball = Ball(detection=None)

    if match:
        ball.set_color(match)

    if not detections:
        return ball, prev_center, velocity

    detection = detections[0]

    # Compute center
    x1, y1 = detection.absolute_points[0]
    x2, y2 = detection.absolute_points[1]
    new_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    dy = frame_length * 0.05
    dx = frame_height * 0.07
    max_speed_px_per_frame = 15

    # Sanity check logic
    if (new_center[0] - dx) > prev_center[0] or (new_center[1] - dy) > prev_center[1]:
        dx1 = new_center[0] - prev_center[0]
        dy1 = new_center[1] - prev_center[1]
        distance = np.sqrt(dx1 ** 2 + dy1 ** 2)

        if distance > max_speed_px_per_frame:
            # Predict next position
            predicted_x = prev_center[0] + velocity[0]
            predicted_y = prev_center[1] + velocity[1]

            corrected_points = np.array([
                [predicted_x - 5, predicted_y - 5],
                [predicted_x + 5, predicted_y + 5]
            ])
            detection.absolute_points = corrected_points

            # Recalculate center
            x1, y1 = corrected_points[0]
            x2, y2 = corrected_points[1]
            new_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Update velocity
    velocity = (new_center[0] - prev_center[0], new_center[1] - prev_center[1])
    prev_center = new_center

    ball.detection = detection

    return ball, prev_center, velocity

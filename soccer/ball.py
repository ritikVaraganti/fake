import cv2
import norfair
import numpy as np

from soccer.draw import Draw


class Ball:
    def __init__(self, detection: norfair.Detection):
        """
        Initialize Ball

        Parameters
        ----------
        detection : norfair.Detection
            norfair.Detection containing the ball
        """
        self.detection = detection
        self.color = None
        self.prev_center_abs = None
        self.velocity = (0, 0)
    def set_color(self, match: "Match"):
        """
        Sets the color of the ball to the team color with the ball possession in the match.

        Parameters
        ----------
        match : Match
            Match object
        """
        if match.team_possession is None:
            return

        self.color = match.team_possession.color

        if self.detection:
            self.detection.data["color"] = match.team_possession.color

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    @property
    def center(self) -> tuple:
        """
        Returns the center of the ball

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        center = self.get_center(self.detection.points)
        round_center = np.round_(center)

        return round_center

    @property
    def center_abs(self) -> tuple:
        """
        Returns the center of the ball in absolute coordinates

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        center = self.get_center(self.detection.absolute_points)
        round_center = np.round_(center)

        return round_center

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the ball on the frame

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with ball drawn
        """
        if self.detection is None:
            return frame

        return Draw.draw_detection(self.detection, frame)

    
    def update_with_sanity_check(self, new_detection: norfair.Detection, frame_length, frame_height, max_speed_px_per_frame=60):
        """
        Updates ball detection, rejecting unrealistic jumps.
        If jump is unrealistic, estimate new center using velocity.
        """
        new_center = self.get_center(new_detection.absolute_points)

        dy = frame_length * 0.05
        dx = frame_height * 0.07
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



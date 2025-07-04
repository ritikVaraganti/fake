import argparse
from ultralytics import YOLO

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
import math

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5, llava
from inference.filters import filters
from inference.llava import (
    llava_mega_image_inference
)
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import cv2
import numpy as np
from PIL import Image

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
llava_model.to("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
player_detector = YOLO("yolov8m.pt")
ball_detector = YOLO(args.model)

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(
    name="Chelsea",
    abbreviation="CHE",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(name="Man City", abbreviation="MNC", color=(240, 230, 188))
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city


player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=80,            # Smaller for small objects like the ball
    initialization_delay=1,           # Ball gets tracked as soon as it's detected once
    detection_threshold=0.2,          # Accepts lower-confidence detections (your p ~ 0.3)
    hit_counter_max=30,               # Object persists for a short while if detection is lost
)

#point of concern, have to update for every vid?




motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

prev_center = (0, 0)
velocity = (0, 0)

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

track_player_id = None
frame_buffer = []
buffer_size = 5
#prompt = "Evaluate player performance. Focus on key passes, tackles, fouls, and provide feedback."
z = 0
tracked_player = None
for i, frame in enumerate(video):
    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    for det in ball_detections:
        print("Bounding box points:", det.points)
        print("Data dictionary:", det.data)
        print("Confidence score:", det.scores)

    


    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )
    print(f'BALL TRACK OBJECTS{ball_track_objects}')
    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
    print(f'ball_detections 2: {ball_detections}')
    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )
    height, width, _ = frame.shape
    # Match update
    ball, prev_center, velocity = get_main_ball(
        detections=ball_detections,
        frame_length=height,
        frame_height=width,
        prev_center=prev_center,
        velocity=velocity,
        match=match
    )
    #print(f'ball: {ball}')
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    movement_threshold = 7  # pixels
    if tracked_player != None:   
        if tracked_player and math.sqrt(velocity[0]**2 + velocity[1]**2) > movement_threshold:
            frame_buffer.append(frame_rgb)


    if track_player_id is None:
        print("\nChoose a player ID to track:")
        for p in players:
            print(f"Player ID: {p.detection.data}")
        try:
            track_player_id = 5
        except:
            track_player_id = None

    tracked_player = next(
        (p for p in players if 'id' in p.detection.data and p.detection.data['id'] == track_player_id),
        None
    )
    if tracked_player is None:
        print(f"Player {track_player_id} not found on screen. They may be subbed off.")
        track_player_id = None
        frame_buffer.clear()
        continue
    current_time = i/fps
    if len(frame_buffer) == buffer_size:
        prompt = f"<|user|>\n<image>\nPretend you are a soccer coach talking to your player. In this series of images depicting player #{track_player_id}'s movement around {current_time:.1f} into the game, give me a rating on a scale of 1 to 100 and tell me in specific what he is doing right or wrong. If nothing can be observed, make your best judgement. Describe his actions as a whole, not on an image basis.\n<|assistant|>"
        result = llava_mega_image_inference(llava_processor, llava_model, frame_buffer, prompt)
        print(f"\nLLaVA output for player {track_player_id}: {result}\n")
        frame_buffer.clear()

    frame_pil = PIL.Image.fromarray(frame_rgb)
    if args.possession:
        frame_pil = Player.draw_players(players, frame_pil, confidence=False, id=True)
        frame_pil = path.draw(frame_pil, ball.detection, coord_transformations, match.team_possession.color)
        frame_pil = match.draw_possession_counter(frame_pil, possession_background, debug=False)
        if ball:
            frame_pil = ball.draw(frame_pil)

    if args.passes:
        frame_pil = Pass.draw_pass_list(frame_pil, match.passes, coord_transformations)
        frame_pil = match.draw_passes_counter(frame_pil, passes_background, debug=False)

    video.write(np.array(frame_pil))

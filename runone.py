import argparse
from ultralytics import YOLO

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from llava import (
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
player_detector = YOLO("yolov8m.pt")
ball_detector = YOLO(args.model)
hsv_classifier = HSVClassifier(filters=filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)


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

player_tracker = Tracker(mean_euclidean, 250, 3, 90)
ball_tracker = Tracker(mean_euclidean, 80, 1, 0.2, 30)
motion_estimator = MotionEstimator()
path = AbsolutePath()
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

track_player_id = None
frame_buffer = []
buffer_size = 5
prompt = "Evaluate player performance. Focus on key passes, tackles, fouls, and provide feedback."

for i, frame in enumerate(video):
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections
    coord_transformations = update_motion_estimator(motion_estimator, detections, frame)
    player_track_objects = player_tracker.update(players_detections, coord_transformations)
    ball_track_objects = ball_tracker.update(ball_detections, coord_transformations)
    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
    player_detections = classifier.predict_from_detections(player_detections, frame)
    height, width, _ = frame.shape
    ball, _, _ = get_main_ball(ball_detections, height, width, (0, 0), (0, 0), match)
    players = Player.from_detections(player_detections, teams)
    match.update(players, ball)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb)

    if track_player_id is None:
        print("\nChoose a player ID to track:")
        for p in players:
            print(f"Player ID: {p.id}")
        try:
            track_player_id = int(input("Enter player ID: "))
        except:
            track_player_id = None

    tracked_player = next((p for p in players if p.id == track_player_id), None)

    if tracked_player is None:
        print(f"Player {track_player_id} not found on screen. They may be subbed off.")
        track_player_id = None
        frame_buffer.clear()
        continue

    if len(frame_buffer) == buffer_size:
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


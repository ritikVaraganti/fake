import cv2
import numpy as np
from PIL import Image
import torch



def llava_mega_image_inference(llava_processor, llava_model, frames, prompt, resize_dim=(640, 360)):
    """
    Concatenates frames into a mega image and runs LLaVA inference on it.
    
    Parameters:
    -----------
    frames : list of np.ndarray
        List of video frames (BGR format, as from OpenCV).
    prompt : str
        Prompt for LLaVA to guide its output.
    resize_dim : tuple
        Dimensions to resize each frame before concatenation.
    
    Returns:
    --------
    result : str
        LLaVA's text output analyzing the mega image.
    """
    if not frames:
        return "No frames provided."

    # Resize and convert frames to RGB
    resized_frames = [cv2.resize(f, resize_dim) for f in frames]
    resized_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized_frames]

    # Concatenate horizontally
    mega_image_np = np.hstack(resized_frames)

    # Convert to PIL.Image
    mega_image_pil = Image.fromarray(mega_image_np)

    # Run LLaVA inference
    #prompt = "<|user|>\n<image>\nPretend you are a soccer coach talking to your player. In this series of images depicting player #12's movement with relation to the game, give me a rating on a scale of 1 to 10 and tell me in specific what he is doing right or wrong. If nothing can be observed, make your best judgement. Describe his actions as a whole, not on an image basis.\n<|assistant|>"

    inputs = llava_processor(text=prompt, images=mega_image_pil, return_tensors="pt").to("cuda:0")
    
    output = llava_model.generate(**inputs, max_new_tokens=200)
    #print(processor.tokenizer.decode(output[0], skip_special_tokens=True))

    # with torch.no_grad():
    #     outputs = llava_model.generate(**inputs, max_new_tokens=100)

    result = llava_processor.tokenizer.decode(output[0], skip_special_tokens=True)

    return result

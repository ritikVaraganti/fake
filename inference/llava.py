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
    inputs = llava_processor(text=prompt, images=mega_image_pil, return_tensors="pt").to(llava_model.device)

    with torch.no_grad():
        outputs = llava_model.generate(**inputs, max_new_tokens=100)

    result = llava_processor.decode(outputs[0], skip_special_tokens=True)

    return result

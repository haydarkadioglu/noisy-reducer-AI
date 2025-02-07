import tensorflow as tf
import numpy as np
import librosa
import cv2
import soundfile as sf
import os
import tempfile
from tqdm import tqdm
from predictions import predict

def process_video(video_path, model, batching_size=12000):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Temp file paths
        temp_audio = os.path.join(temp_dir, "temp_audio.wav")
        output_audio = os.path.join(temp_dir, "processed_audio.wav")
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        
        # Extract audio
        ffmpeg_command = f'ffmpeg -i "{video_path}" -ab 160k -ac 1 -ar 16000 -vn "{temp_audio}"'
        if os.system(ffmpeg_command) != 0:
            raise RuntimeError("Failed to extract audio from video")
            
        if not os.path.exists(temp_audio):
            raise FileNotFoundError("Failed to create temporary audio file")
            
        # Process audio
        try:
            processed_audio = predict(temp_audio, model, batching_size)
            sf.write(output_audio, processed_audio, 16000)
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")
            
        # Create output path
        output_path = os.path.join(
            os.path.dirname(video_path),
            os.path.splitext(os.path.basename(video_path))[0] + '_processed.mp4'
        )
        
        # Combine video with processed audio
        combine_command = f'ffmpeg -i "{video_path}" -i "{output_audio}" -c:v copy -c:a aac "{output_path}"'
        if os.system(combine_command) != 0:
            raise RuntimeError("Failed to combine video with processed audio")
            
        return output_path


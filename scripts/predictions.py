import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import os

def get_audio_in_batches(path, batching_size=12000):
    audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    audio_np = audio.numpy().squeeze()  


    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate.numpy(), target_sr=target_sample_rate)
    
    audio_batches = []
    total_samples = len(audio_np)
    for i in range(0, total_samples, batching_size):
        batch = audio_np[i:i + batching_size]
        
        if len(batch) < batching_size:
            batch = np.pad(batch, (0, batching_size - len(batch)), mode='constant')
        
        audio_batches.append(batch)

    return tf.stack(audio_batches)


def predict(path, model, batching_size=12000,
            gate=True, 
            expansion=True, 
            smooth=True):
    
    audio_batches = get_audio_in_batches(path, batching_size=batching_size)
    original_length = len(get_audio(path))
    predicted_batches = []
    for batch in audio_batches:
        frame = tf.squeeze(model.predict(tf.expand_dims(tf.expand_dims(batch, -1), 0), verbose=0))
        predicted_batches.append(frame.numpy())
    
    predicted_audio = np.concatenate(predicted_batches)
    
    predicted_audio = predicted_audio[:original_length]
    
    if gate:
        predicted_audio = noise_gate(predicted_audio)
    if expansion:
        predicted_audio = dynamic_expansion(predicted_audio)
    if smooth:
        predicted_audio = exponential_smooth(predicted_audio)



    return predicted_audio



def get_audio(path):
    audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    audio_np = audio.numpy().squeeze()  

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate.numpy(), target_sr=target_sample_rate)

    return audio_np
    

def noise_gate(data, threshold=0.01):
    return np.where(np.abs(data) > threshold, data, 0)

def dynamic_expansion(data, threshold=0.35, ratio=1.5):
    expanded = np.where(
        np.abs(data) > threshold,
        np.sign(data) * (np.abs(data) ** ratio),
        data
    )
    return expanded / np.max(np.abs(expanded))  


def exponential_smooth(data, alpha=0.9):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    return smoothed


"""
# This function has not after processing
"""

def predict2(path, model, batching_size=12000):
    audio_batches = get_audio_in_batches(path, batching_size=batching_size)
    original_length = len(get_audio(path))
    predicted_batches = []
    for batch in audio_batches:
        frame = tf.squeeze(model.predict(tf.expand_dims(tf.expand_dims(batch, -1), 0), verbose=0))
        predicted_batches.append(frame.numpy())
    
    predicted_audio = np.concatenate(predicted_batches)
    
    predicted_audio = predicted_audio[:original_length]
    
    return predicted_audio


"""
# This function for tflite model
"""

def predict_tflite(path, tflite_model_path, batching_size=12000):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get audio batches
    audio_batches = get_audio_in_batches(path, batching_size=batching_size)
    original_length = len(get_audio(path))
    
    predicted_batches = []
    for batch in audio_batches:
        # Prepare input
        input_data = np.expand_dims(np.expand_dims(batch, -1), 0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        frame = interpreter.get_tensor(output_details[0]['index'])
        predicted_batches.append(frame.squeeze())
    
    # Concatenate predictions
    predicted_audio = np.concatenate(predicted_batches)
    predicted_audio = predicted_audio[:original_length]
    
    # Apply post-processing
    noise_gated = noise_gate(predicted_audio)
    expanded = dynamic_expansion(noise_gated)
    smoothed = exponential_smooth(expanded)
    
    return smoothed
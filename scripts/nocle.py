import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import os
import noisereduce as nr


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
            smooth=True,
            noise_reduce=True):  # Yeni bir parametre ekledik
    
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
    
    if noise_reduce:
        # Gürültü azaltma işlemi
        predicted_audio = nr.reduce_noise(y=predicted_audio, sr=16000)  # Örnekleme oranını uygun şekilde ayarlayın

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

"""
# Spektral Gating (Gürültü Bastırma)
def spectral_gating(noisy_signal, sr):
    stft = librosa.stft(noisy_signal, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    
    noise_thresh = np.median(magnitude, axis=1)[:, None]  # Gürültü eşiğini belirle
    mask = magnitude > (1.5 * noise_thresh)  # Belirli bir eşikten düşük frekansları bastır
    
    filtered_stft = stft * mask  # Gürültüyü filtrele
    return librosa.istft(filtered_stft, hop_length=512)


def apply_wiener_filter(audio, noise_power=None, mysize=15, noise_var=0.01):
    """
    Wiener filtresi ile ses sinyalindeki gürültüyü azaltır.
    
    :param audio: Gürültülü ses sinyali (numpy array)
    :param noise_power: Gürültü gücü (None ise otomatik belirler)
    :param mysize: Filtre pencere boyutu (Daha büyük değer = Daha fazla gürültü bastırma)
    :param noise_var: Gürültü varyansı (Düşük değer = Daha agresif gürültü azaltma)
    :return: Gürültüsü azaltılmış ses sinyali
    """
    from scipy.signal import wiener
    filtered_audio = wiener(audio, mysize=mysize, noise=noise_power if noise_power else noise_var)
    return filtered_audio



def apply_gaussian_blur(audio, sigma=2):
    """
    Ses sinyaline Gauss filtresi uygular (bulanıklaştırma efekti).
    
    :param audio: Ses sinyali (numpy array)
    :param sigma: Gauss filtresinin standart sapması (Daha büyük = Daha fazla bulanıklaştırma)
    :return: İşlenmiş ses sinyali
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(audio, sigma=sigma)


def predict4(path, model, batching_size=12000,
            gate=True, 
            smooth=True,
            noise_reduce=True, 
            extra_filter=True,
            params=[],
            
            ):  # Yeni parametre ekledik
    
    audio_batches = get_audio_in_batches(path, batching_size=batching_size)
    original_length = len(get_audio(path))
    predicted_batches = []
    for batch in audio_batches:
        frame = tf.squeeze(model.predict(tf.expand_dims(tf.expand_dims(batch, -1), 0), verbose=0))
        predicted_batches.append(frame.numpy())
    
    predicted_audio = np.concatenate(predicted_batches)
    
    predicted_audio = predicted_audio[:original_length]
    
    if extra_filter:
        predicted_audio = spectral_gating(predicted_audio, sr=16000)  # Gürültüyü daha da azalt
        predicted_audio = apply_wiener_filter(predicted_audio, mysize= params[0] if params else 15)  # Cızırtıyı azalt
        predicted_audio = apply_gaussian_blur(predicted_audio, sigma= params[1] if len(params) > 1 else 2)  # Ses sinyaline bulanıklaştırma efekti uygula
    if gate:
        predicted_audio = noise_gate(predicted_audio)
    
    
    if smooth:
        predicted_audio = exponential_smooth(predicted_audio)
    
    if noise_reduce:
        predicted_audio = nr.reduce_noise(y=predicted_audio, sr=16000)  


    return predicted_audio

    
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
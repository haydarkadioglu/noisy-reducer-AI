"""

THIS FILE IS NOT WORKİNG

I'M STILL WORKİNG ON IT


"""


import numpy as np
import pyaudio
import wave
import logging
import tensorflow as tf
from queue import Queue
from threading import Thread
import time

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
BUFFER_SIZE = 12000
VIRTUAL_CABLE_NAME = "CABLE Input"

class AudioProcessor:
    def __init__(self, tflite_model_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.audio = pyaudio.PyAudio()
        self.buffer = np.array([], dtype=np.float32)
        self.input_queue = Queue()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Find and print available devices
        self.list_devices()
        self.virtual_cable_index = self.get_virtual_cable_index()
        
    def list_devices(self):
        self.logger.info("Available Audio Devices:")
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            self.logger.info(f"Device {i}: {dev['name']}")
    
    def get_virtual_cable_index(self):
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            if VIRTUAL_CABLE_NAME in dev['name']:
                self.logger.info(f"Found Virtual Cable at index {i}")
                return i
        raise Exception("Virtual Cable not found")
    
    def start_stream(self):
        try:
            self.input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._input_callback
            )
            
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                output_device_index=self.virtual_cable_index,
                frames_per_buffer=CHUNK
            )
            
            self.running = True
            Thread(target=self._process_thread).start()
            self.logger.info("Audio processing started")
            
        except Exception as e:
            self.logger.error(f"Stream error: {e}")
            self.stop()
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Monitor input levels
        level = np.abs(audio_data).mean()
        if level > 0.01:  # Only log when there's significant audio
            self.logger.info(f"Input level: {level:.3f}")
            
        self.input_queue.put(audio_data)
        return (None, pyaudio.paContinue)

    def _process_thread(self):
        while self.running:
            if not self.input_queue.empty():
                audio_chunk = self.input_queue.get()
                self.buffer = np.append(self.buffer, audio_chunk)
                
                while len(self.buffer) >= BUFFER_SIZE:
                    input_chunk = self.buffer[:BUFFER_SIZE]
                    self.buffer = self.buffer[BUFFER_SIZE:]
                    
                    # Reshape and normalize for TFLite model
                    input_chunk = input_chunk.reshape(1, BUFFER_SIZE, 1).astype(np.float32)
                    
                    try:
                        # Set input tensor
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_chunk)
                        
                        # Run inference
                        self.interpreter.invoke()
                        
                        # Get output tensor
                        processed = self.interpreter.get_tensor(self.output_details[0]['index']).squeeze()
                        
                        # Normalize output
                        max_val = np.max(np.abs(processed))
                        if max_val > 0:
                            processed = processed / max_val * 0.9
                        
                        # Write to output
                        self.output_stream.write(processed.astype(np.float32).tobytes())
                        
                        # Debug: Save a sample to WAV file
                        if not hasattr(self, 'debug_saved'):
                            self._save_debug_audio(processed)
                            self.debug_saved = True
                            
                    except Exception as e:
                        self.logger.error(f"Processing error: {e}")

    def _save_debug_audio(self, audio_data):
        with wave.open('debug_output.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    def stop(self):
        self.running = False
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()
        self.logger.info("Audio processing stopped")

if __name__ == "__main__":
    processor = AudioProcessor("model/quantized_model.tflite")
    processor.start_stream()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        processor.stop()
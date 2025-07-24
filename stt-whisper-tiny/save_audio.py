import pyaudio
import threading
import numpy as np
import scipy.io.wavfile

class AudioSave:
    def __init__(self, path=None):
        """
        요청 받았을때 오디오를 스트리밍 하여 원하는 만큼 저장
        """
        self.path = path
        self.sr = 16000
        self.chunk = int(self.sr / 10)
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
    
    def run(self):
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.start()

    def _run(self):
        '''
        run() thread
        '''
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.sr, input=True, frames_per_buffer=self.chunk)
        self.buffer = []
        self.streaming_status = True
        while self.streaming_status:
            one_chunk = self.stream.read(self.chunk)
            self.buffer.append(one_chunk)

    def stop(self, filename):
        '''
        녹음 중지
        '''
        self.streaming_status = False
        self.run_thread.join()
        return self._buffer_to_numpy(self.buffer, filename), self.sr

    def _buffer_to_numpy(self, buffer, filename):
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Convert to float32
        
        # Save to output.wav
        scipy.io.wavfile.write(filename, self.sr, (audio_data * 32768).astype(np.int16))
        
        return audio_data
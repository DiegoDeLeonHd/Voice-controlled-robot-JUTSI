import os
import rclpy
from rclpy.node import Node
import pyaudio
import wave
from pydub import AudioSegment
import whisper
import json
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from gtts import gTTS
import tempfile
import subprocess
import signal
import time 

# sistema_robot por robot_system "POSIBLE CAMBIO DE NOMBRE DEL PAQUETE, A TOMAR EN CUENTA EN ROS"

model = whisper.load_model("medium")


class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')
        
        # Subscriber
        self.create_subscription(String, 'feedback', self.callback_retro, 10)
        
        #Publisher
        self.publisher_ = self.create_publisher(String, 'voice_command', 10)
        self.start_node("sistema_robot", "vision_node"  )

        # Package path and internal subdirectories
        package_dir = get_package_share_directory('sistema_robot')
        self.base_dir = os.path.join(package_dir, 'data')
        self.audios_dir = os.path.join(self.base_dir, 'B_Audios')
        self.transcribes_dir = os.path.join(self.base_dir, 'D_Transcribes')

        os.makedirs(self.audios_dir, exist_ok=True)
        os.makedirs(self.transcribes_dir, exist_ok=True)

        # Initial greeting with gTTS and mpg123
        self.initial_greeting()

        while rclpy.ok():
            self.get_logger().info("Press 'Enter' to start recording...")
            input("Waiting for 'Enter'...")
            self.record_and_process()

    def initial_greeting(self):
        speech = "Hi, my name is iutsi, your medical assistant. How can I help you today? Please press enter to give me instructions." 
        self.get_logger().info("Playing welcome greeting with gTTS and mpg123...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                gTTS(text=speech, lang='en').save(f.name)
                os.system(f"mpg123 {f.name}")
        except Exception as e:
            self.get_logger().error(f"Error playing greeting: {e}")
            
    def callback_feedback(self, msg):
        self.parts_feedback = msg.data.split(";")
                
    def record_and_process(self):
        
        self.parts_feedback = []
        
        # Wait for feedback from the vision node
        wait_time = 0
        while not self.parts_feedback and wait_time < 5:
            
            self.get_logger().info("Waiting for vision-detected tools...")
            rclpy.spin_once(self, timeout_sec=0.5)
            wait_time += 0.5

        if not self.parts_feedback:
            self.get_logger().warn("No tools detected. Aborting recording.")
            return
        
        parts = self.parts_feedback if self.parts_feedback else []
                
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        DURACION = 5

        wav_path = os.path.join(self.audios_dir, 'audio.wav')
        mp3_path = os.path.join(self.audios_dir, 'audio.mp3')
        json_path = os.path.join(self.transcribes_dir, 'speak.json')

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        self.get_logger().info("Recording audio...")
        frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * DURACION))]
        self.get_logger().info("Recording finished.")
        

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        result = model.transcribe(wav_path, language='en')["text"].lower()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f"Transcription: {result}")

        action = self.search_match('C_Words', result)
        object = self.search_match('C_Objects', result)
        size = self.search_match('C_Sizes', result)
        
        if action:
            if object and object in parts :
                message = f"{action};{object}"
                msg = String()
                msg.data = message
                self.publisher_.publish(msg)
                self.get_logger().info(f"Command sent: {message}")
                
            elif action == "take" and object not in parts:
                message = f"{action};{object}"
                msg = String()
                msg.data = message
                self.publisher_.publish(msg)
                self.get_logger().info(f"Command sent: {message}")
                
            elif size and action == "bind up" and "bandage" in parts:
                message = f"{action};{size}"
                msg = String()
                msg.data = message
                self.publisher_.publish(msg)
                self.get_logger().info(f"Command sent: {message}")
                
                
            elif action == "cut" and "scalpel" in parts:
                message = f"{action};body"
                msg = String()
                msg.data = message
                self.publisher_.publish(msg)
                self.get_logger().info(f"Command sent: {message}") 
                
                
            else:
                self.get_logger().warn("No object or size detected.")
                speech = "No object or size detected." 
                self.get_logger().info("Playing...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        gTTS(text=speech, lang='en').save(f.name)
                        os.system(f"mpg123 {f.name}")
                except Exception as e:
                    self.get_logger().error(f"Error playing greeting: {e}")
        else:
            self.get_logger().warn("No object or size detected.")
            speech = "No object or size detected." 
            self.get_logger().info("Playing...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        gTTS(text=speech, lang='en').save(f.name)
                        os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"Error playing greeting: {e}")
    
    def start_node(self, sistema_robot, vision_node):
            command = ["gnome-terminal","--","ros2", "run", "sistema_robot", "vision_node"]
            process = subprocess.Popen(command)
            self.get_logger().info(f"Nodo {vision_node} lanzado.")
            return process
        
    def search_match(self, subfolder, text):
        package_dir = get_package_share_directory('sistema_robot')
        file = os.path.join(package_dir, 'data', subfolder)

        if not os.path.exists(file):
            self.get_logger().warn(f"Folder not found: {file}")
            return None

        for archive in os.listdir(file):
            if archive.endswith(".txt"):
                path_file = os.path.join(file, archive)
                with open(path_file, 'r', encoding='utf-8') as f:
                    synonyms = [p.strip().lower() for p in f.read().split(',') if p.strip()]
                    for word in synonyms:
                        if word in text:
                            self.get_logger().info(f"Match found: '{word}' on file '{archive}'")
                            return os.path.splitext(archive)[0]
        return None

def main(args=None):
    
    rclpy.init(args=args)
    nodo = VoiceNode()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()
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

model = whisper.load_model("medium")


class VozCompletaNode(Node):
    def _init_(self):
        super()._init_('voz_comando_node')
        
        # Suscriptores
        self.create_subscription(String, 'retroalimentacion', self.callback_retro, 10)
        
        self.publisher_ = self.create_publisher(String, 'voz_comando', 10)
        self.start_node("sistema_robot", "vision_node"  )
        # 📦 Ruta del paquete y subdirectorios internos
        package_dir = get_package_share_directory('sistema_robot')
        self.base_dir = os.path.join(package_dir, 'data')
        self.audios_dir = os.path.join(self.base_dir, 'B_Audios')
        self.transcribes_dir = os.path.join(self.base_dir, 'D_Transcribes')

        os.makedirs(self.audios_dir, exist_ok=True)
        os.makedirs(self.transcribes_dir, exist_ok=True)

        # 🔊 Saludo inicial con gTTS y mpg123
        self.saludo_inicial()

        while rclpy.ok():
            self.get_logger().info("Presiona 'Enter' para iniciar grabación...")
            input("▶️ Esperando tecla 'Enter'...")
            self.grabar_y_procesar()

    def saludo_inicial(self):
        texto = "Hi" #, my name is iutsi, your medical assistant. How can I help you today? Please press enter to give me instructions.
        self.get_logger().info("🔊 Reproduciendo saludo de bienvenida con gTTS y mpg123...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                gTTS(text=texto, lang='en').save(f.name)
                os.system(f"mpg123 {f.name}")
        except Exception as e:
            self.get_logger().error(f"❌ Error al reproducir saludo: {e}")
            
    def callback_retro(self, msg):
        self.partes_retro = msg.data.split(";")
                
    def grabar_y_procesar(self):
        
        self.partes_retro = []
        
        #Espera por retroalimentación del nodo de visión
        wait_time = 0
        while not self.partes_retro and wait_time < 5:
            self.get_logger().info("Esperando herramientas detectadas por visión...")
            rclpy.spin_once(self, timeout_sec=0.5)
            wait_time += 0.5

        if not self.partes_retro:
            self.get_logger().warn("No se detectaron herramientas. Abortando grabación.")
            return
        
        partes = self.partes_retro if self.partes_retro else []
                
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

        self.get_logger().info("🎤 Grabando audio...")
        frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * DURACION))]
        self.get_logger().info("✅ Grabación finalizada.")

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

        self.get_logger().info(f"📝 Transcripción: {result}")

        accion = self.buscar_coincidencia('C_Words', result)
        objeto = self.buscar_coincidencia('C_Objects', result)
        tamaño = self.buscar_coincidencia('C_Sizes', result)
        
        if accion:
            if objeto and objeto in partes :
                mensaje = f"{accion};{objeto}"
                msg = String()
                msg.data = mensaje
                self.publisher_.publish(msg)
                self.get_logger().info(f"📢 Comando enviado: {mensaje}")
                
            elif accion == "take" and objeto not in partes: 
                mensaje = f"{accion};{objeto}"
                msg = String()
                msg.data = mensaje
                self.publisher_.publish(msg)
                self.get_logger().info(f"📢 Comando enviado: {mensaje}")
                
            elif tamaño and accion == "bind up" and "bandage" in partes:
                mensaje = f"{accion};{tamaño}"
                msg = String()
                msg.data = mensaje
                self.publisher_.publish(msg)
                self.get_logger().info(f"📢 Comando enviado: {mensaje}")
            elif accion == "cut" and "scalpel" in partes:
                mensaje = f"{accion};body"
                msg = String()
                msg.data = mensaje
                self.publisher_.publish(msg)
                self.get_logger().info(f"📢 Comando enviado: {mensaje}") 
            else:
                self.get_logger().warn("No se detectó objeto o tamaño.")
        else:
            self.get_logger().warn("Accion u objeto no encontrada.")

    def start_node(self, sistema_robot, vision_node):
            command = ["gnome-terminal","--", "ros2", "run", "sistema_robot", "vision_node"]
            process = subprocess.Popen(command)
            self.get_logger().info(f"Nodo {vision_node} lanzado.")
            return process
        
    def buscar_coincidencia(self, subcarpeta, texto):
        paquete_dir = get_package_share_directory('sistema_robot')
        carpeta = os.path.join(paquete_dir, 'data', subcarpeta)

        if not os.path.exists(carpeta):
            self.get_logger().warn(f"📁 Carpeta no encontrada: {carpeta}")
            return None

        for archivo in os.listdir(carpeta):
            if archivo.endswith(".txt"):
                archivo_path = os.path.join(carpeta, archivo)
                with open(archivo_path, 'r', encoding='utf-8') as f:
                    sinonimos = [p.strip().lower() for p in f.read().split(',') if p.strip()]
                    for palabra in sinonimos:
                        if palabra in texto:
                            self.get_logger().info(f"🔍 Coincidencia encontrada: '{palabra}' en archivo '{archivo}'")
                            return os.path.splitext(archivo)[0]

        return None

def main(args=None):
    
    rclpy.init(args=args)
    nodo = VozCompletaNode()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown(

import os
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import math
import time
import queue
import datetime
import random
import traceback
import termios
import tty
import threading
from xarm import version
from xarm.wrapper import XArmAPI
from gtts import gTTS
import tempfile
import pyrealsense2 as rs
import numpy as np
import cv2

def coordinates_detection(): 

    # Start the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Obtener la intrínseca de la cámara color
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def preprocesar_imagen(img_color):
        gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        return gris

    def detectar_circulo(gris):
        circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                    param1=10, param2=30, minRadius=15, maxRadius=20)
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            return circulos[0][0]
        return None

    def detectar_cuadrado(gris):
        bordes = cv2.Canny(gris, 13, 13)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2 and w > 20 and h > 20:
                    cx = x + w // 2
                    cy = y + h // 2
                    return (cx, cy), approx
        return None, None

    coordenadas_inicio = None
    coordenadas_fin = None
    coordenadas_guardadas = False
    offset = np.array([86.8, 0, 69.5])

    try:
        cv2.namedWindow("Coordinates Detection", cv2.WINDOW_NORMAL)
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            img_color = np.asanyarray(color_frame.get_data())
            img_gris = preprocesar_imagen(img_color)
            img_display = img_color.copy()

            circulo = detectar_circulo(img_gris)
            cuadrado, contorno = detectar_cuadrado(img_gris)

            if circulo is not None:
                x, y, r = circulo
                cv2.circle(img_display, (x, y), r, (0, 255, 0), 2)
                depth_m = depth_frame.get_distance(x, y)
                coords_circulo_3D = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth_m)
                coords_circulo_3D =[round(coord * 100, 1) for coord in coords_circulo_3D]  # mm
                coords_circulo_3D_robot = np.array(coords_circulo_3D) + offset

                if coords_circulo_3D_robot[2] <= 99.3:
                    thick_offset_start = [10.3,0,34]
                    coords_circle_tool_3D = np.array(coords_circulo_3D_robot) - thick_offset_start
                
                elif 104.2 > coords_circulo_3D_robot[2] > 99.3:
                    mid_offset_start = [10.3,0,46.2]
                    coords_circle_tool_3D = np.array(coords_circulo_3D_robot) - mid_offset_start

                elif coords_circulo_3D_robot[2] >= 104.2:
                    thin_offset_start = [10.3,0,57.8] 
                    coords_circle_tool_3D = np.array(coords_circulo_3D_robot) - thin_offset_start

                texto_circulo = f"Inicio: {coords_circle_tool_3D[0]:.1f}, {coords_circle_tool_3D[1]:.1f}, {coords_circle_tool_3D[2]:.1f} mm"
                cv2.putText(img_display, texto_circulo, (x + 10, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                texto_circulo = f"Inicio: {coords_circulo_3D_robot[0]:.1f}, {coords_circulo_3D_robot[1]:.1f}, {coords_circulo_3D_robot[2]:.1f} mm"
                cv2.putText(img_display, texto_circulo, (x + 10, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if cuadrado is not None:
                cx, cy = cuadrado
                cv2.drawContours(img_display, [contorno], -1, (0, 0, 255), 2)
                depth_m = depth_frame.get_distance(cx, cy)
                coords_cuadrado_3D = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth_m)
                coords_cuadrado_3D = [round(coord * 100, 1) for coord in coords_cuadrado_3D]  # mm
                coords_cuadrado_3D_robot = np.array(coords_cuadrado_3D) + offset

                if coords_cuadrado_3D_robot[2] >= 91.9:
                    thick_offset_end = [20, 0, 23]
                    coords_square_tool_3D = np.array(coords_cuadrado_3D_robot) - thick_offset_end
                elif 99.8> coords_cuadrado_3D_robot > 91.9:
                    mid_offset_end = [20, 0, 29]
                    coords_square_tool_3D = np.array(coords_cuadrado_3D_robot) - mid_offset_end
                elif coords_cuadrado_3D_robot >= 99.8: 
                    thin_offset_end = [20, 0, 35]
                    coords_square_tool_3D = np.array(coords_cuadrado_3D_robot) - thin_offset_end

                texto_cuadrado = f"Fin: {coords_cuadrado_3D_robot[0]:.1f}, {coords_cuadrado_3D_robot[1]:.1f}, {coords_cuadrado_3D_robot[2]:.1f} mm"
                cv2.putText(img_display, texto_cuadrado, (cx + 10, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if circulo is not None and cuadrado is not None and not coordenadas_guardadas:
                # coordenadas_inicio = [(364 - coords_circle_tool_3D[0]), 1.5, coords_circle_tool_3D[2]] 
                # coordenadas_fin = [(458 - coords_square_tool_3D[0]), 1.5, coords_square_tool_3D[2]]
                coordenadas_inicio = [290.4, -6.5, 68.7] 
                coordenadas_fin = [380.3, -6.5, 76.9]
                
                #if coordenadas_inicio[2] >= 42 and coordenadas_inicio[2] <= 60 and coordenadas_fin[2]>=40 and coordenadas_fin[2] <=84: 
                coordenadas_guardadas = True
                print("Coordenadas guardadas:")
                print(f"Inicio: {coordenadas_inicio}")
                print(f"Fin: {coordenadas_fin}")

            if circulo is not None and cuadrado is not None:
                cv2.line(img_display, (x, y), (cx, cy), (255, 0, 0), 2)

            cv2.imshow("Coordinates Detection", img_display)
            if cv2.waitKey(2) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        return coordenadas_inicio, coordenadas_fin

class Movimiento(Node):
    def __init__(self, arm):  
        super().__init__('movement')  
        
        # Suscripción al tópico voz_comando
        self.subscription = self.create_subscription(String,'voice_command',self.callback_voz,10)
       
        #Variables de estado
        self.alive = True
        self._arm = arm
        self._ignore_exit_state = False
        self.emergency_mode = False
        self.key_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.key_thread.start()
        self.modo_exclusivo = False


        # Velocidades y aceleraciones (puedes ajustar si quieres)
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500

        self._vars = {}
        self._funcs = {}
        #self.saludo_inicial()
        
        # Diccionario con coordenadas (vacío o con datos si quieres)
        self.coordenadas = {
            "scalpel": {"arriba": [-336.6, -268.5, 188.1, 179.7, -0.2, -179.2],
                        "abajo": [-336.6, -268.5, 16.7, 179.7, -0.2, -179.2],
                        "dejar": [-257.8, -36.1, 40, 180, 0, 0.2]
                        },
            
            "tweezers": {"arriba": [-351.1, -132.2, 188.1, 179.7, -0.2, -179.2],
                         "abajo": [-351.1, -132.2, 26.5, 179.7, -0.2, -179.2],
                         "dejar": [-257.8, -36.1, 40, 180, 0, 0.2]
                         },
            
            "bandage": {"arriba": [-195.4,-255.9,165.1,179.7,0.2,-164.1],
                        "arriba2": [-196.5,-257.5,165.1,-179.9,0.3,-89.6],
                        "abajo": [-196.5,-257.5,126.2,179.7,0.2,-164.1],
                        "abajo2": [-196.5,-257.5,126.2,-179.9,0.3,-89.6],
                        "intermedio": [-196.5,-261,147.3,179.7,0.2,-164.1],
                        "dejar": [-257.8, -36.1, 160, 180, 0, 0.2]},
            
            "small": {},
            "big": {},
            
            
        }
        # Flag para bloquear subrutinas si hay error
        self.return_subrutina = True
        # Inicializar el robot (limpieza, callbacks, etc)
        self._robot_init()#cambiar arm por robot si no funciona
        self.get_logger().warn('Nodo movimiento inicializado y robot conectado')
    
    
 
 
    def _keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == 'q':
                    self.get_logger().warn("🟡 Tecla 'q' presionada: CANCELANDO todo y ejecutando rutina especial")
                    
                    self._arm.set_pause_time(0)
                    self._arm.set_state(4)  # ⛔ Detener

                    self.modo_exclusivo = True

                    threading.Thread(target=self.mi_rutina_especial, daemon=True).start()

                elif ch == 'w':
                    self.get_logger().info("🟢 Tecla 'w' presionada: saliendo de modo exclusivo")

                    self.modo_exclusivo = False
                    self._arm.set_mode(0)
                    self._arm.set_state(0)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


    def safe_move(self, coords, label):
        if self.modo_exclusivo:
            self.get_logger().warn(f"⛔ {label} cancelado: modo exclusivo activo antes de iniciar movimiento")
            return False

        code = self._arm.set_position(*coords, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if code != 0:
            self.get_logger().error(f"❌ Error en {label}: código {code}")
            return False

        while self._arm.get_is_moving():
            if self.modo_exclusivo:
                self.get_logger().warn(f"🛑 Cancelando '{label}' en movimiento por emergencia")
                self._arm.set_pause_time(0)
                self._arm.set_state(4)
                return False
            time.sleep(0.05)

        return True


    def mi_rutina_especial(self):
        self.get_logger().info("🔧 Ejecutando mi rutina especial")

        # Poner en modo exclusivo y detener todo
        self._arm.set_pause_time(0)
        self._arm.set_state(4)
        self.modo_exclusivo = True
        time.sleep(1)

        # Reactivar el brazo antes de comenzar
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.5)

        # Paso 1
        self.get_logger().info("🟢 Paso 1")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-129.2, -1.3, 88.9, 0.0, 90.2, 23.9], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 1'): return

        # Paso 2
        self.get_logger().info("🟢 Paso 2")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-173.1, 6.9, 86.6, 0.0, 79.7, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 2'): return

        # Paso 3
        self.get_logger().info("🟢 Paso 3")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.8, 13.0, 56.3, 0.5, 43.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 3'): return

        # Paso 4
        self.get_logger().info("🟢 Paso 4")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.6, 42.1, 53.6, 0.9, 11.3, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 4'): return

        # Pausa
        time.sleep(2)

        # Paso 5: abrir gripper
        self.get_logger().info("🟢 Paso 5: abrir gripper")
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'Paso 5 - abrir gripper'): return
        time.sleep(1.5)

        # Paso 6: volver a Paso 3
        self.get_logger().info("🟢 Paso 6: volver a Paso 3")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.8, 13.0, 56.3, 0.5, 43.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 6'): return
        time.sleep(1.5)

        # Paso 7: cerrar gripper
        self.get_logger().info("🟢 Paso 7: cerrar gripper")
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'Paso 7 - cerrar gripper'): return
        time.sleep(1)

        # Paso 8: detener gripper
        self.get_logger().info("🟢 Paso 8: detener gripper")
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'Paso 8 - detener gripper'): return
        time.sleep(1)

        # Paso 9: mover a posición final 1
        self.get_logger().info("🟢 Paso 9")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-112.9, -0.8, 60.7, 0.4, 60.8, -112.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 9'): return

        # Paso 10: posición final 2
        self.get_logger().info("🟢 Paso 10")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[0.0, -6.9, 38.9, 0.2, 45.9, -0.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Paso 10'): return

        # Final
        self.get_logger().info("✅ Rutina especial completada correctamente")
            
        
      
     

    
    
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)

        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint(f'Error detectado: err={data["error_code"]}, quit')
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('Estado 4 detectado, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint(f'Contador: {data["count"]}')

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(f'{label}, code={code}, connected={self._arm.connected}, state={self._arm.state}, error={self._arm.error_code}, ret1={ret1}, ret2={ret2}')
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}][{stack_tuple[1]}] {" ".join(map(str, args))}')
        except Exception:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._ignore_exit_state:
                return True
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    def callback_voz(self, msg):
        instruccion = msg.data.strip()
        self.get_logger().info(f'Recibida instrucción: {instruccion}')

        partes = instruccion.split(";")
        if len(partes) < 2:
            self.get_logger().warn('Instrucción incompleta.')
            return

        accion = partes[0]
        objeto = partes[1]

        if accion == 'bring':
            #self.bring(objeto)
            texto = f"Bringing the {objeto}, please receive it in the designated area."
            self.get_logger().info("🔊 Reproduciendo mensaje con gTTS y mpg123...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=texto, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error al reproducir saludo: {e}")
        
            if objeto == "bandage":
                self.bring_bandage(objeto)
            else:
                self.bring(objeto)
         
             
        elif accion == 'take':
            #self.take(objeto)
            texto = f"I will take the {objeto} to the area for contaminated tools, please deliver it in the designated area and restock when possible."
            self.get_logger().info("🔊 Reproduciendo mensaje con gTTS y mpg123...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=texto, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error al reproducir saludo: {e}")
            self.take(objeto)
                
        elif accion == 'cut':
            #self.cut(objeto)
            texto = "Please, make sure the arm is on the designated area, and has the start and ending points marked.  The robot will start the cut in a phew seconds,  press stop in case of emergency"
            self.get_logger().info("🔊 Reproduciendo...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=texto, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error al reproducir saludo: {e}")
            self.cut()
            
        elif accion == 'bind up':
            #self.bind_up(objeto)
            texto = "For this action please place your arm on the holder, and make sure your wrist is aligning with the bandage.   Don't move your hand and keep it open with the palm towards the ceiling, your fingers must be fully extended.  The robot will start shortly."
            self.get_logger().info("🔊 Reproduciendo mensaje...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=texto, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error al reproducir saludo: {e}")
            if objeto == "small":
                self.bind_small(objeto)
            elif objeto == "big":
                self.bind_big(objeto)    
            
        else:
            self.get_logger().warn(f'Acción no reconocida: {accion}')
            texto = "Unknown command"
            self.get_logger().info("🔊 Reproduciendo...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=texto, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error al reproducir saludo: {e}")  
                
                
                    
    def bring(self, objeto):
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina")
            return

        self.get_logger().info(f'Iniciando bring de {objeto}')
        if not self.return_subrutina:
            return

        if objeto not in self.coordenadas:
            self.get_logger().error(f'Coordenadas no definidas para: {objeto}')
            return

        coords_arriba = self.coordenadas[objeto]['arriba']
        coords_abajo = self.coordenadas[objeto]['abajo']

        if not self.safe_move([199.8, 1.8, 199.0, 180.0, 0.0, 0.2], 'bring - home'):
            return

        if self.modo_exclusivo:
            return

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'bring - open_lite6_gripper'):
            return

        if not self.safe_move([57.8, -189.1, 199.0, 180.0, 0, -92.7], 'bring - intermedio'):
            return

        if not self.safe_move(coords_arriba, 'bring - coords_arriba'):
            return

        time.sleep(0.5)

        if not self.safe_move(coords_abajo, 'bring - coords_abajo'):
            return

        time.sleep(2)

        if self.modo_exclusivo:
            return

        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'bring - close_lite6_gripper'):
            return

        time.sleep(2)

        if not self.safe_move(coords_arriba, 'bring - levantar objeto'):
            return

        time.sleep(0.5)

        if not self.safe_move([-336.6, -176.9, 225.9, 179.7, 0.2, -179.2], 'bring - mov1'):
            return

        if not self.safe_move([1.6, -321.3, 192.6, 179.7, -0.2, -78.1], 'bring - mov2'):
            return

        if not self.safe_move([360, 217, 199, -179.7, -0.2, 20.4], 'bring - entrega'):
            return

        time.sleep(2)

        if self.modo_exclusivo:
            return

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'bring - open_lite6_gripper final'):
            return

        time.sleep(2)

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'bring - stop_lite6_gripper'):
            return

        time.sleep(2)

        if not self.safe_move([199.8, 1.8, 199.0, 180.0, 0.0, 0.2], 'bring - regreso a home'):
            return
            
    
    # SUBRUTINA bring bandage
    def bring_bandage(self, objeto):
        
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina")
            return
        
        self.get_logger().info(f'Iniciando bring de {objeto}')
        if not self.return_subrutina:
            return

        if objeto not in self.coordenadas:
            self.get_logger().error(f'Coordenadas no definidas para: {objeto}')
            return
        
        # Mover brazo a la posición de inicio
        if self.modo_exclusivo: return
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.set_collision_sensitivity(1)
        if not self._check_code(code, 'set_collision_sensitivity'):
            return
        self._tcp_speed = 85
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-33.2, -211.7, 205.2, -179.8, 0.2, -26.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-194.3, -262.2, 165.1, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        
        if self.modo_exclusivo: return
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        self._tcp_speed = 5
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-194.3, -262.2, 147.3, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        
        if self.modo_exclusivo: return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        
        if self.modo_exclusivo: return
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-194.3, -262.2, 126.2, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)
        self._tcp_speed = 65
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-196.5, -257.5, 126.2, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[-196.5, -257.5, 165.1, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[124.6, -301.5, 252.3, -179.8, 0.1, -29.8], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.modo_exclusivo: return
        code = self._arm.set_position(*[293.7, 297.5, 199.2, -179.9, -0.1, 43.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(2)
        
        if self.modo_exclusivo: return
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.modo_exclusivo: return
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.modo_exclusivo: return
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
     
    #SUBRUTINA bind up small
    def bind_small(self, objeto):
        
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina bring.")
            return
        
        self.get_logger().info(f'Iniciando bind de {objeto}')
        if not self.return_subrutina:
            return

        if objeto not in self.coordenadas:
            self.get_logger().error(f'Coordenadas no definidas para: {objeto}')
            return
       
        code = self._arm.set_collision_sensitivity(2)
        if not self._check_code(code, 'set_collision_sensitivity'):
            return
        self._tcp_speed = 85
        code = self._arm.set_position(*[-33.2, -211.7, 205.2, -179.8, 0.2, -26.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[-195.4, -259.5, 165.1, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        self._tcp_speed = 10
        code = self._arm.set_position(*[-197.5, -260.5, 147.3, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.set_position(*[-197.6, -260.5, 126.2, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        self._tcp_speed = 65
        code = self._arm.set_position(*[-197.6, -260.5, 126.2, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[-197.6, -260.5, 174.5, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        for i in range(int(2)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.9, 311.3, 47.8, -88.1, 132.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.8, 79.0, 88.8, -11.0, 90.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 86.1, 126.7, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 44.1, 142.1, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, -7.6, 79.0, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(2)
        code = self._arm.set_position(*[405.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[185.7, 287.5, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[132.8, 252.4, 90.4, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[248.3, 113.0, 165.5, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[-33.6, 64.5, 41.3, 144.3, 110.8, -103.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[333.5, -65.4, 410.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[380.0, 250.0, 311.3, 47.8, -88.1, 132.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[333.5, 339.3, 79.0, 88.8, -11.0, 90.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[288.4, 86.1, 143.8, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[288.4, 23.4, 170.6, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[337.2, -48.1, 131.4, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[332.3, -7.6, 326.1, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        for i in range(int(1)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.9, 311.3, 47.8, -88.1, 132.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.8, 79.0, 88.8, -11.0, 90.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 86.1, 126.7, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 44.1, 142.1, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, -7.6, 79.0, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[167.6, 265.5, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[132.8, 252.4, 90.4, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[329.2, 68.9, 165.5, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[-33.6, 64.5, 41.3, 144.3, 110.8, -103.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[333.5, -65.4, 410.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[380.0, 250.0, 311.3, 47.8, -88.1, 132.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[333.5, 339.3, 79.0, 88.8, -11.0, 90.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[288.4, 86.1, 143.8, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[288.4, 23.4, 170.6, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[380.0, -7.6, 79.0, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[380.0, -7.6, 326.1, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        for i in range(int(1)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.9, 311.3, 47.8, -88.1, 132.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, 296.8, 79.0, 88.8, -11.0, 90.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 86.1, 126.7, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[314.8, 44.1, 142.1, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[380.0, -7.6, 79.0, -10.7, -88.8, -169.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_position(*[380.0, -7.6, 311.3, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
    
    
        
    #SUBRUTINA bind up big
    def bind_big(self, objeto):
        
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina bring.")
            return
        
        self.get_logger().info(f'Iniciando bind de {objeto}')
        if not self.return_subrutina:
            return

        if objeto not in self.coordenadas:
            self.get_logger().error(f'Coordenadas no definidas para: {objeto}')
            return
        
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_collision_sensitivity(1)
        if not self._check_code(code, 'set_collision_sensitivity'):
            return
        self._tcp_speed = 85
        code = self._arm.set_position(*[-33.2, -211.7, 205.2, -179.8, 0.2, -26.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[-194.3, -262.2, 165.1, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        self._tcp_speed = 5
        code = self._arm.set_position(*[-194.3, -262.2, 147.3, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(2)
        code = self._arm.set_position(*[-194.3, -262.2, 126.2, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)
        self._tcp_speed = 65
        code = self._arm.set_position(*[-196.5, -257.5, 126.2, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[-196.5, -257.5, 165.1, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[-1.8, -2.8, 43.3, 178.0, 45.5, -88.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        self._tcp_speed = 85
        time.sleep(10)
        for i in range(int(2)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[360.0, 140.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, 296.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[359.8, 296.2, 210.0, -90.3, -1.9, -91.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_servo_angle(angle=[56.6, 40.1, 44.2, 238.1, 87.6, -86.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_position(*[273.4, 43.4, 175.4, -90.3, 0.4, -91.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 210.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 350.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_servo_angle(angle=[48.3, -87.6, 19.8, 191.2, -18.3, -111.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[226.8, 209.5, 187.8, -116.0, 60.7, -70.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[1.5, 16.3, 22.7, 180.3, 71.7, -122.2], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[-3.1, -37.3, 19.2, 175.0, 35.3, -85.2], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(1)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[360.0, 140.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, 296.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[359.8, 296.2, 210.0, -90.3, -1.9, -91.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[273.4, 43.4, 175.4, -90.3, 0.4, -91.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 210.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 350.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_servo_angle(angle=[48.3, -87.6, 19.8, 191.2, -18.3, -111.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[226.8, 209.5, 187.8, -116.0, 60.7, -70.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[8.9, 12.1, 3.1, 188.8, 97.4, -137.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[250.0, 13.8, 386.3, -92.9, -48.6, -87.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        for i in range(int(1)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_position(*[360.0, 140.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, 296.0, 349.7, 32.5, -88.6, 147.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[359.8, 296.2, 210.0, -90.3, -1.9, -91.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_servo_angle(angle=[56.6, 40.1, 44.2, 238.1, 87.6, -86.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_position(*[273.4, 43.4, 175.4, -90.3, 0.4, -91.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 210.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[360.0, -7.6, 350.0, -88.8, 0.4, -90.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        
    #Subrutina Take
    def take(self, objeto):
        
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina bring.")
            return
        
        self.get_logger().info(f'Iniciando take de {objeto}')
        if not self.return_subrutina:
            return

        if objeto not in self.coordenadas:
            self.get_logger().error(f'Coordenadas no definidas para: {objeto}')
            return
        
        coords_dejar = self.coordenadas[objeto]['dejar']
        #take bandag real
    
        code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[360.9, 217.0, 199.0, 180.0, 0.0, 39.4], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(5)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[199.8, -173.0, 339.6, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_position(*[-257.8, -36.1, 338.9, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        #Bandage to bandage-holder
        
        if objeto == "bandage": 
            self._tcp_speed = 30
            code = self._arm.set_position(*[-383.0, -31.7, 250.0, -179.7, -0.3, 0.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-383.0, -31.7, 158.8, -179.7, -0.3, 0.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            time.sleep(1)
            code = self._arm.set_servo_angle(angle=[-176.1, 66.9, 132.1, 0.0, 65.3, -176.3], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            time.sleep(1)
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-421.9, -31.5, 193.6, -179.7, -0.3, 0.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            time.sleep(1)
            self._tcp_speed = 60
            code = self._arm.set_position(*[-257.8, -36.1, 184.6, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[-87.6, -221.3, 263.3, 179.6, 0.1, -0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            
        else: 
            code = self._arm.set_position(*[-257.8, -36.1, 184.6, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*coords_dejar, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(3)
            code = self._arm.set_position(*[-257.8, -36.1, 184.6, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            time.sleep(3)
            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            code = self._arm.set_position(*[-87.6, -221.3, 263.3, 179.6, 0.1, -0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return
         
    #cut     
    
    def cut(self):
        
        if self.modo_exclusivo:
            self.get_logger().warn("🛑 Modo exclusivo activo. Cancelando rutina bring.")
            return
        
        self.get_logger().info(f'Iniciando cut')
        if not self.return_subrutina:
            return
        
        # Mover brazo a la posición de inicio
        code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        #abre gripper
        code = self._arm.open_lite6_gripper() 
        if not self._check_code(code, 'bring - open_lite6_gripper'):
            return
        #posición intermedia
        code = self._arm.set_position(*[57.8, -189.1, 199.0, 180.0, 0, -92.7], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        #arriba": [-336.6, -268.5, 188.1, 179.7, -0.2, -179.2],
        #abajo": [-336.6, -268.5, 16.7, 179.7, -0.2, -179.2]
        #mover brazo arriba del objeto
        code = self._arm.set_position(*[-336.6, -268.5, 188.1, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#arriba
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
        
        # Bajar brazo hasta el objeto
        code = self._arm.set_position(*[-336.6, -268.5, 16.7, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
       
        # Cerrar el gripper para agarrar el objeto
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'bring - close_lite6_gripper'):
            return
        time.sleep(0.5)

        # Subir brazo con el objeto
        code = self._arm.set_position(*[-336.6, -268.5, 188.1, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
        # Mover brazo 
        code = self._arm.set_position(*[-336.6, -176.9, 225.9, 179.7, 0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
            
        code = self._arm.set_position(*[38.1, -225.9, 255.4, 178, 1.3, -91.4], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        
        code = self._arm.set_position(*[228.3, 1.3, 287.7, 179.8, 0.0, 0.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
                                
        # Start the camera
        
        inicio, fin = coordinates_detection()
        
        time.sleep(4)
        
        # if not inicio or not fin:
        #     self.pprint("Not valid coordinates detected")
        #     return

        inicial_pose = [290.4, -6.5, 68.7] #[inicio[0],1.5, inicio[2], 180, 0, 0]
        final_pose = [380.3, -6.5, 76.9] #[fin[0], 1.5, fin[2], 180, 0, 0]


        # Move to the beginnig positon
        code = self._arm.set_position(*inicial_pose, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(2)

        # Move to the end position
        code = self._arm.set_position(*final_pose, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        #Move to home
        
        code = self._arm.set_position(*[199.8, 1.8, 199, 180, 0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        code = self._arm.set_servo_angle(angle=[-174.7, 9.9, 31.8, 0, 21.9, -175.7], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_position(*[-254.1, -34.7, 64.9, -179.7, -0.2, 1.8], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        code = self._arm.open_lite6_gripper() 
        if not self._check_code(code, 'bring - open_lite6_gripper'):
            return
        
        time.sleep(2)
        
        code = self._arm.stop_lite6_gripper() 
        if not self._check_code(code, 'bring - stop_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        


def main(args=None):
    import sys
    import threading

    arm = XArmAPI('192.168.1.177', baud_checkset=False)

    rclpy.init(args=args)
    node = Movimiento(arm)
    

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

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

    # Obtains the intrinsic of the color camera
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def preprocess_image(img_color):
        grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        return grey

    def detect_circle(grey):
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                    param1=10, param2=30, minRadius=15, maxRadius=20)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0][0]
        return None

    def detect_sqr(grey):
        edges = cv2.Canny(grey, 13, 13)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
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

    coords_start = None
    coords_end = None
    coords_saved = False
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
            img_grey = preprocess_image(img_color)
            img_display = img_color.copy()

            circle = detect_circle(img_grey)
            sqr, contour = detect_sqr(img_grey)

            if circle is not None:
                x, y, r = circle
                cv2.circle(img_display, (x, y), r, (0, 255, 0), 2)
                depth_m = depth_frame.get_distance(x, y)
                coords_circle_3D = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth_m)
                coords_circle_3D =[round(coord * 100, 1) for coord in coords_circle_3D]  # mm
                coords_circle_3D_robot = np.array(coords_circle_3D) + offset

                if coords_circle_3D_robot[2] <= 99.3:
                    thick_offset_start = [10.3,0,34]
                    coords_circle_tool_3D = np.array(coords_circle_3D_robot) - thick_offset_start
                
                elif 104.2 > coords_circle_3D_robot[2] > 99.3:
                    mid_offset_start = [10.3,0,46.2]
                    coords_circle_tool_3D = np.array(coords_circle_3D_robot) - mid_offset_start

                elif coords_circle_3D_robot[2] >= 104.2:
                    thin_offset_start = [10.3,0,57.8] 
                    coords_circle_tool_3D = np.array(coords_circle_3D_robot) - thin_offset_start

                text_circle = f"Start: {coords_circle_tool_3D[0]:.1f}, {coords_circle_tool_3D[1]:.1f}, {coords_circle_tool_3D[2]:.1f} mm"
                cv2.putText(img_display, text_circle, (x + 10, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                text_circle = f"Start: {coords_circle_3D_robot[0]:.1f}, {coords_circle_3D_robot[1]:.1f}, {coords_circle_3D_robot[2]:.1f} mm"
                cv2.putText(img_display, text_circle, (x + 10, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if sqr is not None:
                cx, cy = sqr
                cv2.drawContours(img_display, [contour], -1, (0, 0, 255), 2)
                depth_m = depth_frame.get_distance(cx, cy)
                coords_sqr_3D = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth_m)
                coords_sqr_3D = [round(coord * 100, 1) for coord in coords_sqr_3D]  # mm
                coords_sqr_3D_robot = np.array(coords_sqr_3D) + offset

                if coords_sqr_3D_robot[2] >= 91.9:
                    thick_offset_end = [20, 0, 23]
                    coords_square_tool_3D = np.array(coords_sqr_3D_robot) - thick_offset_end
                elif 99.8> coords_sqr_3D_robot > 91.9:
                    mid_offset_end = [20, 0, 29]
                    coords_square_tool_3D = np.array(coords_sqr_3D_robot) - mid_offset_end
                elif coords_sqr_3D_robot >= 99.8: 
                    thin_offset_end = [20, 0, 35]
                    coords_square_tool_3D = np.array(coords_sqr_3D_robot) - thin_offset_end

                text_sqr = f"End: {coords_sqr_3D_robot[0]:.1f}, {coords_sqr_3D_robot[1]:.1f}, {coords_sqr_3D_robot[2]:.1f} mm"
                cv2.putText(img_display, text_sqr, (cx + 10, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if circle is not None and sqr is not None and not coords_saved:
                # coords_start = [(364 - coords_circle_tool_3D[0]), 1.5, coords_circle_tool_3D[2]] 
                # coords_end = [(458 - coords_square_tool_3D[0]), 1.5, coords_square_tool_3D[2]]
                coords_start = [290.4, -6.5, 68.7] 
                coords_end = [380.3, -6.5, 76.9]
                
                #if coords_start[2] >= 42 and coords_start[2] <= 60 and coords_end[2]>=40 and coords_end[2] <=84: 
                coords_saved = True
                print("Coords saved:")
                print(f"Start: {coords_start}")
                print(f"End: {coords_end}")

            if circle is not None and sqr is not None:
                cv2.line(img_display, (x, y), (cx, cy), (255, 0, 0), 2)

            cv2.imshow("Coordinates Detection", img_display)
            if cv2.waitKey(2) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        return coords_start, coords_end

class Movimiento(Node):
    def __init__(self, arm):  
        super().__init__('movement')  
        
        # Subscription to topic
        self.subscription = self.create_subscription(String,'voice_command',self.callback_voz,10)
       
        #State variables
        self.alive = True
        self._arm = arm
        self._ignore_exit_state = False
        self.emergency_mode = False
        self.key_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.key_thread.start()
        self.exclusive_mode = False


        # Speed and Acceleration
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500

        self._vars = {}
        self._funcs = {}
        #self.saludo_inicial()
        
        # Coordinates dictionaries
        self.coordinates = {
            "scalpel": {"up": [-336.6, -268.5, 188.1, 179.7, -0.2, -179.2],
                        "down": [-336.6, -268.5, 16.7, 179.7, -0.2, -179.2],
                        "leave": [-257.8, -36.1, 40, 180, 0, 0.2]
                        },
            
            "tweezers": {"up": [-351.1, -132.2, 188.1, 179.7, -0.2, -179.2],
                         "down": [-351.1, -132.2, 26.5, 179.7, -0.2, -179.2],
                         "leave": [-257.8, -36.1, 40, 180, 0, 0.2]
                         },
            
            "bandage": {"up": [-195.4,-255.9,165.1,179.7,0.2,-164.1],
                        "up2": [-196.5,-257.5,165.1,-179.9,0.3,-89.6],
                        "down": [-196.5,-257.5,126.2,179.7,0.2,-164.1],
                        "down2": [-196.5,-257.5,126.2,-179.9,0.3,-89.6],
                        "mid": [-196.5,-261,147.3,179.7,0.2,-164.1],
                        "leave": [-257.8, -36.1, 160, 180, 0, 0.2]},
            
            "small": {},
            "big": {},
            
            
        }
        # Flag to block in case of emergency
        self.return_subroutine = True
        # Start the robot (clean)
        self._robot_init() #change for arm if error apears
        self.get_logger().warn('Movement node connected')
    
    def _keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == 'q':
                    self.get_logger().warn("'q' pressed: Executing STOP")
                    
                    self._arm.set_pause_time(0)
                    self._arm.set_state(4)  # STOP

                    self.exclusive_mode = True

                    threading.Thread(target=self.spetial_stop, daemon=True).start()

                elif ch == 'w':
                    self.get_logger().info("'w' pressed: quiting exclusive mode")

                    self.exclusive_mode = False
                    self._arm.set_mode(0)
                    self._arm.set_state(0)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def safe_move(self, coords, label):
        if self.exclusive_mode:
            self.get_logger().warn(f"{label} exclusive mode ON")
            return False

        code = self._arm.set_position(*coords, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if code != 0:
            self.get_logger().error(f"Error in {label}: code {code}")
            return False

        while self._arm.get_is_moving():
            if self.exclusive_mode:
                self.get_logger().warn(f"Cancelling '{label}' emergency movement")
                self._arm.set_pause_time(0)
                self._arm.set_state(4)
                return False
            time.sleep(0.05)

        return True

    def spetial_stop(self):
        self.get_logger().info("Executing Spetial Stop")

        # Entering exclusive mode
        self._arm.set_pause_time(0)
        self._arm.set_state(4)
        self.exclusive_mode = True
        time.sleep(1)

        # Restart arm
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.5)

        # Step 1
        self.get_logger().info("Step 1")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-129.2, -1.3, 88.9, 0.0, 90.2, 23.9], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 1'): return

        # Step 2
        self.get_logger().info("Step 2")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-173.1, 6.9, 86.6, 0.0, 79.7, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 2'): return

        # Step 3
        self.get_logger().info("Step 3")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.8, 13.0, 56.3, 0.5, 43.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 3'): return

        # Step 4
        self.get_logger().info("Step 4")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.6, 42.1, 53.6, 0.9, 11.3, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 4'): return

        # Pause
        time.sleep(2)

        # Step 5: open gripper
        self.get_logger().info("Step 5: open gripper")
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'Step 5 - open gripper'): return
        time.sleep(1.5)

        # Step 6
        self.get_logger().info("Step 6")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-172.8, 13.0, 56.3, 0.5, 43.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 6'): return
        time.sleep(1.5)

        # Step 7: close gripper
        self.get_logger().info("Step 7: close gripper")
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'Step 7 - close gripper'): return
        time.sleep(1)

        # Step 8: stop gripper
        self.get_logger().info("Step 8: stop gripper")
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'Step 8 - stop gripper'): return
        time.sleep(1)

        # Step 9:move to final position 1
        self.get_logger().info("Step 9")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[-112.9, -0.8, 60.7, 0.4, 60.8, -112.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 9'): return

        # Step 10: final posotion 2
        self.get_logger().info("Step 10")
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)
        code = self._arm.set_servo_angle(angle=[0.0, -6.9, 38.9, 0.2, 45.9, -0.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'Step 10'): return

        # Final
        self.get_logger().info("Spetial stop executed")
    
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
            self.pprint(f'Error detected: err={data["error_code"]}, quit')
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('State 4 detected, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint(f'Count: {data["count"]}')

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
        instruction = msg.data.strip()
        self.get_logger().info(f'Instruction received: {instruction}')

        parts = instruction.split(";")
        if len(parts) < 2:
            self.get_logger().warn('Instruction incomplete.')
            return

        action = parts[0]
        object = parts[1]

        if action == 'bring':
            #self.bring(object)
            speach = f"Bringing the {object}, please receive it in the designated area."
            self.get_logger().info("🔊 Playing message with gTTS and mpg123...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=speach, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error playing message: {e}")
        
            if object == "bandage":
                self.bring_bandage(object)
            else:
                self.bring(object)
         
             
        elif action == 'take':
            #self.take(object)
            speach = f"I will take the {object} to the area for contaminated tools, please deliver it in the designated area and restock when possible."
            self.get_logger().info("🔊 Playing message with gTTS and mpg123...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=speach, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error playing message: {e}")
            self.take(object)
                
        elif action == 'cut':
            #self.cut(object)
            speach = "Please, make sure the arm is on the designated area, and has the start and ending points marked.  The robot will start the cut in a phew seconds,  press stop in case of emergency"
            self.get_logger().info("🔊 Playing message...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=speach, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error playing message: {e}")
            self.cut()
            
        elif action == 'bind up':
            #self.bind_up(object)
            speach = "For this action please place your arm on the holder, and make sure your wrist is aligning with the bandage.   Don't move your hand and keep it open with the palm towards the ceiling, your fingers must be fully extended.  The robot will start shortly."
            self.get_logger().info("🔊 Playing message...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=speach, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error playing message: {e}")
            if object == "small":
                self.bind_small(object)
            elif object == "big":
                self.bind_big(object)    
            
        else:
            self.get_logger().warn(f'Unrecognized action: {action}')
            speach = "Unknown command"
            self.get_logger().info("🔊 Playing message...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=speach, lang='en').save(f.name)
                    os.system(f"mpg123 {f.name}")
            except Exception as e:
                self.get_logger().error(f"❌ Error playing message: {e}")    
                    
    def bring(self, object):
        if self.exclusive_mode:
            self.get_logger().warn("🛑 Exclusive mode active. Canceling routine")
            return

        self.get_logger().info(f'Initiating bring de {object}')
        if not self.return_subroutine:
            return

        if object not in self.coordinates:
            self.get_logger().error(f'Coordinates not defined for: {object}')
            return

        coords_up = self.coordinates[object]['up']
        coords_down = self.coordinates[object]['down']

        if not self.safe_move([199.8, 1.8, 199.0, 180.0, 0.0, 0.2], 'bring - home'):
            return

        if self.exclusive_mode:
            return

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'bring - open_lite6_gripper'):
            return

        if not self.safe_move([57.8, -189.1, 199.0, 180.0, 0, -92.7], 'bring - mid'):
            return

        if not self.safe_move(coords_up, 'bring - coords_up'):
            return

        time.sleep(0.5)

        if not self.safe_move(coords_down, 'bring - coords_down'):
            return

        time.sleep(2)

        if self.exclusive_mode:
            return

        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'bring - close_lite6_gripper'):
            return

        time.sleep(2)

        if not self.safe_move(coords_up, 'bring - lift up object'):
            return

        time.sleep(0.5)

        if not self.safe_move([-336.6, -176.9, 225.9, 179.7, 0.2, -179.2], 'bring - mov1'):
            return

        if not self.safe_move([1.6, -321.3, 192.6, 179.7, -0.2, -78.1], 'bring - mov2'):
            return

        if not self.safe_move([360, 217, 199, -179.7, -0.2, 20.4], 'bring - deliver'):
            return

        time.sleep(2)

        if self.exclusive_mode:
            return

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'bring - open_lite6_gripper final'):
            return

        time.sleep(2)

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'bring - stop_lite6_gripper'):
            return

        time.sleep(2)

        if not self.safe_move([199.8, 1.8, 199.0, 180.0, 0.0, 0.2], 'bring - return to home'):
            return
            
    #SUBROUTINE bring bandage
    def bring_bandage(self, object):
        
        if self.exclusive_mode:
            self.get_logger().warn("Exclusive mode active. Canceling routine")
            return
        
        self.get_logger().info(f'Initiating bring of {object}')
        if not self.return_subroutine:
            return

        if object not in self.coordinates:
            self.get_logger().error(f'Coordinates not defined for: {object}')
            return
        
        # Move arm to start position
        if self.exclusive_mode: return
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.set_collision_sensitivity(1)
        if not self._check_code(code, 'set_collision_sensitivity'):
            return
        self._tcp_speed = 85
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-33.2, -211.7, 205.2, -179.8, 0.2, -26.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-194.3, -262.2, 165.1, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        
        if self.exclusive_mode: return
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        self._tcp_speed = 5
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-194.3, -262.2, 147.3, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        
        if self.exclusive_mode: return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        
        if self.exclusive_mode: return
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-194.3, -262.2, 126.2, 179.7, 0.2, -164.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)
        self._tcp_speed = 65
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-196.5, -257.5, 126.2, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[-196.5, -257.5, 165.1, -179.9, 0.3, -89.6], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[124.6, -301.5, 252.3, -179.8, 0.1, -29.8], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        
        if self.exclusive_mode: return
        code = self._arm.set_position(*[293.7, 297.5, 199.2, -179.9, -0.1, 43.5], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(2)
        
        if self.exclusive_mode: return
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.exclusive_mode: return
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(2)
        
        if self.exclusive_mode: return
        code = self._arm.set_servo_angle(angle=[-1.2, -5.8, 34.4, -0.4, 40.4, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
     
    #SUBROUTINE bind up small
    def bind_small(self, object):
        
        if self.exclusive_mode:
            self.get_logger().warn("Exclusive mode active. Canceling routine.")
            return
        
        self.get_logger().info(f'Starting bind up for {object}')
        if not self.return_subroutine:
            return

        if object not in self.coordinates:
            self.get_logger().error(f'Coordinates not defined for: {object}')
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
     
    #SUBROUTINE bind up big
    def bind_big(self, object):
        
        if self.exclusive_mode:
            self.get_logger().warn("Exclusive mode active. Canceling routine.")
            return
        
        self.get_logger().info(f'Initiating bind for {object}')
        if not self.return_subroutine:
            return

        if object not in self.coordinates:
            self.get_logger().error(f'Coordinates not defined for: {object}')
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
        
    #SUBROUTINE Take
    def take(self, object):
        
        if self.exclusive_mode:
            self.get_logger().warn("Exclusive mode active. Canceling routine.")
            return
        
        self.get_logger().info(f'Initiating take of {object}')
        if not self.return_subroutine:
            return

        if object not in self.coordinates:
            self.get_logger().error(f'Coordinates not defined for: {object}')
            return
        
        coords_leave = self.coordinates[object]['leave']
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
        
        if object == "bandage": 
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
            code = self._arm.set_position(*coords_leave, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
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
         
    #SUBROUTINE Cut     
    def cut(self):
        
        if self.exclusive_mode:
            self.get_logger().warn("Exclusive mode active. Canceling routine.")
            return
        
        self.get_logger().info(f'Initiating cut')
        if not self.return_subroutine:
            return
        code = self._arm.set_position(*[199.8, 1.8, 199.0, 180.0, 0.0, 0.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        code = self._arm.open_lite6_gripper() 
        if not self._check_code(code, 'bring - open_lite6_gripper'):
            return
        code = self._arm.set_position(*[57.8, -189.1, 199.0, 180.0, 0, -92.7], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        #up": [-336.6, -268.5, 188.1, 179.7, -0.2, -179.2],
        #down": [-336.6, -268.5, 16.7, 179.7, -0.2, -179.2]
        #move arm above the object
        code = self._arm.set_position(*[-336.6, -268.5, 188.1, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#up
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
        code = self._arm.set_position(*[-336.6, -268.5, 16.7, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'bring - close_lite6_gripper'):
            return
        time.sleep(0.5)
        code = self._arm.set_position(*[-336.6, -268.5, 188.1, 179.7, -0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'bring - set_position'):
            return
        time.sleep(0.5)
        code = self._arm.set_position(*[-336.6, -176.9, 225.9, 179.7, 0.2, -179.2], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        code = self._arm.set_position(*[38.1, -225.9, 255.4, 178, 1.3, -91.4], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)#home
        if not self._check_code(code, 'set_position'):
                return
        
        #Move to scanning zone
        code = self._arm.set_position(*[228.3, 1.3, 287.7, 179.8, 0.0, 0.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
                                
        # Start the camera
        start, end = coordinates_detection()
        time.sleep(2)
        
        # if not start or not fin:
        #     self.pprint("Not valid coordinates detected")
        #     return

        inicial_pose = [290.4, -6.5, 68.7] #[start[0],1.5, start[2], 180, 0, 0]
        final_pose = [380.3, -6.5, 76.9] #[end[0], 1.5, end[2], 180, 0, 0]


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

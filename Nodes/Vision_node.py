import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from ultralytics import YOLO

class VisionNode(Node):
    def _init_(self):
        super()._init_('vision_node')

        # Cargar modelo
        self.model_path = '/home/diego/jutsi/src/sistema_robot/Dataset_final_2/runs/detect/train/weights/best.pt'
        self.model = YOLO(self.model_path)
        self.class_names = ['bandage', 'scalpel', 'tweezers']

        # Publicador
        self.retro_pub = self.create_publisher(String, 'retroalimentacion', 10)

        # Cámara
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir la cámara.")
            rclpy.shutdown()
            return

        cv2.namedWindow("Detección de Objetos", cv2.WINDOW_NORMAL)
        self.timer = self.create_timer(0.2, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No se pudo capturar un frame.")
            return

        results = self.model(frame)
        objetos_detectados = set()

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf < 0.75 or cls >= len(self.class_names):
                    continue

                name = self.class_names[cls]
                objetos_detectados.add(name)

                # Mostrar recuadro
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (255, 0, 0) if name == 'bandage' else (0, 255, 0) if name == 'tweezers' else (0, 0, 255)
                label = f"{name} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        #Detecta y publica objetos:
        msg = String()
        msg.data = ";".join(objetos_detectados)
        self.retro_pub.publish(msg)
        self.get_logger().info(f"Herramientas detectadas: {msg.data}")

        # Mostrar ventana
        cv2.imshow('Detección de Objetos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Tecla 'q' presionada. Cerrando nodo.")
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
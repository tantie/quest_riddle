import cv2
import mediapipe as mp
import numpy as np
import RPi.GPIO as GPIO

# Инициализация GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.setwarnings(False)

def overlay_image(bg, img, x, y):

    #Накладывает одно изображение на другое в заданной позиции

    y1, y2 = max(0, y), min(bg.shape[0], y + img.shape[0])
    x1, x2 = max(0, x), min(bg.shape[1], x + img.shape[1])

    img_area = img[max(0, -y):y2 - y, max(0, -x):x2 - x]
    bg_area = bg[y1:y2, x1:x2]

    if img_area.shape[2] == 4:
        alpha = img_area[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        bg[y1:y2, x1:x2] = alpha * img_area[:, :, :3] + (1 - alpha) * bg_area
    else:
        bg[y1:y2, x1:x2] = img_area
    return bg

def is_point_inside_circle(point, center, radius):

    #Проверяет, находится ли точка внутри заданного круга

    return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 <= radius ** 2

def draw_circle_on_image(image, center, radius, color, thickness):

    #Рисует круг на изображении

    if show_circle:
        cv2.circle(image, center, radius, color, thickness)

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 25)
black_image = np.zeros((800, 1280, 3), dtype=np.uint8)


# Определение начального положения центра круга
initial_x_offset = 695  # Смещение по горизонтали
initial_y_offset = 488  # Смещение по вертикали
circle_center = (initial_x_offset, initial_y_offset)
#circle_center = (black_image.shape[1] // 2, black_image.shape[0] // 2) //если в центре

# Определение радиуса и центра круглой области
circle_radius = 495  # Задайте желаемый радиус

show_circle = True  # Переменная для управления отображением круга
#show_circle = False  # Переменная для управления отображением круга

# Позиции квадратов
square_positions = [(550, 150), (750, 150), (550, 400), (750, 400), (650, 280)]
static_square_positions = [(270, 600), (295, 165), (620, 0), (945, 155), (973, 650)]

# Загрузка изображений
static_images = [cv2.imread(f'{i+1}.jpg') for i in range(5)]
moving_images = [cv2.imread(f'{(i+1)*11}.png', -1) for i in range(5)]

captured_square = None
win_check = [False] * len(square_positions)

# Параметр погрешности в пикселях
tolerance = 22

cv2.namedWindow("Hands", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            break

        black_image.fill(0)

        # Отображение статичных изображений
        for i, static_square_position in enumerate(static_square_positions):
            black_image = overlay_image(black_image, static_images[i], static_square_position[0], static_square_position[1])

        # Отрисовка круга
        draw_circle_on_image(black_image, circle_center, circle_radius, (0, 255, 0), 2)

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_contours = np.array([(lmk.x * image.shape[1], lmk.y * image.shape[0])
                                          for lmk in hand_landmarks.landmark], dtype=np.int32)
                cv2.drawContours(black_image, [hand_contours], 0, (255, 255, 255), 1)

                thumb_tip = hand_contours[4]
                index_tip = hand_contours[8]
                dist = np.linalg.norm(thumb_tip - index_tip)

                if dist < 40:  # Расстояние между пальцами для "захвата" изображения
                    tip = index_tip
                    if captured_square is None:
                        for i, square_position in enumerate(square_positions):
                            square_image = moving_images[i]
                            if (tip[0] >= square_position[0] and tip[0] <= square_position[0] + square_image.shape[1] and
                                tip[1] >= square_position[1] and tip[1] <= square_position[1] + square_image.shape[0]):
                                captured_square = i
                                break
                    if captured_square is not None:
                        square_image = moving_images[captured_square]
                        new_x = tip[0] - square_image.shape[1] // 2
                        new_y = tip[1] - square_image.shape[0] // 2
                        square_corners = [
                            (new_x, new_y),
                            (new_x + square_image.shape[1], new_y),
                            (new_x, new_y + square_image.shape[0]),
                            (new_x + square_image.shape[1], new_y + square_image.shape[0])
                        ]

                        # Проверка, что все изображения находятся внутри круга
                        if all(is_point_inside_circle(corner, circle_center, circle_radius) for corner in square_corners):
                            square_positions[captured_square] = (new_x, new_y)
                else:
                    captured_square = None

        # Отображение движущихся изображений
        for i, square_position in enumerate(square_positions):
            black_image = overlay_image(black_image, moving_images[i], square_position[0], square_position[1])

        # Проверка соответствия позиций с учетом погрешности
        for i, square_position in enumerate(square_positions):
            for j, static_square_position in enumerate(static_square_positions):
                if (square_position[0] >= static_square_position[0] - tolerance and 
                    square_position[0] <= static_square_position[0] + static_images[j].shape[1] + tolerance and
                    square_position[1] >= static_square_position[1] - tolerance and 
                    square_position[1] <= static_square_position[1] + static_images[j].shape[0] + tolerance):
                    if i == j:
                        win_check[i] = True

        if all(win_check):
            #cv2.putText(black_image, 'WIN', (black_image.shape[1]//2, black_image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            GPIO.output(23, GPIO.HIGH) #high volt

        cv2.imshow('Hands', black_image)

        if cv2.waitKey(1) & 0xff == ord('c'):
            show_circle = not show_circle  # Переключение отображения круга

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    GPIO.output(23, GPIO.LOW) #low volt

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

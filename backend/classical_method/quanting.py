import cv2
import numpy as np

from preprocessing import prepare_img
from utils.show_image import show_image


def quanting(img_np, colours_cnt=12):
    print("Модуль quanting")
    print("-"*20)

    print("Точка в RGB:", img_np[0][0])
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)  # теперь img в цветовом пространстве LAB
    print("Точка в CIELAB:", img_lab[0][0])

    # cv2 преобразовала в формат lab не очень корректно, нужно отмасштабировать L
    L = img_lab[:,:,0] * (100/255.0)
    a = img_lab[:,:,1]
    b = img_lab[:,:,2]

    lab_true = np.stack([L, a, b], axis=2)  # lab, в котором всё супер

    # тут превращаю данные в список точек (L, a, b). k-means будет искать Евклидовы расстояния для них
    pixels = lab_true.reshape((-1, 3)).astype("float32")

    # условия остановки k-means - max 40 итераций или изменения центров кластеров меньше 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40,  1.0)
    attempts = 3  # k-means 3 раза выполнит кластеризацию с разными центрами и выберет лучшую

    # запускаем обучение. flags=cv2.KMEANS_PP_CENTERS - умный разброс начальных точек.
    # Он пытается выбрать далекие друг от друга стартовые центры.
    compactness, labels, centers = cv2.kmeans(data=pixels, K=colours_cnt, bestLabels=None, criteria=criteria,
                                              attempts=attempts, flags=cv2.KMEANS_PP_CENTERS)

    quant_lab = centers[labels.flatten()]      # заменяем каждый пиксель центром кластера
    quant_lab = quant_lab.reshape(lab_true.shape)  # Возвращаем изображение формата (H, W, 3)

    L = quant_lab[:,:,0] * (255.0/100.0)   # 0..255 (обратно вернули L)
    a = quant_lab[:,:,1]
    b = quant_lab[:,:,2]

    lab_for_cv = np.stack([L, a, b], axis=2).astype(np.uint8)
    quant_rgb = cv2.cvtColor(lab_for_cv, cv2.COLOR_LAB2RGB)

    show_image(quant_rgb, title=f"Изображение из {colours_cnt} цветов")


# Пример, как должен отработать код
img_np = prepare_img(img_path='test_images/wolf.webp') # результат работы первого модуля
quanting(img_np, colours_cnt=12)

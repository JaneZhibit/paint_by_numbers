'''
Цель данного этапа подготовить изображение к алгоритмам квантования и сегментации
Сопровождать код буду не комментариями, а выводом в консоль (мне кажется так удобнее)
'''
import numpy as np
from PIL import Image, ImageFilter
from utils.show_image import show_image


def prepare_img(img_path, target_max_side=1000):
    print('-' * 20)
    print("Модуль prepare_img")
    print('-' * 20)
    print("Шаг 1. Загрузка изображения", img_path)
    img = Image.open(img_path).convert("RGB")

    # show_image(img, "Исходное изображение")

    w, h = img.size
    print("    Ширина и высота изображения:", w, h)

    print("Шаг 2. Будем масштабировать изображение. Длину большей стороны хотим привести к", target_max_side)

    scale = target_max_side / max(w, h)
    print(f"    Изменить масштаб изображения нужно в {scale} раз")

    new_size = (int(w * scale), int(h * scale))
    print("    Новый размер", new_size)

    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print("Шаг 3. Мы изменили масштаб изображения. Теперь его нужно сгладить. Можно написать сглаживание самому, "
          "это не так сложно, но мне сейчас явно лень")

    # show_image(img, "Изменённый масштаб")

    """
      Про логику работы медианного фильтра можешь почитать,
      Если кратко, то он скользит по изображению квадратом со стороной size=3 (для нашей задачи 3 - оптимально)
      Смотрит на цвета пикселей (в данном случае 9-ти). Сортирует их по цвету (есть разные способы сортировки)
      Выбирается медианный цвет и средняя точка в квадрате им закрашивается (поэтому параметр - нечётное число)
      """
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # show_image(img, "С размытием")

    print("Шаг 4. Преобразование изображения в numpy список")
    print('-' * 20)

    return np.array(img)

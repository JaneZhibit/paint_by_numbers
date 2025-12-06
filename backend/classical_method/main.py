"""
Отсюда создание экземпляра класса и игра с ним
"""


from utils.show_image import show_image
from imgToPaintByNumbers import ClassicalPaintByNumbers

config = {
    'img_path': 'test_images/wolf.webp',
    'target_max_side': 1000,
    'canvas_width_mm': 400,
    'canvas_height_mm': 300,
    'min_diameter_mm': 3,
    'colours_cnt': 12,
    'logging': True
}

generator = ClassicalPaintByNumbers(config)
# сейчас вызываем модули отдельно, чтобы не забывать, откуда растут ноги
# потом всё будет выполнено внутри классаодним вызовом
generator.preprocessing()
#show_image(generator.preprocessed_img, title="Изображение с изменённым размером (масштаб уменьшен)")

generator.quantizing()
show_image(generator.quant_rgb, title="Изображение квантеризовано")

generator.postprocessing()
show_image(generator.postprocessed_img, title='Постобработка')

import sys
import ctypes
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# --- Твои физические константы ---
PPI = 243.0  # Плотность пикселей твоего Honor MagicBook X14
# 1 дюйм = 25.4 мм
PPM = PPI / 25.4  # Пикселей на миллиметр (~9.567 px/mm)

def enable_dpi_awareness():
    """Отключает масштабирование Windows (200%), чтобы работать с реальными пикселями матрицы."""
    if sys.platform == 'win32':
        try:
            # 2 = PROCESS_PER_MONITOR_DPI_AWARE
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as e:
            print(f"Не удалось установить DPI awareness: {e}")

def show_physical_scale(img_path=None, canvas_w_mm=400, canvas_h_mm=300):
    enable_dpi_awareness()
    
    # 1. Считаем, сколько ФИЗИЧЕСКИХ пикселей должно занимать изображение
    target_w_px = int(canvas_w_mm * PPM)
    target_h_px = int(canvas_h_mm * PPM)
    
    print(f"Размер холста: {canvas_w_mm}x{canvas_h_mm} мм")
    print(f"Физический размер на экране: {target_w_px}x{target_h_px} px")

    # Инициализация окна
    root = tk.Tk()
    root.title(f"Предпросмотр 1:1 (Холст: {canvas_w_mm}x{canvas_h_mm} мм)")
    
    # Выбор файла, если не передан
    if not img_path:
        img_path = filedialog.askopenfilename(
            title="Выбери изображение (например, из папки истории)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )
        if not img_path:
            return

    # 2. Загружаем и масштабируем изображение под физические пиксели
    try:
        img = Image.open(img_path)
        # Используем NEAREST, если смотрим карту контуров, чтобы края оставались резкими,
        # или LANCZOS для обычного изображения. Для картин по номерам NEAREST часто лучше.
        img = img.resize((target_w_px, target_h_px), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return

    # 3. Настройка UI с прокруткой (холст больше экрана ноутбука)
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(frame, bg="#333333", cursor="hand2")
    
    # Скроллбары
    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Помещаем картинку на Canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.config(scrollregion=(0, 0, target_w_px, target_h_px))

    # Раскрываем окно почти на весь экран
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{int(screen_width*0.9)}x{int(screen_height*0.9)}")
    root.state('zoomed') # Развернуть окно (Windows)

    # 4. Логика перетаскивания (Pan) мышью
    def move_from(event):
        canvas.scan_mark(event.x, event.y)
    
    def move_to(event):
        canvas.scan_dragto(event.x, event.y, gain=1)

    canvas.bind("<ButtonPress-1>", move_from)
    canvas.bind("<B1-Motion>", move_to)

    # Инструкция для пользователя
    print("\nИнструкция:")
    print("1. Зажми левую кнопку мыши, чтобы двигать холст (Pan).")
    print("2. Если приложить реальную линейку к экрану, размеры объектов должны совпасть!")
    
    root.mainloop()

if __name__ == "__main__":
    # Можно запустить скрипт напрямую, он попросит выбрать файл.
    # Размеры укажи те, которые стоят в конфиге (по умолчанию 400x300 мм)
    show_physical_scale(canvas_w_mm=600, canvas_h_mm=800)
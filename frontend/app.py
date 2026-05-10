import sys
import os
import tempfile
import traceback
from io import BytesIO, StringIO
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import numpy as np
import contextlib
import json
import cv2

# Добавляем путь к backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'classical_method'))

from pipeline import ClassicalPaintByNumbers, cache
from utils.comparison import save_stages_comparison

# Импортируем локальные модули
from presets import PRESETS, apply_preset, load_custom_presets, save_custom_preset, delete_custom_preset
from config_defaults import get_default_algo_config, UI_PARAMS_META, get_all_sections, get_params_by_section
from help_texts import HELP_TEXTS
from history_manager import HistoryManager

# Создаём глобальный экземпляр менеджера истории
history_manager = HistoryManager()


# ============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================================

st.set_page_config(page_title="Paint by Numbers", layout="wide", page_icon="🎨")
st.title("🎨 Paint by Numbers Generator")


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================================================

def init_session_state():
    """Инициализирует session_state дефолтными значениями."""
    if 'config' not in st.session_state:
        st.session_state['config'] = get_default_algo_config()
    
    if 'active_preset' not in st.session_state:
        st.session_state['active_preset'] = None
    
    # Инициализируем значения параметров из конфига
    for param in UI_PARAMS_META:
        param_path = param['param_path']
        if param_path not in st.session_state:
            # Получаем значение из конфига по dot-path или используем дефолт
            keys = param_path.split('.')
            value = st.session_state['config']
            try:
                for key in keys:
                    value = value[key]
            except (KeyError, TypeError):
                # Если ключ не найден, используем дефолтное значение из метаданных
                value = param.get('default')
            st.session_state[param_path] = value


init_session_state()


# ============================================================================
# ФУНКЦИИ ВСПОМОГАТЕЛЬНЫЕ
# ============================================================================

def set_value_by_path(config, path, value):
    """Устанавливает значение в словаре по dot-path."""
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_value_by_path(config, path):
    """Получает значение из словаря по dot-path."""
    keys = path.split('.')
    current = config
    for key in keys:
        current = current[key]
    return current


def build_config_from_session():
    """Собирает конфиг из session_state."""
    config = get_default_algo_config()
    
    for param in UI_PARAMS_META:
        param_path = param['param_path']
        if param_path in st.session_state:
            value = st.session_state[param_path]
            set_value_by_path(config, param_path, value)
    
    return config


def pil_to_bytes(pil_image):
    """Конвертирует PIL Image в PNG bytes. Возвращает None, если изображение None."""
    if pil_image is None:
        return None
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def apply_history_config_callback(history_config):
    """Callback для безопасного применения конфига из истории до рендера UI."""
    st.session_state['active_preset'] = "Из истории"
    
    algo_config = history_config.get('algorithm', {})
    
    # Восстанавливаем параметры алгоритма
    for param in UI_PARAMS_META:
        param_path = param['param_path']
        keys = param_path.split('.')
        val = algo_config
        try:
            for k in keys:
                val = val[k]
            st.session_state[param_path] = val
        except (KeyError, TypeError):
            pass
            
    # Восстанавливаем базовые параметры
    base_params = ['target_max_side', 'canvas_width_mm', 'canvas_height_mm', 'min_diameter_mm', 'colours_cnt']
    for base_key in base_params:
        if base_key in history_config:
            st.session_state[base_key] = history_config[base_key]


# ============================================================================
# SIDEBAR — ОСНОВНЫЕ ПАРАМЕТРЫ
# ============================================================================

st.sidebar.markdown("## ⚙️ Параметры изображения")

if 'target_max_side' not in st.session_state:
    st.session_state['target_max_side'] = 1000

target_max_side = st.sidebar.slider(
    "Макс. сторона (px)",
    min_value=200,
    max_value=2000,
    step=100,
    key='target_max_side'
)

if 'canvas_width_mm' not in st.session_state:
    st.session_state['canvas_width_mm'] = 600

canvas_width_mm = st.sidebar.slider(
    "Ширина холста (мм)",
    min_value=100,
    max_value=1200,
    step=50,
    key='canvas_width_mm'
)

if 'canvas_height_mm' not in st.session_state:
    st.session_state['canvas_height_mm'] = 300

canvas_height_mm = st.sidebar.slider(
    "Высота холста (мм)",
    min_value=100,
    max_value=1200,
    step=50,
    key='canvas_height_mm'
)

if 'min_diameter_mm' not in st.session_state:
    st.session_state['min_diameter_mm'] = 3.0

min_diameter_mm = st.sidebar.slider(
    "Мин. диаметр мазка (мм)",
    min_value=1.0,
    max_value=15.0,
    step=0.5,
    key='min_diameter_mm'
)

# Проверяем, включён ли auto_color
auto_color_enabled = st.session_state.get('quantizing.auto_color.enabled', False)

if auto_color_enabled:
    st.sidebar.info("ℹ️ Количество цветов определится автоматически (Elbow Method)")
    colours_cnt = 'auto'
else:
    if 'colours_cnt' not in st.session_state:
        st.session_state['colours_cnt'] = 16
    
    colours_cnt = st.sidebar.slider(
        "Количество цветов",
        min_value=4,
        max_value=40,
        step=1,
        key='colours_cnt'
    )

st.sidebar.markdown("---")


# ============================================================================
# SIDEBAR — ПРЕСЕТЫ
# ============================================================================

st.sidebar.markdown("### ⚡ Пресеты")

# Кнопка сброса пресетов
if st.sidebar.button("🛠 Свои настройки (Сброс)", use_container_width=True):
    st.session_state['active_preset'] = None
    st.session_state['config'] = get_default_algo_config()
    
    # Откатываем все параметры к дефолтам
    for param in UI_PARAMS_META:
        param_path = param['param_path']
        value = get_value_by_path(st.session_state['config'], param_path)
        st.session_state[param_path] = value
    
    st.rerun()

# Загружаем пользовательские пресеты
all_presets = {**PRESETS, **load_custom_presets()}

for preset_name in all_presets.keys():
    # Проверяем, является ли пресет пользовательским
    is_custom = preset_name not in PRESETS
    
    if is_custom:
        # Для пользовательских пресетов: две колонки (кнопка + удаление)
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            if st.button(preset_name, use_container_width=True, key=f"apply_{preset_name}"):
                st.session_state['active_preset'] = preset_name
                # Применяем пресет
                preset_config = apply_preset(preset_name, get_default_algo_config())
                st.session_state['config'] = preset_config
                
                # Обновляем session_state для всех параметров
                for param in UI_PARAMS_META:
                    param_path = param['param_path']
                    value = get_value_by_path(preset_config, param_path)
                    st.session_state[param_path] = value
                
                st.rerun()
        
        with col2:
            if st.button("❌", key=f"del_{preset_name}", help="Удалить пресет"):
                delete_custom_preset(preset_name)
                st.sidebar.success(f"✅ Пресет '{preset_name}' удалён!")
                st.rerun()
    else:
        # Для встроенных пресетов: кнопка на всю ширину
        if st.sidebar.button(preset_name, use_container_width=True, key=f"apply_{preset_name}"):
            st.session_state['active_preset'] = preset_name
            # Применяем пресет
            preset_config = apply_preset(preset_name, get_default_algo_config())
            st.session_state['config'] = preset_config
            
            # Обновляем session_state для всех параметров
            for param in UI_PARAMS_META:
                param_path = param['param_path']
                value = get_value_by_path(preset_config, param_path)
                st.session_state[param_path] = value
            
            st.rerun()

if st.session_state['active_preset']:
    st.sidebar.success(f"✅ Активный пресет: {st.session_state['active_preset']}")

st.sidebar.markdown("---")

# Форма сохранения пользовательского пресета
with st.sidebar.expander("💾 Сохранить свой пресет"):
    preset_name = st.text_input("Название пресета", key="custom_preset_name")
    if st.button("Сохранить", use_container_width=True):
        if preset_name.strip():
            # Собираем текущий конфиг
            current_config = build_config_from_session()
            save_custom_preset(preset_name, current_config)
            st.success(f"✅ Пресет '{preset_name}' сохранён!")
            st.rerun()
        else:
            st.error("❌ Введи название пресета")

st.sidebar.markdown("---")


# ============================================================================
# SIDEBAR — ПАРАМЕТРЫ АЛГОРИТМА
# ============================================================================

st.sidebar.markdown("## 🔧 Параметры алгоритма")

sections = get_all_sections()

for section in sections:
    params = get_params_by_section(section)
    
    with st.sidebar.expander(f"📁 {section}", expanded=False):
        for param in params:
            param_path = param['param_path']
            label = param['label']
            param_type = param['type']
            
            # Логика скрытия: проверяем зависимость напрямую из st.session_state
            is_visible = True
            if 'depends_on' in param:
                depends_on = param['depends_on']
                parent_val = st.session_state.get(depends_on['path'])
                
                if 'value' in depends_on and parent_val != depends_on['value']:
                    is_visible = False
                elif 'values' in depends_on and parent_val not in depends_on['values']:
                    is_visible = False
            
            if not is_visible:
                continue
                
            # Отрисовка виджета
            help_text = HELP_TEXTS.get(param['help_key'], "") if param['help_key'] else ""
            
            if param_type == 'bool':
                st.checkbox(label, key=param_path, help=help_text)
            elif param_type == 'int':
                st.slider(label, min_value=param['min'], max_value=param['max'], step=param['step'], key=param_path, help=help_text)
            elif param_type == 'float':
                st.slider(label, min_value=param['min'], max_value=param['max'], step=param['step'], key=param_path, help=help_text)
            elif param_type == 'select':
                st.selectbox(label, options=param['options'], key=param_path, help=help_text)

st.sidebar.markdown("---")
st.sidebar.markdown("## 💾 Управление кэшем")

if st.sidebar.button("🗑 Очистить кэш (Освободить место)", use_container_width=True):
    try:
        cache.clear()
        st.sidebar.success("✅ Кэш успешно очищен!")
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка при очистке кэша: {e}")


# ============================================================================
# ГЛАВНАЯ ОБЛАСТЬ — ВКЛАДКИ
# ============================================================================

tab_generate, tab_help, tab_guide, tab_history = st.tabs(["🖼️ Генерация", "📖 Справка", "📚 Руководство", "⏱️ История"])


# ============================================================================
# ВКЛАДКА "ГЕНЕРАЦИЯ"
# ============================================================================

with tab_generate:
    col_upload, col_preview = st.columns([1, 2])
    
    with col_upload:
        st.markdown("### 📤 Загрузка изображения")
        uploaded_file = st.file_uploader("Выбери фото", type=["jpg", "jpeg", "png", "webp", "bmp"])
        
        if uploaded_file:
            st.success("✅ Файл загружен")
    
    with col_preview:
        if uploaded_file:
            st.markdown("### 👁️ Исходник")
            image = Image.open(uploaded_file)
            aspect_ratio = (canvas_width_mm, canvas_height_mm)
            cropped_image = st_cropper(image, aspect_ratio=aspect_ratio, box_color='#FF0000', return_type='image')
            
            st.markdown("---")
            st.markdown("### ⚡ Быстрый предпросмотр фильтров")
            live_preview = st.toggle("Включить Live-превью (быстро, в низком разрешении)", value=False, help="Мгновенно показывает результат сглаживания и карту внимания при изменении параметров слева.")
            
            if live_preview:
                with st.spinner("Обновление превью..."):
                    # Сохраняем во временный файл
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        cropped_image.save(tmp_file, format="PNG")
                        tmp_path = tmp_file.name
                    
                    # Собираем конфиг, но принудительно ставим маленький размер для скорости
                    preview_config = build_config_from_session()
                    fast_config = {
                        'img_path': tmp_path,
                        'target_max_side': 500,  # Маленький размер для мгновенного рендера!
                        'logging': False,
                        'algorithm': preview_config
                    }
                    
                    try:
                        from core.preprocessing import preprocess_image
                        _, prep_img, sal_map = preprocess_image(fast_config)
                        
                        # Выводим в две колонки
                        prev_col1, prev_col2 = st.columns(2)
                        with prev_col1:
                            st.image(prep_img, caption="Сглаживание (500px)", use_container_width=True)
                        
                        with prev_col2:
                            if sal_map is not None:
                                smap_norm = cv2.normalize(sal_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                hm = cv2.applyColorMap(smap_norm, cv2.COLORMAP_JET)
                                st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), caption="Карта внимания", use_container_width=True)
                            else:
                                st.info("Карта внимания отключена")
                                
                    except Exception as e:
                        st.error(f"Ошибка превью: {str(e)}")
    
    st.markdown("---")
    
    # Управление процессом генерации
    st.markdown("### ⚙️ Управление процессом")
    execution_stage = st.radio(
        "Генерировать до этапа:",
        options=["Только сглаживание", "Сглаживание + Квантование", "Полный цикл (с постобработкой)", "Полный цикл + Контуры"],
        index=3,
        help="Останови генерацию раньше, чтобы быстро проверить настройки фильтров или палитры."
    )
    
    st.markdown("---")
    
    # Кнопка генерации
    if st.button("🚀 Сгенерировать", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("❌ Пожалуйста, загрузи изображение")
        else:
            try:
                with st.spinner("⏳ Генерирую..."):
                    # Сохраняем загруженный файл во временный файл
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        cropped_image.save(tmp_file, format="PNG")
                        tmp_path = tmp_file.name
                    
                    # Собираем конфиг из session_state
                    algo_config = build_config_from_session()
                    
                    # Собираем полный конфиг
                    config = {
                        'img_path': tmp_path,
                        'target_max_side': target_max_side,
                        'canvas_width_mm': canvas_width_mm,
                        'canvas_height_mm': canvas_height_mm,
                        'min_diameter_mm': min_diameter_mm,
                        'colours_cnt': colours_cnt,
                        'logging': True,
                        'algorithm': algo_config
                    }
                    
                    # Запускаем pipeline с перехватом логов
                    gen = ClassicalPaintByNumbers(config)
                    
                    # Перехватываем вывод (логи)
                    log_capture = StringIO()
                    with contextlib.redirect_stdout(log_capture):
                        # Выполняем этапы пошагово в зависимости от выбора
                        gen.preprocessing()
                        
                        if execution_stage in ["Сглаживание + Квантование", "Полный цикл (с постобработкой)", "Полный цикл + Контуры"]:
                            gen.quantizing()
                            
                        if execution_stage in ["Полный цикл (с постобработкой)", "Полный цикл + Контуры"]:
                            gen.postprocessing()
                        
                        if execution_stage == "Полный цикл + Контуры":
                            gen.vectorization()
                    
                    # Получаем перехваченные логи
                    logs = log_capture.getvalue()
                    
                    # Сохраняем результаты в session_state
                    st.session_state['generation_result'] = {
                        'original': gen.original_img,
                        'preprocessed': gen.preprocessed_img,
                        'saliency_map': gen.saliency_map,
                        'quantized': gen.quant_rgb,
                        'postprocessed': gen.postprocessed_img,
                        'final_colors': gen.final_colors,
                        'borders_mask': gen.borders_mask,
                        'timings': gen.timings if hasattr(gen, 'timings') else None,
                        'logs': logs,
                        'execution_stage': execution_stage,
                        'complexity_metrics': gen.complexity_metrics
                    }
                    
                    # Добавляем в историю обработок (только если есть результат постобработки)
                    if st.session_state['generation_result'].get('postprocessed') is not None:
                        history_manager.add_entry(
                            config,
                            st.session_state['generation_result']['postprocessed'],
                            st.session_state['generation_result'].get('timings'),
                            st.session_state.get('active_preset', 'Пользовательские')
                        )
                    
                    st.success("✅ Генерация завершена!")
                    
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                with st.expander("📋 Подробности ошибки"):
                    st.code(traceback.format_exc())
    
    # Показываем результаты, если они есть
    if 'generation_result' in st.session_state:
        result = st.session_state['generation_result']
        
        st.markdown("---")
        st.markdown("### 📊 Результаты обработки")
        
        # Четыре стадии в двух строках
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ Исходник")
            st.image(result['original'], use_container_width=True)
        
        with col2:
            st.markdown("#### 2️⃣ Сглаживание")
            st.image(result['preprocessed'], use_container_width=True)
            
            # Визуализация Saliency Map, если она есть
            if result.get('saliency_map') is not None:
                st.markdown("🔥 **Тепловая карта внимания**")
                # Автоматически растягиваем контраст: самый темный пиксель станет 0 (синим), самый светлый 255 (красным)
                smap_norm = cv2.normalize(result['saliency_map'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap = cv2.applyColorMap(smap_norm, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, use_container_width=True, caption="Красный - фокус, Синий - фон")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if result.get('quantized') is not None:
                st.markdown("#### 3️⃣ Квантование")
                st.image(result['quantized'], use_container_width=True)
            else:
                st.info("ℹ️ Этап квантования был пропущен.")
        
        with col4:
            if result.get('postprocessed') is not None:
                st.markdown("#### 4️⃣ Результат")
                st.image(result['postprocessed'], use_container_width=True)
            else:
                st.info("ℹ️ Этап постобработки был пропущен.")
        
        # Превью контуров (если этап векторизации был выполнен)
        if result.get('borders_mask') is not None:
            st.markdown("---")
            st.markdown("### ✏️ Контуры (векторизация)")
            # Инвертируем маску: чёрные линии на белом фоне
            inverted = 255 - result['borders_mask']
            st.image(inverted, use_container_width=True, caption="Чёрные линии — границы секторов")
        
        # Вывод таймингов
        st.markdown("---")
        st.markdown("### ⏱️ Время выполнения")
        
        if result.get('timings'):
            timings = result['timings']
            
            # Выводим время каждого этапа
            timing_cols = st.columns(len(timings))
            total_time = 0
            
            for i, (stage_name, stage_time) in enumerate(timings.items()):
                with timing_cols[i]:
                    st.metric(stage_name, f"{stage_time:.2f}s")
                    total_time += stage_time
            
            # Общее время
            st.metric("Общее время", f"{total_time:.2f}s")
        
        # Вывод метрик сложности
        if result.get('complexity_metrics'):
            metrics = result['complexity_metrics']
            st.markdown("---")
            st.subheader("📊 Анализ сложности закрашивания")
            
            # Общий индекс сложности
            st.metric("Индекс сложности", f"{metrics['complexity_score']:.1f}", help="Чем выше, тем сложнее. Зависит от количества мелких деталей, узких мест и изрезанности границ.")
            
            # Колонки для размеров пятен
            st.markdown("#### Размеры пятен")
            col_s, col_m, col_l = st.columns(3)
            with col_s:
                st.metric("Мелкие", f"{metrics['count_small']} шт", f"{metrics['pct_small']:.1f}% площади")
            with col_m:
                st.metric("Средние", f"{metrics['count_medium']} шт", f"{metrics['pct_medium']:.1f}% площади")
            with col_l:
                st.metric("Крупные", f"{metrics['count_large']} шт", f"{metrics['pct_large']:.1f}% площади")
                
            # Колонки для проблемных мест и формы
            st.markdown("#### Проблемные места и форма")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Непрокрашиваемые", f"{metrics['pct_unpaintable_max']:.1f}%", help="Площадь пятен, в которые кисть вообще не пролезет (max_dt < brush_radius)")
            with col_p2:
                st.metric("Узкие / Хвостатые", f"{metrics['pct_unpaintable_median']:.1f}%", help="Площадь пятен, где кисть не пролезет в большей части пятна (median_dt < brush_radius)")
            with col_p3:
                st.metric("Изрезанность", f"{metrics['border_density']:.3f}", help="Отношение длины всех границ к общей площади. Чем выше, тем более «рваные» края у пятен.")
        
        # Вывод логов
        if result.get('logs'):
            st.markdown("---")
            with st.expander("📝 Логи выполнения алгоритма"):
                st.code(result['logs'], language="text")
        
        # Кнопка скачивания
        st.markdown("---")
        if result.get('postprocessed') is not None:
            postprocessed_img = result['postprocessed']
            if isinstance(postprocessed_img, np.ndarray):
                postprocessed_img = Image.fromarray(postprocessed_img)
            result_png = pil_to_bytes(postprocessed_img)
            if result_png is not None:
                st.download_button(
                    label="💾 Скачать результат (PNG)",
                    data=result_png,
                    file_name="paint_by_numbers_result.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Результат постобработки недоступен. Выполните полный цикл генерации.")


# ============================================================================
# ВКЛАДКА "СПРАВКА"
# ============================================================================

with tab_help:
    st.markdown("## 📖 Справка по параметрам алгоритма")
    
    for key in sorted(HELP_TEXTS.keys()):
        with st.expander(f"ℹ️ {key}", expanded=False):
            st.markdown(HELP_TEXTS[key])


# ============================================================================
# ВКЛАДКА "РУКОВОДСТВО"
# ============================================================================

with tab_guide:
    st.markdown("## 📚 Глобальное руководство")
    
    # Пытаемся прочитать файл README_PARAMS.md
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README_PARAMS.md')
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        st.markdown(readme_content)
    except FileNotFoundError:
        st.warning(
            "⚠️ Файл README_PARAMS.md не найден. "
            "Убедитесь, что файл находится в корне проекта."
        )
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")


# ============================================================================
# ВКЛАДКА "ИСТОРИЯ"
# ============================================================================

with tab_history:
    st.markdown("## ⏱️ История обработок")
    
    history = history_manager.get_history()
    
    if not history:
        st.info("📭 История пуста. Обработайте изображение, чтобы добавить запись в историю.")
    else:
        st.markdown(f"**Всего записей:** {len(history)}")
        st.markdown("---")
        
        for idx, entry in enumerate(history, 1):
            with st.expander(
                f"#{idx} | {entry['timestamp']} | {entry['preset']}",
                expanded=False
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🖼️ Результат")
                    # Отображаем миниатюру результата из файла
                    try:
                        st.image(entry['img_path'], use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось загрузить миниатюру: {e}")
                
                with col2:
                    st.markdown("### 📋 Информация")
                    st.write(f"**Время:** {entry['timestamp']}")
                    st.write(f"**Пресет:** {entry['preset']}")
                    
                    if entry.get('timings'):
                        st.markdown("**Время выполнения:**")
                        timings_text = "\n".join(
                            [f"- {stage}: {time:.2f}s" for stage, time in entry['timings'].items()]
                        )
                        st.markdown(timings_text)
                    
                    st.markdown("---")
                    
                    # Кнопка для применения конфига из истории с callback
                    st.button(
                        "🔄 Применить эти настройки",
                        key=f"apply_{idx}",
                        on_click=apply_history_config_callback,
                        args=(entry['config'],),
                        use_container_width=True
                    )
                
                # Визуализация конфига
                with st.expander("⚙️ Посмотреть JSON конфигурации"):
                    st.json(entry['config'])

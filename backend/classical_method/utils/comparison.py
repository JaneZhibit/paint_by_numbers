import matplotlib.pyplot as plt
import os


def save_stages_comparison(generator, stages, title="Сравнение этапов", save_path="comparison.png"):
    """
    Собирает указанные стадии пайплайна в одно изображение и сохраняет.
    Доступные стадии: 'original', 'preprocessing', 'quantizing', 'postprocessing', 'contours'
    """
    stage_mapping = {
        'original': (generator.original_img, "Исходник"),
        'preprocessing': (generator.preprocessed_img, "Сглаживание"),
        'quantizing': (generator.quant_rgb, "Квантование"),
        'postprocessing': (generator.postprocessed_img, "Постобработка"),
        'contours': (generator.contours_img, "Контуры")
    }

    images_to_plot = []
    titles = []

    for stage in stages:
        if stage in stage_mapping and stage_mapping[stage][0] is not None:
            images_to_plot.append(stage_mapping[stage][0])
            titles.append(stage_mapping[stage][1])
        else:
            print(f"⚠️ Стадия '{stage}' недоступна или не сгенерирована.")

    n = len(images_to_plot)
    if n == 0:
        print("Нет изображений для отображения.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, img, t in zip(axes, images_to_plot, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=16)
        ax.axis('off')

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()

    # Создаем папку, если ее нет
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Сравнение сохранено: {save_path}")

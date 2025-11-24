import matplotlib.pyplot as plt


def show_image(img, title="Image", figsize=(6, 6), save_path=None):
    """
    Отображает изображение (PIL.Image или np.ndarray).
    """

    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title, fontsize=14)
    plt.axis('off')

    # если нужно, сохраняет
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Сохранено: {save_path}")

    plt.show()

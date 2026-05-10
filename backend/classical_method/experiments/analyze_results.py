import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results():
    history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'workspace', 'history'))
    results_file = os.path.join(history_dir, 'experiment_results.csv')
    plots_file = os.path.join(history_dir, 'experiment_plots.png')

    if not os.path.exists(results_file):
        print(f"Файл с результатами не найден: {results_file}")
        return

    df = pd.DataFrame(pd.read_csv(results_file))
    
    if df.empty:
        print("Файл с результатами пуст.")
        return

    # Усредняем данные по количеству цветов (по всем картинкам)
    mean_df = df.groupby('colours_cnt').mean(numeric_only=True).reset_index()
    std_df = df.groupby('colours_cnt').std(numeric_only=True).reset_index()

    # Настраиваем стиль графиков
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Влияние количества цветов на метрики сложности картины', fontsize=16)

    # График 1: Непрокрашиваемые области и узкие места
    ax1 = axes[0]
    ax1.plot(mean_df['colours_cnt'], mean_df['pct_unpaintable_max'], marker='o', label='Непрокрашиваемые (max, %)', color='red')
    ax1.fill_between(mean_df['colours_cnt'], 
                     mean_df['pct_unpaintable_max'] - std_df['pct_unpaintable_max'], 
                     mean_df['pct_unpaintable_max'] + std_df['pct_unpaintable_max'], 
                     color='red', alpha=0.2)
                     
    ax1.plot(mean_df['colours_cnt'], mean_df['pct_unpaintable_median'], marker='s', label='Узкие места (median, %)', color='orange')
    ax1.fill_between(mean_df['colours_cnt'], 
                     mean_df['pct_unpaintable_median'] - std_df['pct_unpaintable_median'], 
                     mean_df['pct_unpaintable_median'] + std_df['pct_unpaintable_median'], 
                     color='orange', alpha=0.2)
                     
    ax1.set_title('Проблемные области')
    ax1.set_xlabel('Количество цветов')
    ax1.set_ylabel('Площадь (%)')
    ax1.legend()
    ax1.grid(True)

    # График 2: Количество пятен по размерам
    ax2 = axes[1]
    ax2.plot(mean_df['colours_cnt'], mean_df['count_small'], marker='o', label='Мелкие', color='blue')
    ax2.plot(mean_df['colours_cnt'], mean_df['count_medium'], marker='s', label='Средние', color='green')
    ax2.plot(mean_df['colours_cnt'], mean_df['count_large'], marker='^', label='Крупные', color='purple')
    
    ax2.set_title('Количество пятен разных размеров')
    ax2.set_xlabel('Количество цветов')
    ax2.set_ylabel('Количество (шт)')
    ax2.legend()
    ax2.grid(True)

    # График 3: Общий индекс сложности и плотность границ
    ax3 = axes[2]
    color_score = 'tab:blue'
    ax3.set_xlabel('Количество цветов')
    ax3.set_ylabel('Индекс сложности', color=color_score)
    line1 = ax3.plot(mean_df['colours_cnt'], mean_df['complexity_score'], marker='o', color=color_score, label='Индекс сложности')
    ax3.tick_params(axis='y', labelcolor=color_score)
    ax3.grid(True)

    ax3_twin = ax3.twinx()  
    color_density = 'tab:red'
    ax3_twin.set_ylabel('Плотность границ', color=color_density)
    line2 = ax3_twin.plot(mean_df['colours_cnt'], mean_df['border_density'], marker='s', color=color_density, label='Плотность границ')
    ax3_twin.tick_params(axis='y', labelcolor=color_density)
    
    # Объединяем легенды для ax3 и ax3_twin
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.set_title('Общая сложность и изрезанность')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Сохраняем график
    plt.savefig(plots_file, dpi=300, bbox_inches='tight')
    print(f"Графики сохранены в: {plots_file}")
    
    # Опционально показываем
    plt.show()

if __name__ == "__main__":
    analyze_results()

"""Shared plotting helpers."""
import matplotlib.pyplot as plt


def show_table(df, title, fontsize=9):
    """Render a DataFrame as a matplotlib table with a dark header."""
    fig, ax = plt.subplots(figsize=(13, 0.45 * (len(df) + 2)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_facecolor('#2c2c2c')
            cell.set_text_props(color='white', weight='bold')
    ax.set_title(title)
    plt.tight_layout()
    return fig

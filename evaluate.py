import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('hotdog_detection/hotdog_model/weights/best.pt')  # Load your fine-tuned weights

metrics = model.val()

precision = metrics.box.P
recall = metrics.box.R
mAP50 = metrics.box.map50  # mAP at 0.5 IoU
mAP5095 = metrics.box.map  # mAP from 0.5 to 0.95 IoU

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP@0.5: {mAP50:.4f}")
print(f"mAP@0.5:0.95: {mAP5095:.4f}")

def generate_table(precision, recall, mAP50, mAP5095):

    data = [
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["mAP@0.5", f"{mAP50:.4f}"],
        ["mAP@0.5:0.95", f"{mAP5095:.4f}"]
    ]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 2))  # Adjust the size as needed

    # Hide the axes
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data, colLabels=["Metric", "Value"], cellLoc='center', loc='center')

    # Customize the table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    header_cells = table[0, 0], table[0, 1]  # Header cells are in the 0th row (Metric, Value)
    for cell in header_cells:
        cell.set_text_props(weight='bold')

    # Save the table as a PNG file
    plt.savefig('metrics_table.png', bbox_inches='tight', dpi=300)

    # Show the table
    plt.show()

generate_table(precision, recall, mAP50, mAP5095)
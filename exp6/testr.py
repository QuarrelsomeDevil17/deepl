import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 6))

# Set up the main flowchart boxes
main_flow = [
    "Start",
    "Gas Sensors Detect LPG/Propane",
    "Microcontroller Processes Data",
    "Is Gas Level Safe?",
    "Display Real-time Gas Levels",
    "Trigger Alert (Buzzer/LED/Notification)",
    "End"
]

# Positions for the main flowchart boxes
main_positions = [
    (0.5, 0.9),
    (0.5, 0.75),
    (0.5, 0.6),
    (0.5, 0.45),
    (0.5, 0.3),
    (0.5, 0.15),
    (0.5, 0.05)
]

# Draw the main flowchart boxes
for box, pos in zip(main_flow, main_positions):
    ax.text(pos[0], pos[1], box, 
            ha='center', va='center', 
            bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))

# Arrows between the boxes
arrowprops = dict(facecolor='black', arrowstyle='->')
for i in range(len(main_positions) - 1):
    ax.annotate('', xy=main_positions[i + 1], xytext=main_positions[i], arrowprops=arrowprops)

# Decision branch for "Is Gas Level Safe?"
ax.annotate('', xy=(0.8, 0.45), xytext=(0.5, 0.45), arrowprops=arrowprops)
ax.text(0.65, 0.475, "No", ha='center', va='center')
ax.text(0.8, 0.45, "Yes", ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.3'))

# Layout adjustments
ax.axis('off')
plt.title("Smart Gas Leakage Detector Bot - System Flowchart", fontsize=14)
plt.show()

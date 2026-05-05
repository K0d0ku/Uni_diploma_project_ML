# # includes test results and evalutaion
# import matplotlib
#
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import re
# import os
# import pandas as pd
# import numpy as np
#
# # limit
# # log_file_path = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\full holistics_training LIMIT.txt"
# # mode_suffix = "LIMIT"
#
# # no limit
# log_file_path = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\full holistics_training NO LIMIT.txt"
# mode_suffix = "NO_LIMIT"
#
# output_folder = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\graphs"
#
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# with open(log_file_path, 'r', encoding='utf-8') as f:
#     full_content = f.read()
#
# # 1. PARSE DATASET & MODEL INFO
# dataset_info = {
#     'Total Samples': re.search(r'X shape: \((\d+)', full_content),
#     'Classes': re.search(r'Number of classes: (\d+)', full_content),
#     'Train Size': re.search(r'Train set: \((\d+)', full_content),
#     'Val Size': re.search(r'Validation set: \((\d+)', full_content),
#     'Total Params': re.search(r'Total params: (\d+)', full_content)
# }
# # Extract values safely
# dataset_info = {k: (v.group(1) if v else "N/A") for k, v in dataset_info.items()}
#
# # 2. SPLIT PHASES
# phases = re.split(r'################', full_content)
# training_data = {}
# for phase in phases:
#     if "train lstm" in phase.lower():
#         training_data['Initial'] = phase
#     elif "continue training" in phase.lower():
#         training_data['Continuation'] = phase
#
#
# def extract_metrics(content):
#     epochs = re.findall(r'Epoch (\d+)/', content)
#     loss = re.findall(r' - loss: ([\d.]+)', content)
#     acc = re.findall(r' - accuracy: ([\d.]+)', content)
#     v_loss = re.findall(r' - val_loss: ([\d.]+)', content)
#     v_acc = re.findall(r' - val_accuracy: ([\d.]+)', content)
#
#     e_int = [int(x) for x in epochs]
#     l_flt = [float(x) for x in loss]
#     a_flt = [float(x) for x in acc]
#     vl_flt = [float(x) for x in v_loss]
#     va_flt = [float(x) for x in v_acc]
#
#     min_len = min(len(e_int), len(l_flt), len(a_flt), len(vl_flt), len(va_flt))
#     return e_int[:min_len], l_flt[:min_len], a_flt[:min_len], vl_flt[:min_len], va_flt[:min_len]
#
#
# # 3. PLOT TRAINING
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle(f'Training Performance: {mode_suffix}', fontsize=16)
#
# total_epochs_offset = 0
# for phase_name in ['Initial', 'Continuation']:
#     if phase_name in training_data:
#         e, l, a, vl, va = extract_metrics(training_data[phase_name])
#         if not e: continue
#         adj_epochs = [i + total_epochs_offset + 1 for i in range(len(e))]
#         ax1.plot(adj_epochs, a, label=f'{phase_name} Train', linewidth=2)
#         ax1.plot(adj_epochs, va, label=f'{phase_name} Val', linestyle='--', linewidth=2)
#         ax2.plot(adj_epochs, l, label=f'{phase_name} Train', linewidth=2)
#         ax2.plot(adj_epochs, vl, label=f'{phase_name} Val', linestyle='--', linewidth=2)
#         total_epochs_offset += len(e)
#
# ax1.set_title('Accuracy');
# ax1.legend();
# ax1.grid(True, alpha=0.3)
# ax2.set_title('Loss');
# ax2.legend();
# ax2.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, f"training_comparison_{mode_suffix}.png"))
#
# # 4. DATASET & MODEL SUMMARY GRAPH (New Table Image)
# fig_info, ax_info = plt.subplots(figsize=(6, 4))
# ax_info.axis('off')
# table_data = [[k, v] for k, v in dataset_info.items()]
# table = ax_info.table(cellText=table_data, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left')
# table.scale(1, 2)
# plt.title(f"Model & Dataset Summary ({mode_suffix})", pad=20)
# plt.savefig(os.path.join(output_folder, f"model_summary_{mode_suffix}.png"))
#
# # 5. TEST CONFIDENCE SCATTER PLOT (Replaces the bar chart)
# # Extract Video #, Confidence, and Result
# test_details = re.findall(r'VIDEO (\d+)/10.*?Confidence: ([\d.]+)%.*?RESULT: (PASS|FAIL)', full_content, re.DOTALL)
# if test_details:
#     vids = [int(v) for v, c, r in test_details]
#     conf = [float(c) for v, c, r in test_details]
#     results = [r for v, c, r in test_details]
#     colors = ['#4CAF50' if r == 'PASS' else '#F44336' for r in results]
#
#     plt.figure(figsize=(10, 6))
#     for i in range(len(vids)):
#         plt.scatter(vids[i], conf[i], color=colors[i], s=100, edgecolors='black', zorder=3)
#         plt.text(vids[i], conf[i] + 1, f"{conf[i]}%", ha='center', fontsize=9)
#
#     plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)  # 50% threshold line
#     plt.ylim(0, 105)
#     plt.xticks(range(1, 11))
#     plt.xlabel('Video Test Case Number')
#     plt.ylabel('Prediction Confidence (%)')
#     plt.title(f'Test Results: Confidence vs. Success ({mode_suffix})')
#     plt.grid(True, axis='y', alpha=0.3)
#
#     # Legend
#     from matplotlib.lines import Line2D
#
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label='PASS', markerfacecolor='#4CAF50', markersize=10),
#                        Line2D([0], [0], marker='o', color='w', label='FAIL', markerfacecolor='#F44336', markersize=10)]
#     plt.legend(handles=legend_elements, loc='lower right')
#
#     plt.savefig(os.path.join(output_folder, f"test_confidence_scatter_{mode_suffix}.png"))
#
# # 6. COMPREHENSIVE TEST EVALUATION (Grid/Checkerboard Style)
# test_runs = re.split(r'################ test on videos \d+:', full_content)
# test_runs = [run for run in test_runs if "Testing:" in run]
#
# if test_runs:
#     # Scale figure height based on number of test runs
#     fig_test, axes = plt.subplots(len(test_runs), 1, figsize=(22, 10 * len(test_runs)), squeeze=False)
#     fig_test.suptitle(f'Detailed Inference Evaluation: {mode_suffix}', fontsize=22, y=1.01)
#
#     # Define the valid Class IDs (0-81, skipping 63)
#     all_class_ids = [i for i in range(82) if i != 63]
#
#     for idx, run_content in enumerate(test_runs):
#         ax = axes[idx, 0]
#
#         # Regex to grab: Video Filename, True ID, Pred ID, Confidence, Result
#         pattern = r"Opened: (.*?)\s+True ID: (\d+).*?Pred ID: (\d+).*?Confidence: ([\d.]+)%.*?RESULT: (PASS|FAIL)"
#         matches = re.findall(pattern, run_content, re.DOTALL)
#
#         if not matches:
#             continue
#
#         # Reverse matches so Video 1/10 is at the bottom
#         matches = matches[::-1]
#
#         video_names = [m[0] for m in matches]
#         true_ids = [int(m[1]) for m in matches]
#         pred_ids = [int(m[2]) for m in matches]
#         confs = [float(m[3]) for m in matches]
#         results = [m[4] for m in matches]
#
#         y_pos = np.arange(len(video_names))
#
#         # Create checkerboard grid effect
#         for i in all_class_ids:
#             ax.axvline(x=i, color='gray', linestyle='-', alpha=0.1, zorder=0)
#         for j in y_pos:
#             ax.axhline(y=j, color='gray', linestyle='-', alpha=0.1, zorder=0)
#
#         # Plot markers
#         # True IDs as Blue circles
#         ax.scatter(true_ids, y_pos, color='#1f77b4', s=180, label='True ID', marker='o', edgecolors='black', zorder=3)
#
#         for i in range(len(video_names)):
#             is_pass = results[i] == 'PASS'
#             color = '#4CAF50' if is_pass else '#F44336'
#
#             # FIXED: removed 'fontweight', used 'linewidths' for boldness on 'x' marker
#             ax.scatter(pred_ids[i], y_pos[i], color=color, s=150, marker='x', linewidths=3, zorder=4)
#
#             # Draw line between True and Pred for FAIL cases
#             if not is_pass:
#                 ax.plot([true_ids[i], pred_ids[i]], [y_pos[i], y_pos[i]], color='#F44336', linestyle='--', alpha=0.5,
#                         zorder=2)
#
#             # Confidence text - positioned slightly to the side
#             label_x = max(true_ids[i], pred_ids[i]) + 0.8
#             ax.text(label_x, y_pos[i], f"{confs[i]}%", va='center', fontsize=10, fontweight='bold',
#                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
#
#         # Formatting
#         ax.set_title(f'Test Run {idx + 1}: Class Correspondence Grid', fontsize=18, pad=20)
#
#         ax.set_xticks(all_class_ids)
#         ax.set_xticklabels(all_class_ids, rotation=90, fontsize=9)
#         ax.set_xlabel('Class ID (0-81, skipping 63)', fontsize=14)
#         ax.set_xlim(-1, 82)
#
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(video_names, fontsize=11)
#         ax.set_ylabel('Tested Video Files', fontsize=14)
#
#         # Legend
#         from matplotlib.lines import Line2D
#
#         legend_elements = [
#             Line2D([0], [0], marker='o', color='w', label='True Label', markerfacecolor='#1f77b4', markersize=12,
#                    markeredgecolor='black'),
#             Line2D([0], [0], marker='x', color='w', label='Correct Prediction', markeredgecolor='#4CAF50',
#                    markersize=12, mew=3),
#             Line2D([0], [0], marker='x', color='w', label='Incorrect Prediction', markeredgecolor='#F44336',
#                    markersize=12, mew=3)
#         ]
#         ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend")
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f"test_grid_evaluation_{mode_suffix}.png"), bbox_inches='tight', dpi=300)
#
# print(f"Grid comparison graph saved for {mode_suffix}.")
# print(f"All graphs (including Model Summary and Scatter Plot) generated for {mode_suffix}.")






#  V 1.4 train & continue train
# import matplotlib.pyplot as plt
# import re
# import os
#
# # input_file = r'C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\logs\dl\dl_train_1_512.txt'
# input_file = r'C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\logs\dl\dl_train_Continue_1_512.txt'
# output_dir = r'C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\graphs\dl'
# # output_filename = 'V_1.4_training.png'
# output_filename = 'V_1.4_continue_training.png'
#
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# epochs, acc, val_acc, loss, val_loss = [], [], [], [], []
#
# with open(input_file, 'r') as f:
#     content = f.read()
#     # Split by "Epoch X/240"
#     epoch_sections = re.split(r'Epoch (\d+)/\d+', content)
#
#     for i in range(1, len(epoch_sections), 2):
#         e_num = int(epoch_sections[i])
#         metrics = epoch_sections[i + 1]
#
#         # Regex to find the metrics in your specific log format
#         m_acc = re.search(r'accuracy: ([\d.]+)', metrics)
#         m_v_acc = re.search(r'val_accuracy: ([\d.]+)', metrics)
#         m_loss = re.search(r'loss: ([\d.]+)', metrics)
#         m_v_loss = re.search(r'val_loss: ([\d.]+)', metrics)
#
#         if all([m_acc, m_v_acc, m_loss, m_v_loss]):
#             epochs.append(e_num)
#             acc.append(float(m_acc.group(1)))
#             val_acc.append(float(m_v_acc.group(1)))
#             loss.append(float(m_loss.group(1)))
#             val_loss.append(float(m_v_loss.group(1)))
#
# plt.style.use('default')
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# # fig.suptitle('Training Performance: KrSL-FluentSigners Kuro V 1.4', fontsize=18)
# fig.suptitle('Continue Training Performance: KrSL-FluentSigners Kuro V 1.4', fontsize=18)
#
# ax1.plot(epochs, acc, label='Initial Train', color='#1f77b4', linewidth=2)
# ax1.plot(epochs, val_acc, label='Initial Val', color='#ff7f0e', linestyle='--', linewidth=2)
# ax1.set_title('Accuracy', fontsize=14)
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Accuracy')
# ax1.grid(True, alpha=0.3)
# ax1.legend()
#
# ax2.plot(epochs, loss, label='Initial Train', color='#1f77b4', linewidth=2)
# ax2.plot(epochs, val_loss, label='Initial Val', color='#ff7f0e', linestyle='--', linewidth=2)
# ax2.set_title('Loss', fontsize=14)
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Loss')
# ax2.grid(True, alpha=0.3)
# ax2.legend()
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# save_path = os.path.join(output_dir, output_filename)
# plt.savefig(save_path, dpi=300)
# print(f"Graph successfully saved to: {save_path}")
# plt.show()


### v 1.4 video test
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

file_path = r'C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\logs\dl\dl_test_1_Video_10_Tests.txt'
output_dir = r'C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\graphs\dl'


def parse_and_plot_all(filename, save_path):
    all_data = []
    current_file, current_true, current_pred = None, None, None

    if not os.path.exists(filename):
        print(f"Error: Log file not found at {filename}")
        return

    # Parse all data from the log file
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("File:"):
                current_file = line.replace("File:", "").strip()
            elif line.startswith("True ID:"):
                current_true = int(line.replace("True ID:", "").strip())
            elif line.startswith("Pred ID:"):
                current_pred = int(line.replace("Pred ID:", "").strip())
            elif line.startswith("Confidence:"):
                conf_val = float(line.replace("Confidence:", "").replace("%", "").strip())
                if current_file and current_true is not None and current_pred is not None:
                    all_data.append(
                        {'file': current_file, 'true': current_true, 'pred': current_pred, 'conf': conf_val})
                    current_file, current_true, current_pred = None, None, None

    if len(all_data) < 20:
        print(f"Only found {len(all_data)} entries. Plotting as single graph...")
        # Fallback to single plot logic if file is shorter than expected
        return

        # Split data for the two models
    best_data = all_data[:10]
    supreme_data = all_data[10:20]

    # Create one large figure with 2 subplots (stacked vertically)[cite: 1, 2]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    plt.subplots_adjust(hspace=0.3)  # Space between the two graphs

    # Helper function to draw on specific axes
    def draw_subplot(ax, data, title):
        filenames = [d['file'] for d in data]
        true_labels = [d['true'] for d in data]
        predicted_labels = [d['pred'] for d in data]
        confidences = [d['conf'] for d in data]
        y_axis = range(len(filenames))

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.scatter(true_labels, y_axis, color='#1f77b4', s=120, label='Target Class', zorder=3)

        for i in range(len(filenames)):
            is_correct = true_labels[i] == predicted_labels[i]
            color = 'green' if is_correct else 'red'
            ax.scatter(predicted_labels[i], i, marker='x', color=color, s=150, zorder=4)
            ax.text(predicted_labels[i] + 1.5, i, f"{confidences[i]}%", fontsize=10, color=color,
                    verticalalignment='center', fontweight='bold' if is_correct else 'normal')
            if not is_correct:
                ax.plot([true_labels[i], predicted_labels[i]], [i, i], color='red', linestyle='--', linewidth=1.5,
                        alpha=0.4)

        ax.set_yticks(y_axis)
        ax.set_yticklabels(filenames, fontsize=11)
        ax.set_xticks(range(0, 85, 5))
        ax.set_xlabel("Class ID (Kazakh Sign Language)", fontsize=12)
        ax.set_title(title, fontsize=16, pad=15)
        ax.invert_yaxis()

    # Draw both sections
    draw_subplot(ax1, best_data, "Inference Evaluation: Masked Best Model")
    draw_subplot(ax2, supreme_data, "Inference Evaluation: Masked Supreme Model")

    # Add Shared Legend to the top right[cite: 1, 2]
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Class', markerfacecolor='#1f77b4', markersize=10),
        Line2D([0], [0], marker='x', color='green', label='Correct Prediction', markersize=12, linestyle='None'),
        Line2D([0], [0], marker='x', color='red', label='Incorrect Prediction', markersize=12, linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)

    # Save final high-res composite image[cite: 1, 3]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    final_path = os.path.join(save_path, "full_model_comparison.png")
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"Full comparison graph saved to: {final_path}")
    plt.show()

if __name__ == "__main__":
    parse_and_plot_all(file_path, output_dir)
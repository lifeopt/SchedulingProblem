import matplotlib.pyplot as plt
import numpy as np
def draw_gantt_chart(ax, env_idx, job_schedule_matrix, operation_job_idxs, colors):
    max_job_idx = max(operation_job_idxs)
    # num_machines, maxT = job_schedule_matrix.shape[1:]
    num_machines, maxT = job_schedule_matrix.shape
    rect_height = 1

    ax.set_title(f'Gantt Chart for Environment {env_idx + 1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine Index')
    ax.set_yticks(range(num_machines))
    ax.set_ylim(-0.5, num_machines - 0.5)
    ax.grid(True)

    ax.set_xticks(range(0, maxT + 1, maxT // 10))
    ax.set_xlim(0, maxT)

    legend_elements = [plt.Line2D([0], [0], color=colors[job_idx + 1], lw=4, label=f'Job {job_idx}') for job_idx in range(max_job_idx + 1)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Job Color Key')

    for machine_idx in range(num_machines):
        time_idx = 0
        while time_idx < maxT:
            job_idx = job_schedule_matrix[machine_idx][time_idx]
            if job_idx != -1:
                start_time = time_idx
                while time_idx < maxT and job_schedule_matrix[machine_idx][time_idx] == job_idx:
                    time_idx += 1
                end_time = time_idx
                rect = plt.Rectangle((start_time, machine_idx - rect_height / 2), end_time - start_time, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.annotate(job_idx, (start_time + (end_time - start_time) / 2, machine_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
            else:
                time_idx += 1
    plt.tight_layout()
    
def draw_unallocated_operations_new(ax, env_idx, operation_processing_times, operation_allocation_status, operation_job_idxs, colors):
    max_job_idx = max(operation_job_idxs)
    rect_height = 1

    ax.set_title(f'Unallocated Operations for Environment {env_idx + 1}')
    ax.set_xlabel('Cumulative Processing Time')
    ax.set_ylabel('Job Index')
    ax.set_yticks(range(max_job_idx + 1))
    ax.set_ylim(-0.5, max_job_idx + 0.5)
    ax.grid(True)

    cumulative_processing_time = np.zeros(max_job_idx + 1)
    allocated_ops = [op_idx for op_idx, allocated in enumerate(operation_allocation_status) if allocated]
    unallocated_ops = [op_idx for op_idx, allocated in enumerate(operation_allocation_status) if not allocated]

    start_x = 0
    if unallocated_ops:
        for op_idx in unallocated_ops:
            job_idx = operation_job_idxs[op_idx]
            processing_time = operation_processing_times[op_idx]
            end_x = start_x + processing_time
            rect = plt.Rectangle((start_x, job_idx - rect_height / 2), processing_time, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.annotate(f'op{op_idx}', (start_x + processing_time / 2, job_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
            cumulative_processing_time[job_idx] = end_x
            start_x = end_x

    if allocated_ops:
        for op_idx in allocated_ops:
            job_idx = operation_job_idxs[op_idx]
            processing_time = operation_processing_times[op_idx]
            start_x = cumulative_processing_time[job_idx]
            end_x = start_x + processing_time
            rect = plt.Rectangle((start_x, job_idx - rect_height / 2), processing_time, rect_height, hatch='/', edgecolor='black', linewidth=1, fill=False)
            ax.add_patch(rect)
            ax.annotate(f'op{op_idx}', (start_x + processing_time / 2, job_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
            cumulative_processing_time[job_idx] = end_x

    max_cumulative_processing_time = max(cumulative_processing_time) if cumulative_processing_time.any() else 0
    ax.set_xlim(0, max_cumulative_processing_time)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    legend_elements = [plt.Line2D([0], [0], color=colors[job_idx + 1], lw=4, label=f'Job {job_idx}') for job_idx in range(max_job_idx + 1)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Job Color Key')
    plt.tight_layout()
    
def draw_gantt_chart_v2(env_idx, job_schedule_matrix, processing_times, operation_allocation_status, operation_job_idxs):
    # num_envs, num_machines, maxT = job_schedule_matrix.shape
    max_job_idx = max(operation_job_idxs)
    colors = plt.cm.tab20b(np.linspace(0, 1, max_job_idx + 2))

    # for env_idx in range(num_envs):
    fig, ax = plt.subplots()
    draw_gantt_chart(ax, env_idx, job_schedule_matrix, operation_job_idxs, colors)

    fig2, ax2 = plt.subplots()
    draw_unallocated_operations_new(ax2, env_idx, processing_times, operation_allocation_status, operation_job_idxs, colors)
    # draw_unallocated_operations(ax2, env_idx, processing_times, operation_allocation_status, operation_job_idxs, colors)

    plt.show()
    
                       
if __name__ == '__main__':
    num_envs = 1
    num_machines = 3

    job_schedule_matrix = np.array([
        [-1, -1,  0,  0,  0,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3, -1, -1, -1],
        [-1,  0,  0,  0, -1, -1, -1,  1,  1,  1,  1,  1,  4,  4,  4,  4,  4, -1, -1,  2,  2,  2,  2,  2, -1, -1, -1,  3,  3,  3],
        [ 0,  0,  0, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  4,  4,  4,  4,  4, -1, -1,  2,  2,  2,  2,  2,  3,  3,  3,  3]
    ])

    operation_processing_times = [3, 5, 5, 5, 5, 5, 7, 3, 3, 3, 3]
    operation_allocation_status = [True, True, True, True, True, True, True, True, False, True, False]
    operation_job_idxs = [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4]

    draw_gantt_chart_v2(0, job_schedule_matrix, operation_processing_times, operation_allocation_status, operation_job_idxs)

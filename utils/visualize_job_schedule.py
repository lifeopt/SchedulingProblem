import matplotlib.pyplot as plt
import numpy as np

def generate_example(num_envs, num_machines, maxT, mean_processing_time, variance):
    job_schedule_matrix = np.full((num_envs, num_machines, maxT), -1)
    
    for env_idx in range(num_envs):
        job_idx = 0
        for machine_idx in range(num_machines):
            time_idx = 0
            while time_idx < maxT:
                processing_time = int(np.random.normal(mean_processing_time, np.sqrt(variance)))
                if time_idx + processing_time <= maxT:
                    job_schedule_matrix[env_idx, machine_idx, time_idx:time_idx + processing_time] = job_idx
                    time_idx += processing_time
                else:
                    break
                job_idx += 1
    return job_schedule_matrix

def draw_gantt_chart1(job_schedule_matrix, env_idx):
    num_envs, num_machines, maxT = job_schedule_matrix.shape
    max_job_idx = np.max(job_schedule_matrix)
    colors = plt.cm.tab20b(np.linspace(0, 1, max_job_idx + 2))

    rect_height = 1

    fig, ax = plt.subplots()
    ax.set_title(f'Gantt Chart for Environment {env_idx + 1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine Index')
    ax.set_yticks(range(num_machines))
    ax.set_ylim(-0.5, num_machines - 0.5)
    ax.grid(True)

    # Find the latest assigned operation time index
    latest_assigned_time = np.max(np.nonzero(job_schedule_matrix[env_idx] != -1)) + 1
    extended_time = latest_assigned_time + 50

    ax.set_xticks(range(0, extended_time + 1, extended_time // 10))
    ax.set_xlim(0, extended_time)

    # Custom legend
    legend_elements = [plt.Line2D([0], [0], color=colors[job_idx + 1], lw=4, label=f'Job {job_idx}') for job_idx in range(max_job_idx + 1)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Job Color Key')

    for machine_idx in range(num_machines):
        time_idx = 0
        while time_idx < maxT:
            job_idx = job_schedule_matrix[env_idx][machine_idx][time_idx]
            if job_idx != -1:
                start_time = time_idx
                while time_idx < maxT and job_schedule_matrix[env_idx][machine_idx][time_idx] == job_idx:
                    time_idx += 1
                end_time = time_idx
                rect = plt.Rectangle((start_time, machine_idx - rect_height / 2), end_time - start_time, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.annotate(job_idx, (start_time + (end_time - start_time) / 2, machine_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
            else:
                time_idx += 1

    plt.tight_layout()
    plt.show()

def draw_gantt_chart_gpt(job_schedule_matrix):
    num_envs, num_machines, maxT = job_schedule_matrix.shape
    max_job_idx = np.max(job_schedule_matrix)
    colors = plt.cm.tab20b(np.linspace(0, 1, max_job_idx + 2))

    rect_height = 1

    for env_idx in range(num_envs):
        fig, ax = plt.subplots()
        ax.set_title(f'Gantt Chart for Environment {env_idx + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine Index')
        ax.set_yticks(range(num_machines))
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.grid(True)

        # Find the latest assigned operation time index
        latest_assigned_time = np.max(np.nonzero(job_schedule_matrix[env_idx] != -1)) + 1
        extended_time = latest_assigned_time + 50

        ax.set_xticks(range(0, extended_time + 1, extended_time // 10))
        ax.set_xlim(0, extended_time)

        # Custom legend
        legend_elements = [plt.Line2D([0], [0], color=colors[job_idx + 1], lw=4, label=f'Job {job_idx}') for job_idx in range(max_job_idx + 1)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Job Color Key')

        for machine_idx in range(num_machines):
            time_idx = 0
            while time_idx < maxT:
                job_idx = job_schedule_matrix[env_idx][machine_idx][time_idx]
                if job_idx != -1:
                    start_time = time_idx
                    while time_idx < maxT and job_schedule_matrix[env_idx][machine_idx][time_idx] == job_idx:
                        time_idx += 1
                    end_time = time_idx
                    rect = plt.Rectangle((start_time, machine_idx - rect_height / 2), end_time - start_time, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.annotate(job_idx, (start_time + (end_time - start_time) / 2, machine_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
                else:
                    time_idx += 1

        plt.tight_layout()
        plt.show()
        
        
def draw_gantt_chart(job_schedule_matrix, processing_times):
    num_envs, num_machines, maxT = job_schedule_matrix.shape
    max_job_idx = np.max(job_schedule_matrix)
    colors = plt.cm.tab20b(np.linspace(0, 1, max_job_idx + 2))

    rect_height = 1

    for env_idx in range(num_envs):
        fig, ax = plt.subplots()
        ax.set_title(f'Gantt Chart for Environment {env_idx + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine Index')
        ax.set_yticks(range(num_machines))
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.grid(True)

        # Find the latest assigned operation time index
        latest_assigned_time = np.max(np.nonzero(job_schedule_matrix[env_idx] != -1)) + 1
        extended_time = latest_assigned_time + 50

        ax.set_xticks(range(0, extended_time + 1, extended_time // 10))
        ax.set_xlim(0, extended_time)

        # Custom legend
        legend_elements = [plt.Line2D([0], [0], color=colors[job_idx + 1], lw=4, label=f'Job {job_idx}') for job_idx in range(max_job_idx + 1)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Job Color Key')

        for machine_idx in range(num_machines):
            time_idx = 0
            while time_idx < maxT:
                job_idx = job_schedule_matrix[env_idx][machine_idx][time_idx]
                if job_idx != -1:
                    start_time = time_idx
                    while time_idx < maxT and job_schedule_matrix[env_idx][machine_idx][time_idx] == job_idx:
                        time_idx += 1
                    end_time = time_idx
                    rect = plt.Rectangle((start_time, machine_idx - rect_height / 2), end_time - start_time, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.annotate(job_idx, (start_time + (end_time - start_time) / 2, machine_idx), color='black', weight='bold', fontsize=8, ha='center', va='center')
                else:
                    time_idx += 1

        # Display unassigned operations
        unassigned_y = -1.5
        for job_idx in range(max_job_idx + 1):
            unassigned_ops = [op_idx for op_idx, op_job_idx in enumerate(job_schedule_matrix[env_idx].flatten()) if op_job_idx == job_idx]
            if unassigned_ops:
                label_x = 0
                ax.annotate(f'Job {job_idx}:', (label_x, unassigned_y), color='black', weight='bold', fontsize=8, ha='left', va='center')
                for op_idx in unassigned_ops:
                    start_x = label_x + 60
                    end_x = start_x + processing_times[op_idx]
                    rect = plt.Rectangle((start_x, unassigned_y - rect_height / 2), end_x - start_x, rect_height, facecolor=colors[job_idx + 1], edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.annotate(op_idx, (start_x + (end_x - start_x) / 2, unassigned_y), color='black', weight='bold', fontsize=8, ha='center', va='center')
                    label_x = end_x + 10
                unassigned_y -= 1

        plt.tight_layout()
        plt.show()
                      
if __name__ == '__main__':



    # Small Example
    num_envs = 1
    num_machines = 3
    maxT = 20

    job_schedule_matrix = np.array([[
        [-1, -1,  1,  1,  1,  1,  1, -1, -1,  2,  2,  2,  2,  2, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  0,  0,  0,  0,  0, -1]
    ]])

    operation_allocation_status = np.array([1, 1, 1, 0])
    operation_job_idxs = np.array([0, 1, 2, 1])
    operation_processing_times = np.array([5, 5, 5, 7])
    
    # draw_gantt_chart(job_schedule_matrix, operation_allocation_status, operation_job_idxs, operation_processing_times)
    draw_gantt_chart1(job_schedule_matrix)
    

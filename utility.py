import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def calculate_tardiness(job_schedule_matrix, due_dates, processing_times):
    num_machines = len(job_schedule_matrix)
    num_time_indexes = len(job_schedule_matrix[0])
    
    completion_times = [0] * num_machines
    tardiness = [0] * num_machines

    for machine_idx in range(num_machines):
        current_time = 0

        for time_idx in range(num_time_indexes):
            job_idx = job_schedule_matrix[machine_idx][time_idx]

            if job_idx != -1:  # If a job is scheduled at this time index
                processing_time = processing_times[job_idx]
                completion_time = current_time + processing_time
                completion_times[machine_idx] = completion_time

                tardiness[machine_idx] += max(0, completion_time - due_dates[job_idx])

                current_time = completion_time

    return tardiness

def plot_gantt_chart(job_schedule_matrix, num_jobs):
    fig, ax = plt.subplots()

    # Define a colormap for different job indices
    colors = plt.cm.get_cmap('tab20', num_jobs)

    for m, machine_schedule in enumerate(job_schedule_matrix):
        for t, job_idx in enumerate(machine_schedule):
            if job_idx != -1:
                # Plot a rectangle for each job with the corresponding color
                rect = plt.Rectangle((t, m), 1, 1, facecolor=colors(job_idx), edgecolor='black')
                ax.add_patch(rect)

    # Set axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine Index')
    ax.set_title('Gantt Chart')
    plt.xticks(np.arange(0, len(machine_schedule), 1))
    plt.yticks(np.arange(0, len(job_schedule_matrix), 1))

    # Display the plot
    plt.show()
    
if __name__ == '__main__':
    plot_gantt_chart
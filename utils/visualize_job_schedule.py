import matplotlib.pyplot as plt
import numpy as np

def display_gantt_chart(job_schedule_matrix, training_iteration, interval=10):
    if training_iteration % interval == 0:
        n_envs, n_machines, n_time_indexes = job_schedule_matrix.shape

        for env_idx in range(n_envs):
            plt.clf()
            fig, ax = plt.subplots()

            cmap = plt.get_cmap('tab20', np.max(job_schedule_matrix[env_idx]) + 2)

            for machine in range(n_machines):
                for time_index in range(n_time_indexes):
                    job_idx = job_schedule_matrix[env_idx, machine, time_index]
                    if job_idx >= 0:
                        ax.broken_barh([(time_index, 1)], (machine, 1), facecolors=(cmap(job_idx + 1),), edgecolor='black')

            ax.set_ylim(0, n_machines)
            ax.set_xlim(0, n_time_indexes)
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Machine Index')
            ax.set_yticks(np.arange(n_machines) + 0.5)
            ax.set_yticklabels(np.arange(n_machines))
            ax.set_xticks(np.arange(0, n_time_indexes + 1, 1))
            ax.grid(True)

            plt.title(f'Job Schedule Matrix for Environment {env_idx} (Iteration {training_iteration})')
            plt.show(block=False)
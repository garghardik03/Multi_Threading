import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num_matrices = 100
matrix_size = 1000
num_threads = [1, 2, 4, 8, 12]


def matrix_multiply(A, B):
    return np.matmul(A, B)


def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)


constant_matrix = generate_random_matrix(matrix_size, matrix_size)


# Function to execute matrix multiplication in a thread
def worker(matrices, constant_matrix, results, start_index, end_index):
    for i in range(start_index, end_index):
        results[i] = matrix_multiply(matrices[i], constant_matrix)


# precompute kardiya so that benchmark ke liye sahi ho
matrices = [
    generate_random_matrix(matrix_size, matrix_size) for _ in range(num_matrices)
]

times = []
for num_thread in num_threads:
    start_time = time.time()

    # Create threads
    results = [None] * num_matrices
    threads = []
    chunk_size = num_matrices // num_thread

    for i in range(num_thread):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        if i == num_thread - 1:
            end_index = num_matrices
        thread = threading.Thread(
            target=worker,
            args=(matrices, constant_matrix, results, start_index, end_index),
        )
        threads.append(thread)
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    end_time = time.time()
    times.append(end_time - start_time)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(num_threads, times)
plt.xlabel("Number of Threads")
plt.ylabel("Time (seconds)")
plt.title("Matrix Multiplication Benchmark")
plt.show()


# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Number of Threads': num_threads,
    'Time Taken (seconds)': times
})

# Print DataFrame
print("Results DataFrame:")
print(results_df)

# Save DataFrame to CSV file
results_df.to_csv('time_vs_threads.csv', index=False)

print("\nResults saved to 'time_vs_threads.csv'")
import os
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

# If using BLAS-based libraries, you can force them to use one thread.
if os.cpu_count() and os.cpu_count() > 1:
    os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# Problem-specific modules/files.
# In Julia these were loaded via include("problem_triangle_free.jl") and "constants.jl".
# Here we assume that any needed constants (like N) and functions are defined.
# For example, if there is a module "problem_triangle_free.py" defining greedy_search_from_startpoint,
# you could import it. Otherwise, we provide a placeholder.

try:
    from problem_triangle_free import greedy_search_from_startpoint
except ImportError:
    def greedy_search_from_startpoint(db, obj):
        # Placeholder implementation:
        # In the actual problem, this should perform a greedy search on the starting object.
        # Here we simply return a list containing the input object.
        return [obj]

# For example, we assume a constant N is defined.
N = 10

def empty_starting_point():
    """Return an empty starting point as a string of zeros for the upper-triangle of an N x N matrix."""
    return "0" * (N * (N - 1) // 2)

# -----------------------------------------------------------------------------
# File and plotting helpers

def find_next_available_filename(base, extension):
    i = 1
    while True:
        filename = os.path.join(write_path, f"{base}_{i}.{extension}")
        if not os.path.isfile(filename):
            return filename
        i += 1

def write_output_to_file(db):
    rewards_sorted = sorted(db.rewards.keys(), reverse=True)
    base_name = "search_output"
    extension = "txt"
    filename = find_next_available_filename(base_name, extension)
    lines_written = 0
    with open(filename, "w") as file:
        curr_rew_index = 0
        while lines_written < final_database_size and curr_rew_index < len(rewards_sorted):
            curr_rew = rewards_sorted[curr_rew_index]
            objs = db.rewards[curr_rew]
            # Write only as many objects as needed to reach final_database_size.
            to_write = objs[:min(final_database_size - lines_written, len(objs))]
            for obj in to_write:
                file.write(obj + "\n")
            lines_written += len(objs)
            curr_rew_index += 1

    print(f"Data written to {filename}")
    if rewards_sorted:
        print(f"An example of an object with maximum reward ({rewards_sorted[0]}):")
        print(db.rewards[rewards_sorted[0]][0])

def write_plot_to_file(db):
    rewards_sorted = sorted(db.rewards.keys(), reverse=True)
    reward_counts = [len(db.rewards[r]) for r in rewards_sorted]

    # Create the overall score distribution plot.
    plt.figure()
    plt.bar(rewards_sorted, reward_counts)
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    base_name = "plot"
    extension = "png"
    filename = find_next_available_filename(base_name, extension)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

    # Save the score distribution to a text file.
    txt_filename = os.path.join(write_path, "distribution.txt")
    with open(txt_filename, "w") as f:
        for rew, count in zip(rewards_sorted, reward_counts):
            f.write(f"Score: {rew}, Count: {count}\n")
    print(f"Score distribution saved to {txt_filename}")

    # Filter rewards to only include up to final_database_size objects.
    cumulative_count = 0
    filtered_rewards = []
    filtered_counts = []
    for rew, count in zip(rewards_sorted, reward_counts):
        if cumulative_count >= final_database_size:
            break
        next_count = min(count, final_database_size - cumulative_count)
        filtered_rewards.append(rew)
        filtered_counts.append(next_count)
        cumulative_count += next_count

    plt.figure()
    plt.bar(filtered_rewards, filtered_counts)
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    base_name = "plot_training"
    extension = "png"
    filename = find_next_available_filename(base_name, extension)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

    txt_filename = os.path.join(write_path, "training_distribution.txt")
    with open(txt_filename, "w") as f:
        for rew, count in zip(filtered_rewards, filtered_counts):
            f.write(f"Score: {rew}, Count: {count}\n")
    print(f"Filtered score distribution saved to {txt_filename}")

# -----------------------------------------------------------------------------
# Database definition and related functions

class Database:
    def __init__(self, objects, rewards, local_search_indices):
        # objects: dict mapping object -> reward
        # rewards: dict mapping reward -> list of objects
        # local_search_indices: dict mapping reward -> int (last index for which local search has been performed)
        self.objects = objects
        self.rewards = rewards
        self.local_search_indices = local_search_indices

def new_db():
    return Database({}, {}, {})

def reward_calc(obj):
    """Calculate the reward for an object.
    Here we assume the reward is the number of '1's in the string."""
    return obj.count("1")

def reward_obj(obj):
    return reward_calc(obj)

def reward_with_db(db, obj):
    if obj in db.objects:
        return db.objects[obj], False
    return reward_obj(obj), True

def add_db(db, list_obj, list_rew=None):
    rewards_new_objects = []
    if list_rew is not None:
        for i, obj in enumerate(list_obj):
            if obj not in db.objects:
                rew = list_rew[i]
                rewards_new_objects.append(rew)
                db.objects[obj] = rew
                if rew not in db.rewards:
                    db.rewards[rew] = [obj]
                    db.local_search_indices[rew] = 0
                else:
                    db.rewards[rew].append(obj)
    else:
        # Compute rewards for new objects.
        list_indices = [i for i, obj in enumerate(list_obj) if obj not in db.objects]
        list_rew = [0] * len(list_obj)
        for i in list_indices:
            list_rew[i] = reward_obj(list_obj[i])
        for i in list_indices:
            obj = list_obj[i]
            rew = list_rew[i]
            rewards_new_objects.append(rew)
            db.objects[obj] = rew
            if rew not in db.rewards:
                db.rewards[rew] = [obj]
                db.local_search_indices[rew] = 0
            else:
                db.rewards[rew].append(obj)
    return rewards_new_objects

def shrink(db):
    """Shrink the database to the target number of objects."""
    count = 0
    rewards_sorted = sorted(db.rewards.keys(), reverse=True)
    for rew in rewards_sorted:
        if count < target_db_size:
            lg = len(db.rewards[rew])
            count += lg
            if count > target_db_size:
                k = count - target_db_size
                # Remove the last k objects.
                to_remove = db.rewards[rew][-k:]
                for obj in to_remove:
                    try:
                        del db.objects[obj]
                    except Exception as e:
                        print("whoopsie", e)
                db.rewards[rew] = db.rewards[rew][:-k]
                db.local_search_indices[rew] = min(db.local_search_indices.get(rew, 0), len(db.rewards[rew]))
        else:
            for obj in db.rewards[rew]:
                if obj in db.objects:
                    del db.objects[obj]
            del db.rewards[rew]
            if rew in db.local_search_indices:
                del db.local_search_indices[rew]

def print_db(db):
    rewards_sorted = sorted(db.rewards.keys(), reverse=True)
    db_size = sum(len(db.rewards[r]) for r in rewards_sorted)
    if db_size > 2 * target_db_size:
        print(f" - Shrinking database to {target_db_size} best objects")
        shrink(db)
        rewards_sorted = sorted(db.rewards.keys(), reverse=True)
    # (Optional) You could print a summary here if desired.

# -----------------------------------------------------------------------------
# Local search functions

def local_search_on_object(db, obj):
    """
    Perform a local search on the given object.
    Returns a tuple (list_of_new_objects, list_of_their_rewards).
    """
    objects_found = []
    rewards_found = []
    # In the actual implementation, greedy_search_from_startpoint might return several candidates.
    greedily_expanded_objs = greedy_search_from_startpoint(db, obj)
    for candidate in greedily_expanded_objs:
        rew, is_new = reward_with_db(db, candidate)
        if is_new:
            objects_found.append(candidate)
            rewards_found.append(rew)
    return objects_found, rewards_found

def local_search(db, lines, start_ind, nb=None):
    """Perform local search on a slice of the input lines.
    
    This function mimics Julia’s multithreaded local search:
      - It creates a pool of objects from lines[start_ind : start_ind+nb]
      - Runs local_search_on_object on each in parallel,
      - And then adds the new objects to the database.
    """
    if nb is None:
        nb = nb_local_searches
    pool = lines[start_ind : min(start_ind + nb, len(lines))]
    local_search_results = []

    num_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(local_search_on_object, db, obj): obj for obj in pool}
        for future in futures:
            try:
                result = future.result()
                local_search_results.append(result)
            except Exception as e:
                print("Error during local search:", e)

    # Update the database with results from all threads.
    for objs, rews in local_search_results:
        add_db(db, objs, rews)

# -----------------------------------------------------------------------------
# Initial input processing

def initial_lines():
    input_file = ""
    args = sys.argv[1:]  # skip the script name
    if "-i" in args or "--input" in args:
        if "-i" in args:
            index = args.index("-i") + 1
        else:
            index = args.index("--input") + 1
        if index < len(args):
            input_file = args[index]
    print("Input file:", input_file)
    lines = []
    if input_file:
        print("Using input file")
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == len(empty_starting_point()):
                    lines.append(line)
    else:
        print("No input file provided")
        for _ in range(num_initial_empty_objects):
            lines.append(empty_starting_point())
    return lines

# -----------------------------------------------------------------------------
# Main search loop

def main():
    db = new_db()
    lines = initial_lines()
    print(len(lines))
    print(len(set(lines)))
    print(f"Using {os.cpu_count() or 1} thread(s)")

    start_idx = 0
    steps = 0
    time_since_previous_output = 0

    while start_idx < len(lines):
        start_time = time.time()
        local_search(db, lines, start_idx)
        time_local_search = time.time() - start_time
        time_since_previous_output += time_local_search
        start_idx += nb_local_searches
        steps += 1
        time_local_search = round(time_local_search, 2)
        print_db(db)
    print_db(db)
    write_output_to_file(db)
    write_plot_to_file(db)

# -----------------------------------------------------------------------------
# Entry point and argument parsing

if __name__ == "__main__":
    # Expect the following command-line arguments:
    # sys.argv[1] = write_path
    # sys.argv[2] = nb_local_searches
    # sys.argv[3] = num_initial_empty_objects
    # sys.argv[4] = final_database_size
    # sys.argv[5] = target_db_size
    if len(sys.argv) < 6:
        print("Usage: script.py write_path nb_local_searches num_initial_empty_objects final_database_size target_db_size")
        sys.exit(1)

    write_path = sys.argv[1]
    nb_local_searches = int(sys.argv[2])
    num_initial_empty_objects = int(sys.argv[3])
    final_database_size = int(sys.argv[4])
    target_db_size = int(sys.argv[5])
    main()

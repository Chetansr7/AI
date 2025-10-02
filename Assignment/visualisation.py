import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_map_image_with_path(image_path="map.jpg", path_positions=None):
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return

    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Map with Solution Path")
    ax.axis('off')

    img_height, img_width = img.shape[0], img.shape[1]
    cell_width = img_width / 10
    cell_height = img_height / 6

    if path_positions:
        xs = [x * cell_width + cell_width / 1 for (x, y) in path_positions]
        ys = [y * cell_height + cell_height * 1 for (x, y) in path_positions]

        ax.plot(xs, ys, color='red', linestyle='-', linewidth=2, label='Path')
        ax.scatter(xs, ys, color='red', s=50)

        ax.scatter(xs[0], ys[0], color='lime', s=100, label='Start', zorder=5)
        ax.scatter(xs[-1], ys[-1], color='blue', s=100, label='Goal', zorder=5)

        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    plt.pause(1)

def print_world_with_path(world, path_positions):
    print("\nMap with Path (•), Treasures ($), Rewards (R), Traps (T), Obstacles (X):\n")
    for row in range(world.rows):
        line = ""
        for col in range(world.cols):
            pos = (col, row)
            node = world.node_grid[col][row]
            if pos in path_positions:
                line += "• "
            elif node.is_obstacle():
                line += "X "
            elif node.is_treasure():
                line += "$ "
            elif node.is_trap():
                line += "T "
            elif node.is_reward():
                line += "R "
            else:
                line += ". "
        print(line)

def generate_step_by_step_solution(path):
    print("\nStep-by-Step Movement Log:")
    for i, step in enumerate(path):
        print(f"Step {i + 1}: {step}")

def print_statistics(final_state, total_steps, nodes_expanded, elapsed_time):
    print("\nSummary Statistics: ")
    print(f"Total steps taken: {total_steps}")
    print(f"Energy consumed: {100 - final_state.energy_remaining}")
    print(f"Treasures collected: {len(final_state.treasures_collected)}")
    print(f"Rewards collected: {len(final_state.rewards_collected)}")
    print(f"Nodes expanded: {nodes_expanded}")
    print(f"Execution time: {elapsed_time:.3f} seconds")

def export_results(path, filename="path_output.txt"):
    with open(filename, "w") as f:
        f.write("Step-by-step path:\n")
        for i, step in enumerate(path):
            f.write(f"Step {i + 1}: {step}\n")

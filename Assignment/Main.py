from Game import Game
from State import State
from visualisation import (
    show_map_image_with_path,
    print_world_with_path,
    generate_step_by_step_solution,
    print_statistics,
    export_results
)
import time
import re
import pathlib

def main():
    file_path = pathlib.Path("Locations.txt")
    absolute_file_path = file_path.absolute()

    game = Game(absolute_file_path)

    start_time = time.time()
    path, nodes_expanded = game.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if path:
        print("\n✓ Path to collect all treasures and rewards found!\n")
        generate_step_by_step_solution(path)

        # Reconstruct final state by replaying the path
        state = State(
            agent_location=(0, 0),
            energy_remaining=100,
            treasures_collected=[],
            remaining_treasures=game.all_treasures.copy(),
            rewards_collected=[],
            remaining_rewards=game.all_rewards.copy(),
            effects_active=[]
        )

        path_positions = []
        for step in path:
            match = re.search(r"\((\d+),\s*(\d+)\)", step)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                next_pos = (x, y)
                path_positions.append(next_pos)
                state = game.apply_move(state, next_pos)

        # Show statistics
        print_statistics(state, len(path), nodes_expanded, elapsed_time)

        # Export path
        export_results(path)

        # Print map + show image
        print_world_with_path(game, path_positions)
        show_map_image_with_path("map.jpg", path_positions)

    else:
        print("\n✗ No path found to collect all treasures and rewards.")
        print(f"Nodes expanded during search: {nodes_expanded}")

if __name__ == '__main__':
    main()

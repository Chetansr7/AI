import heapq
import itertools

from Node import Node
from State import State


class Game:

    # =============================================================================
    #     A hexagonal grid game where an agent collects treasures while avoiding obstacles.
    #
    #     Attributes:
    #         rows (int): Number of rows in the grid
    #         cols (int): Number of columns in the grid
    #         node_grid (list): 2D list representing the game grid
    #         all_treasures (list): Positions of all treasures to collect
    #         start_position (tuple): Starting position of the agent
    # =============================================================================

    def __init__(self, filename):
        self.rows = 6
        self.cols = 10
        self.node_grid = [[None for _ in range(
            self.rows)] for _ in range(self.cols)]
        self.all_treasures = []
        self.all_rewards = []
        self.start_position = (0, 0)
        self.load_map(filename)

    def load_map(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() if line.strip()
                         else '.' for line in f.readlines()]
            index = 0
            for col in range(self.cols):
                for row in range(self.rows):
                    content = lines[index]
                    pos = (col, row)
                    node = Node(pos, content, [], 0)
                    if node.content == 'TR':
                        self.all_treasures.append(pos)
                    elif node.is_reward():
                        self.all_rewards.append(pos)
                    self.node_grid[col][row] = node
                    index += 1
        except FileNotFoundError:
            print("Locations.txt file not found!")

    def start_game(self):
        path, nodes_expanded = self.solve()
        if path:
            print("\nPath to collect all treasures:")
            for step in path:
                print(step)
            print(f"\nTotal steps: {len(path)}")
        else:
            print("\nNo path found to collect all treasures.")

        print(f"Nodes expanded during search: {nodes_expanded}")

    def solve(self):
        if not self.all_treasures and not self.all_rewards:
            return [], 0

        initial_state = State(
            agent_location=(0, 0),
            energy_remaining=100,
            treasures_collected=[],
            remaining_treasures=self.all_treasures.copy(),
            rewards_collected=[],
            remaining_rewards=self.all_rewards.copy(),
            effects_active=[]
        )

        open_list = []
        closed_set = set()
        counter = itertools.count()
        initial_h = self.calculate_heuristic(initial_state)
        start_node = (initial_h,
                      next(counter),
                      initial_state,
                      0,
                      initial_h,
                      [])
        heapq.heappush(open_list, start_node)

        nodes_expanded = 0
        solution_found = None

        while open_list and solution_found is None:
            current_f, _, current_state, current_g, current_h, path = heapq.heappop(
                open_list)

            if current_state.is_goal_state():
                solution_found = (path, nodes_expanded)
                continue

            state_key = hash(current_state)
            if state_key in closed_set:
                continue

            closed_set.add(state_key)
            nodes_expanded += 1

            for next_state, action in self.get_successors(current_state):
                col, row = next_state.agent_location
                node_content = self.node_grid[col][row]
                if not next_state or not next_state.is_valid() or node_content.is_trap():
                    continue

                next_key = hash(next_state)
                if next_key in closed_set:
                    continue

                new_g = current_f
                new_h = self.calculate_heuristic(next_state)
                new_f = new_g + new_h

                action_type = "Move"
                if node_content.content == 'TR' and (col, row) in current_state.remaining_treasures:
                    action_type = "Collect Treasure"
                elif node_content.is_reward() and (col, row) in current_state.remaining_rewards:
                    action_type = "Collect Reward"

                formatted_action = f"\n{action}\nf={current_f}, g={
                    current_g}, h={current_h}\n{action_type}"
                new_path = path + [formatted_action]

                new_node = (new_f,
                            next(counter),
                            next_state,
                            new_g,
                            new_h,
                            new_path)
                heapq.heappush(open_list, new_node)

        return solution_found if solution_found else ([], nodes_expanded)

    def calculate_step_cost(self, current_state, next_state):
        return self.get_energy_multiplier(current_state.effects_active)

    # MST
    def calculate_heuristic(self, state):
        
# =============================================================================
#         Calculates heuristic cost using Minimum Spanning Tree (MST) approach.
#     
#         Combines:
#         1. Distance from agent to nearest goal (treasure/reward)
#         2. MST cost connecting all remaining goals
#         
#         Steps:
#         1. Get all uncollected goals (treasures + rewards)
#         2. If no goals left, return 0 (already at goal state)
#         3. Calculate minimum distance from agent to any goal
#         4. Compute MST cost connecting all remaining goals
#         5. Return sum of (nearest goal distance + MST cost)
# =============================================================================
        
        goals = state.remaining_treasures + state.remaining_rewards
        if not goals:
            return 0

        agent_pos = state.agent_location
        min_agent_to_goal = min(self.hex_distance(
            agent_pos, goal) for goal in goals)

        mst_cost = 0
        visited = set()
        unvisited = set(goals)
        current = unvisited.pop()
        visited.add(current)

        while unvisited:
            min_edge = float('inf')
            next_node = None
            for v in visited:
                for u in unvisited:
                    dist = self.hex_distance(v, u)
                    if dist < min_edge:
                        min_edge = dist
                        next_node = u
            mst_cost += min_edge
            visited.add(next_node)
            unvisited.remove(next_node)

        return min_agent_to_goal + mst_cost

    def hex_distance(self, pos1, pos2):
        q1, r1 = pos1
        q2, r2 = pos2
        dq = q1 - q2
        dr = r1 - r2
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    def get_successors(self, current_state):
        successors = []
        current_pos = current_state.agent_location
        q, r = current_pos
        neighbors = self.get_neighbours(q, r)

        for next_pos in neighbors:
            next_node = self.get_node(next_pos)
            if next_node is None or next_node.content == 'OB':
                continue

            new_state = self.apply_move(current_state, next_pos)
            if new_state is None or not new_state.is_valid():
                continue

            action = f"Move from {current_pos} to {next_pos}"
            successors.append((new_state, action))

        return successors

    def get_neighbours(self, q: int, r: int) -> list[tuple[int, int]]:

        # =============================================================================
        #         Finds the six nearest neighbors for a given hexagonal coordinate (q, r)
        #         in an "even-q" vertical layout.
        #
        #         The "even-q" layout shifts even columns down, which affects the 'r'
        #         coordinate calculation for diagonal neighbors.
        #
        #         Args:
        #             q: The 'q' (column) coordinate of the hexagon.
        #             r: The 'r' (row) coordinate of the hexagon.
        #
        #         Returns:
        #             A list of tuples, where each tuple represents the (q, r) coordinates
        #             of a neighboring hexagon.
        # =============================================================================

        neighbors = []
        if q % 2 == 0:
            # Offsets for even 'q' columns
            # (delta_q, delta_r)
            directions = [
                (1,  0),   # Top-right
                (1,  1),   # Bottom-right
                (0,  1),   # Bottom
                (-1, 1),   # Bottom-left
                (-1, 0),   # Top-left
                (0, -1)    # Top
            ]
        else:
            # Offsets for odd 'q' columns
            # (delta_q, delta_r)
            directions = [
                (1, -1),   # Top-right
                (1,  0),   # Bottom-right
                (0,  1),   # Bottom
                (-1, 0),   # Bottom-left
                (-1, -1),  # Top-left
                (0, -1)    # Top
            ]

        # Calculate the coordinates of each neighbor
        for dq, dr in directions:
            neighbor_q = q + dq
            neighbor_r = r + dr

            if (neighbor_q > -1) and (neighbor_r > -1):
                neighbors.append((neighbor_q, neighbor_r))
        return neighbors

    def get_node(self, position):
        col, row = position
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return self.node_grid[col][row]
        return None

    def get_movement_multiplier(self, effects_active):
        multiplier = 1
        if "speed_boost" in effects_active:
            multiplier *= 0.5
        if "slow" in effects_active:
            multiplier *= 2
        return multiplier

    def get_energy_multiplier(self, effects_active):
        multiplier = 1
        if "high_energy_cost" in effects_active:
            multiplier *= 2
        return multiplier

    def apply_move(self, current_state, next_pos):
        
# =============================================================================
#         Applies a move to the game state and returns the new state after the move.
#     
#         Handles:
#         - Energy cost deduction
#         - Treasure collection
#         - Reward collection and effects
#         - Trap effects
#         - State updates
# =============================================================================
        
        next_node = self.get_node(next_pos)
        if next_node is None:
            return None

        new_treasures_collected = current_state.treasures_collected.copy()
        new_remaining_treasures = current_state.remaining_treasures.copy()
        new_rewards_collected = current_state.rewards_collected.copy()
        new_remaining_rewards = current_state.remaining_rewards.copy()
        new_effects = current_state.effects_active.copy()

        # Deduct energy cost
        energy_cost = self.get_energy_multiplier(current_state.effects_active)
        new_energy = current_state.energy_remaining - energy_cost

        # Collect treasure if exists at this position
        if next_node.content == 'TR' and next_pos in new_remaining_treasures:
            new_remaining_treasures.remove(next_pos)
            new_treasures_collected.append(next_pos)

        # Collect reward if exists at this position
        if next_node.is_reward() and next_pos in new_remaining_rewards:
            new_remaining_rewards.remove(next_pos)
            new_rewards_collected.append(next_pos)
            reward_num = next_node.get_reward_number()
            new_effects = self.apply_reward_effect(reward_num, new_effects)

        # Apply trap effects
        if next_node.is_trap():
            trap_num = next_node.get_trap_number()
            new_effects, new_remaining_treasures = self.apply_trap_effect(
                trap_num, new_effects, new_remaining_treasures)

        return State(
            agent_location=next_pos,
            energy_remaining=new_energy,
            treasures_collected=new_treasures_collected,
            remaining_treasures=new_remaining_treasures,
            rewards_collected=new_rewards_collected,
            remaining_rewards=new_remaining_rewards,
            effects_active=new_effects
        )

    def apply_trap_effect(self, trap_number, effects_active, remaining_treasures):
        if trap_number == 1:
            if "slow" not in effects_active:
                effects_active.append("slow")
        elif trap_number == 2:
            if "high_energy_cost" not in effects_active:
                effects_active.append("high_energy_cost")
        elif trap_number == 3 and remaining_treasures:
            # Remove first treasure in list for determinism
            remaining_treasures.pop(0)
        return effects_active, remaining_treasures

    def apply_reward_effect(self, reward_number, effects_active):
        if reward_number == 1:
            if "speed_boost" not in effects_active:
                effects_active.append("speed_boost")
            # Speed boost cancels slow effect
            if "slow" in effects_active:
                effects_active.remove("slow")
        elif reward_number == 2:
            if "high_energy_cost" in effects_active:
                effects_active.remove("high_energy_cost")
        return effects_active

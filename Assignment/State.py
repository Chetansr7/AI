class State:
    def __init__(self, agent_location, energy_remaining, treasures_collected,
                 remaining_treasures, rewards_collected, remaining_rewards, effects_active):
        self.agent_location = agent_location
        self.energy_remaining = energy_remaining
        self.treasures_collected = list(treasures_collected)
        self.remaining_treasures = list(remaining_treasures)
        self.rewards_collected = list(rewards_collected)
        self.remaining_rewards = list(remaining_rewards)
        self.effects_active = list(effects_active)

    def __eq__(self, other):
        return (self.agent_location == other.agent_location and
                self.energy_remaining == other.energy_remaining and
                sorted(self.treasures_collected) == sorted(other.treasures_collected) and
                sorted(self.remaining_treasures) == sorted(other.remaining_treasures) and
                sorted(self.effects_active) == sorted(other.effects_active))

    def __hash__(self):
        """Ensure the object is hashable for use in sets/dicts"""
        return hash((self.agent_location,
                     self.energy_remaining,
                     tuple(sorted(self.treasures_collected)),
                     tuple(sorted(self.remaining_treasures)),
                     tuple(sorted(self.effects_active))))

    def is_goal_state(self):
        """Check if all treasures AND rewards have been collected."""
        return len(self.remaining_treasures) == 0 and len(self.remaining_rewards) == 0

    def is_valid(self):
        """Check if the state is valid (e.g. energy is not negative)."""
        return self.energy_remaining >= 0

    def __repr__(self):
        return (f"State(pos={self.agent_location}, energy={self.energy_remaining}, "
                f"collected={len(self.treasures_collected)}, remaining={len(self.remaining_treasures)}, "
                f"effects={self.effects_active})")

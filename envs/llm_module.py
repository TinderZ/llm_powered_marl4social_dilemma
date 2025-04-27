# llm_module.py
import random
from typing import Dict, Optional

class LLMModule:
    """
    LLM module for an agent in the Cleanup environment.
    """
    def __init__(self, agent_id: str):
        """
        Initializes the LLM module for a specific agent.
        Args:
            agent_id: The ID of the agent this module belongs to.
        """
        self.agent_id = agent_id
        # In a real scenario, you might load LLM models or configurations here.
        #print(f"LLM Module initialized for {agent_id}")

    def process_game_info(self, game_info: str, llm_type: str, all_cumulative_rewards: Dict[str, float]) -> Optional[str]:
        """
        Processes the received game information and decides on a command.
        This is a placeholder implementation.

        Args:
            game_info: A string describing the current game state
                       (e.g., "River severely polluted, apples cannot grow").
            llm_type: The type of LLM being used (e.g., "rule-based", "gemini", "deepseek", "openai").
        Returns:
            A command string ("clean up", "collect apples") or None.
        """

        if llm_type == "rule-based":
            # Use the rule-based logic based on cumulative reward rank
            command = self._get_rule_based_command(all_cumulative_rewards)
            print(f"{self.agent_id} LLM (rule-based) received: '{game_info}'. Output: {command}")
            return command
        elif llm_type == "random": # Example of another type
            # Original random logic (for comparison or other scenarios)
            if "severely polluted" in game_info:
                action = "clean up"
            else:
                action = random.choice(["collect apples", None, None])
            print(f"{self.agent_id} LLM (random) received: '{game_info}'. Output: {action}")
            return action
        else:
            # TODO: Implement actual LLM logic (e.g., prompting a real model) for other llm_types
            print(f"{self.agent_id} LLM ({llm_type}) received: '{game_info}'. No logic implemented yet (TODO). Output: None")
            return None # No command for unimplemented types


    def _get_rule_based_command(self, all_cumulative_rewards: Dict[str, float]) -> Optional[str]:
        """
        Determines the command based on the agent's rank in cumulative rewards.

        Args:
            all_cumulative_rewards: A dictionary mapping agent IDs to their cumulative rewards.

        Returns:
            "collect apples" if the agent is in the bottom two,
            "clean up" otherwise.
        """
        if not all_cumulative_rewards or self.agent_id not in all_cumulative_rewards:
            # Cannot determine rank if data is missing or agent not found
            print(f"Warning: Could not determine rank for {self.agent_id}. Rewards: {all_cumulative_rewards}")
            return None # Default to no command

        # Sort agents by cumulative reward (ascending)
        sorted_agents = sorted(all_cumulative_rewards.items(), key=lambda item: item[1])

        # Find the rank of the current agent (index + 1)
        try:
            my_rank = [i for i, (agent_id, _) in enumerate(sorted_agents) if agent_id == self.agent_id][0] + 1
        except IndexError:
             print(f"Warning: Agent {self.agent_id} not found in sorted rewards during rank calculation.")
             return None # Agent not found

        num_agents = len(sorted_agents)

        # Check if the agent is in the bottom two
        # Handles cases with 1 or 2 agents as well
        # if num_agents <= 2:
        #      # If only 1 or 2 agents, everyone is "bottom two" technically
        #      command = "collect apples"
        if my_rank <= 2: # Rank is num_agents or num_agents - 1
             command = "collect apples"
        else:
             command = "clean up"

        #print(f"{self.agent_id} Rank: {my_rank}/{num_agents}, Cumulative Reward: {all_cumulative_rewards[self.agent_id]:.2f} -> Command: {command}")
        return command

    def discuss(self, other_llm_outputs: dict):
        """
        Placeholder for LLM discussion logic.

        Args:
            other_llm_outputs: A dictionary mapping other agent IDs to their
                               LLM outputs from the current step.
        """
        # TODO: Implement LLM discussion logic here.
        # This could involve exchanging messages, reaching consensus, etc.
        # For now, it does nothing.
        pass
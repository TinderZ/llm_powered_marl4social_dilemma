# cleanup_env.py
import functools
import random
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


from envs.cleanup_agent import CleanupAgent
from envs.llm_module import LLMModule

from envs.constants import IMMOBILIZE_DURATION_HIT, IMMOBILIZE_DURATION_FIRE, STAY_ACTION_INDEX
from envs.constants import (ACTION_MEANING, APPLE, APPLE_REWARD, APPLE_RESPAWN_PROBABILITY,
                     APPLE_SPAWN, AGENT_CHARS, CLEANUP_MAP, CLEANUP_VIEW_SIZE, CLEAN_BEAM_LENGTH,
                     CLEAN_BEAM_WIDTH, CLEAN_REWARD, CLEANABLE_TILES, CLEAN_BLOCKING_CELLS,
                     CLEANED_TILE_RESULT, CLEAN_BEAM, DEFAULT_COLOURS, EMPTY, FIRE_BEAM_LENGTH,
                     FIRE_BEAM_WIDTH, FIRE_BLOCKING_CELLS, MOVE_ACTIONS, NON_WALKABLE, NUM_ACTIONS,
                     ORIENTATIONS, ORIENTATION_VECTORS, PENALTY_BEAM, PENALTY_FIRE, PENALTY_HIT,
                     RIVER, ROTATION_MAP, SPECIAL_ACTIONS, STREAM, THRESHOLD_DEPLETION,
                     THRESHOLD_RESTORATION, TURN_ACTIONS, VIEW_PADDING, WALL, WASTE,
                     WASTE_INIT, WASTE_SPAWN_PROBABILITY, AGENT_START)
from envs.constants import (CLEAN_BEAM_LENGTH_VALID)

class CleanupEnv(ParallelEnv):
    """
    Cleanup Social Dilemma Environment based on the original implementation,
    refactored for the PettingZoo Parallel API.

    In this environment, agents are incentivized to collect apples (reward=1).
    Apples grow based on the cleanliness of a nearby river. Cleaning the river
    requires agents to use a cleaning beam. However, the river gets polluted by
    waste generated in a separate spawning area. Agents can use a "penalty" beam
    to punish other agents (cost=-1 for firing, reward=-50 for being hit).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "cleanup_v1"}

    def __init__(
        self,
        num_agents: int = 5,
        render_mode: str | None = None,
        max_cycles: int = 1000,
        use_collective_reward: bool = False,  # Not implemented in detail from original, but kept as option
        inequity_averse_reward: bool = False, # Not implemented in detail from original, but kept as option
        alpha: float = 0.0,                   # Parameter for IAR
        beta: float = 0.0,                    # Parameter for IAR

        use_llm: bool = False,                # Enable/disable LLM features
        llm_f_step: int = 50,                 # Frequency of LLM calling
        llm_type: str = "rule-based"          # Type of LLM to use
    ):
        """
        Initializes the Cleanup environment.

        Args:
            num_agents: The number of agents in the environment.
            render_mode: The mode for rendering ('human' or 'rgb_array').
            max_cycles: The maximum number of steps per episode.
            use_collective_reward: Whether to use a collective reward signal.
            inequity_averse_reward: Whether to use inequity aversion rewards.
            alpha: Inequity aversion parameter (penalty for others having more).
            beta: Inequity aversion parameter (penalty for having less than others).

            use_llm: Whether to enable LLM-based observation masking.
            llm_f_step: How often (in steps) the LLM processes game info.
        """
        super().__init__()

        if num_agents <= 0:
            raise ValueError("Number of agents must be positive.")
        if num_agents > len(AGENT_CHARS):
             raise ValueError(f"Maximum number of agents is {len(AGENT_CHARS)}")

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        self.agent_id_map = {i: f"agent_{i}" for i in range(num_agents)}
        self.render_mode = render_mode
        self.max_cycles = max_cycles

        # Reward structure (advanced options not fully implemented from original)
        self.use_collective_reward = use_collective_reward
        self.inequity_averse_reward = inequity_averse_reward
        self.alpha = alpha
        self.beta = beta

        # LLM-related attributes
        self.use_llm = use_llm 
        self.llm_f_step = llm_f_step
        self.llm_type = llm_type
        self.llm_modules: dict[str, LLMModule] = {}
        self.llm_commands: dict[str, str | None] = {} # Store current command per agent
        if self.use_llm:
            print(f"LLM integration enabled. Update frequency: {self.llm_f_step} steps.")
            for agent_id in self.possible_agents:
                 self.llm_modules[agent_id] = LLMModule(agent_id)
                 self.llm_commands[agent_id] = None # Initialize with no command

        # Load map and initialize state variables
        self.base_map = self._ascii_to_numpy(CLEANUP_MAP)
        self.world_map = deepcopy(self.base_map)
        self.map_height, self.map_width = self.base_map.shape

        # Find initial points
        self.spawn_points = self._find_points(AGENT_START)
        self.apple_spawn_points = self._find_points(APPLE_SPAWN)
        self.waste_init_points = self._find_points(WASTE_INIT)
        self.river_points = self._find_points(RIVER)
        self.stream_points = self._find_points(STREAM) # Stream tiles 'S'
        self.wall_points = self._find_points(WALL)
        

        # Waste dynamics related points
        self.potential_waste_area = len(self.waste_init_points) + len(self.river_points)
        self.waste_spawn_points = self.waste_init_points + self.river_points # All points where waste can exist

        # Initialize agents
        self._agents: dict[str, CleanupAgent] = {} # Use dict for agent management
        # Will be populated in reset()

        # State variables updated during steps
        self.current_apple_spawn_prob = APPLE_RESPAWN_PROBABILITY
        self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY
        self._compute_probabilities() # Initial calculation

        self.num_cycles = 0
        self.beam_pos = [] # Positions currently occupied by beams for rendering

        # Setup rendering if needed
        self.fig = None
        self.ax = None
        self.render_im = None

        # --- PettingZoo API Properties ---
        # Define observation space: RGB image view for each agent
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=0, high=255,
                shape=(2 * CLEANUP_VIEW_SIZE + 1, 2 * CLEANUP_VIEW_SIZE + 1, 3),
                dtype=np.uint8
            ) for agent_id in self.possible_agents
        }

        # Define action space: Discrete actions for each agent
        self.action_spaces = {
            agent_id: spaces.Discrete(NUM_ACTIONS) for agent_id in self.possible_agents
        }


    # --- PettingZoo API Methods ---

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for a single agent."""
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for a single agent."""
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> dict[str, np.ndarray]:
        """Resets the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = self.possible_agents[:] # Active agents list
        self._agents = {} # Clear agent objects

        # Reset map to base state (walls, empty spaces, initial waste/river)
        self.world_map = deepcopy(self.base_map)
        # Custom reset: place initial waste and river tiles correctly
        self._reset_map_features()

        # Spawn agents at unique starting locations
        available_spawn_points = deepcopy(self.spawn_points)
        if len(available_spawn_points) < self.num_agents:
            raise ValueError("Not enough spawn points for the number of agents.")
        random.shuffle(available_spawn_points)

        for i, agent_id in enumerate(self.agents):
            spawn_pos = np.array(available_spawn_points[i])
            # Random initial orientation
            orientation = random.choice(list(ORIENTATIONS.keys()))
            self._agents[agent_id] = CleanupAgent(
                agent_id_num=i,
                start_pos=spawn_pos,
                start_orientation=orientation,
                view_len=CLEANUP_VIEW_SIZE
            )
            # Remove agent start 'P' from world map
            self.world_map[spawn_pos[0], spawn_pos[1]] = EMPTY

        # Reset dynamics and counters
        self._compute_probabilities()
        self.num_cycles = 0
        self.beam_pos = []

        # Reset LLM commands
        if self.use_llm:
            for agent_id in self.possible_agents:
                self.llm_commands[agent_id] = None

        # Get initial observations
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents} # Create empty info dict
        #print(f"Initial observations: {observations}")
        #print(f"Initial infos: {infos}")

        if self.render_mode == "human":
            self.render()

        return observations, infos # Return both observations and infos

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Advances the environment by one step based on agent actions."""
        self.num_cycles += 1
        self.beam_pos = [] # Clear beams from previous step

        # <--- 记录步骤开始时的活动智能体 --->
        # 目的是确保后续的奖励、终止/截断状态和观测都基于这个列表计算
        agents_at_step_start = self.agents[:]
        # print(f"Active agents: {agents_at_step_start}")

        # --- Handle Immobilization and Override Actions ---
        original_actions = actions.copy() # Keep original for reference if needed
        for agent_id in agents_at_step_start:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                if agent.is_immobilized():
                    # Override action to STAY
                    actions[agent_id] = STAY_ACTION_INDEX
                    # Decrement counter
                    agent.decrement_immobilization()


        # 1. Process Actions (Movement, Turns, Special Actions)
        # <--- 修改: 使用 agents_at_step_start 初始化奖励字典 --->
        rewards = {agent_id: 0.0 for agent_id in agents_at_step_start}
        
        agent_action_map = {} # Store decoded action strings
        agent_new_positions = {} # Store intended new positions
        agent_new_orientations = {} # Store new orientations

        # Decode actions and handle turns immediately
        for agent_id, action_code in actions.items():
            # <--- 检查 agent_id 是否在 agents_at_step_start 中 --->
            # 忽略那些在该步骤开始时就已经不活动的智能体的动作
            if agent_id not in self._agents or agent_id not in agents_at_step_start: continue # Skip if agent is already done

            agent = self._agents[agent_id]
            action_str = ACTION_MEANING.get(action_code)
            agent_action_map[agent_id] = action_str

            if action_str in TURN_ACTIONS:
                new_orientation = ROTATION_MAP.get((agent.get_orientation(), action_str))
                agent.set_orientation(new_orientation)
                agent_new_orientations[agent_id] = new_orientation
            elif action_str in MOVE_ACTIONS:
                move_vec = MOVE_ACTIONS[action_str]    #形如[0,1]
                # Rotate move vector based on agent orientation
                rotated_move = self._rotate_vector(move_vec, agent.get_orientation())
                intended_pos = agent.get_pos() + rotated_move
                agent_new_positions[agent_id] = intended_pos
            # Special actions (FIRE, CLEAN) are handled after movement

        # 2. Resolve Movement Conflicts and Update Positions
        final_positions = self._resolve_movement_conflicts(agent_new_positions)
        for agent_id, final_pos in final_positions.items():
             self._agents[agent_id].set_pos(final_pos)

        # 3. Handle Consumption (Apples)
        for agent_id in agents_at_step_start:
            #if agent_id not in self._agents: continue # 确保智能体仍然存在
            agent = self._agents[agent_id]
            pos = agent.get_pos()
            tile = self.world_map[pos[0], pos[1]]
            if tile == APPLE:
                agent.add_reward(APPLE_REWARD)
                self._update_map_tile(pos[0], pos[1], EMPTY)


        # 4. Handle Special Actions (Firing/Cleaning Beams) in random order
        beam_updates = [] # Store tile changes from beams
        for agent_id in agents_at_step_start: #shuffled_agent_ids:  无随机性
            agent = self._agents.get(agent_id)
            if not agent: continue # Agent might be done
            action_str = agent_action_map.get(agent_id)
            
            if action_str == "FIRE":
                agent.immobilize(IMMOBILIZE_DURATION_FIRE)
                agent.add_reward(-PENALTY_FIRE) # Cost for firing
                fire_updates = self._fire_beam(
                    agent.get_pos(), agent.get_orientation(), FIRE_BEAM_LENGTH,
                    PENALTY_BEAM, [], [], FIRE_BLOCKING_CELLS, FIRE_BEAM_WIDTH
                )
                beam_updates.extend(fire_updates) # FIRE beam doesn't change tiles, only hits agents

            elif action_str == "CLEAN":
                agent.add_reward(CLEAN_REWARD) # Cost/reward for cleaning
                clean_updates = self._fire_beam(
                    agent.get_pos(), agent.get_orientation(), CLEAN_BEAM_LENGTH,
                    CLEAN_BEAM, CLEANABLE_TILES, CLEANED_TILE_RESULT, CLEAN_BLOCKING_CELLS, CLEAN_BEAM_WIDTH
                )
                beam_updates.extend(clean_updates) # Store tile changes

        # Apply beam updates to the map
        for r, c, char in beam_updates:
             self._update_map_tile(r, c, char)

        # 5. Update Environment State (Waste/Apple Spawning)
        self._compute_probabilities() # Update probs based on current waste
        spawn_updates = self._spawn_apples_and_waste()
        for r, c, char in spawn_updates:
            self._update_map_tile(r, c, char)

        # 6. Calculate Rewards and Termination/Truncation
        # 基于 agents_at_step_start 初始化状态字典 
        terminations = {agent_id: False for agent_id in agents_at_step_start}
        truncations = {agent_id: False for agent_id in agents_at_step_start}
        infos = {agent_id: {} for agent_id in agents_at_step_start}
        
        # Get rewards accumulated by agents
        
        for agent_id in agents_at_step_start:
            # if agent_id in self._agents: # Ensure agent is still active
            agent = self._agents.get(agent_id)
            if agent:  # Check if agent exists before consuming reward
                rewards[agent_id] += agent.consume_reward() # Add rewards from hits/consumption
                agent.add_cumulative_reward(rewards[agent_id]) # Update cumulative apple/hit reward
                
            else:
                print(f"Warning: Agent {agent_id} not found in self._agents.")


        # Apply collective/IAR rewards if enabled (Simplified - full IAR needs careful implementation)
        # collective reeward, inequity penalty
        if self.use_collective_reward:
             total_reward = sum(rewards.values())
             rewards = {agent_id: total_reward for agent_id in agents_at_step_start}
        elif self.inequity_averse_reward and self.num_agents > 1:
             current_rewards = rewards.copy()
             for agent_id_i in agents_at_step_start:
                 inequity_penalty = 0
                 for agent_id_j in agents_at_step_start:
                     if agent_id_i == agent_id_j: continue
                     diff = current_rewards[agent_id_j] - current_rewards[agent_id_i]
                     if diff > 0: # Disadvantageous inequity
                         inequity_penalty += self.alpha * diff
                     elif diff < 0: # Advantageous inequity
                         inequity_penalty += self.beta * abs(diff) # beta is typically negative, so abs() or adjust sign
                 rewards[agent_id_i] -= inequity_penalty / (self.num_agents - 1)


        # Check truncation (max cycles)
        is_truncated = self.num_cycles >= self.max_cycles
        if is_truncated:
            truncations = {agent_id: True for agent_id in agents_at_step_start}
            # Don't clear self.agents here yet


        # 7. get llm commands
        if self.use_llm and (self.num_cycles % self.llm_f_step == 0):
            # 1. Gather Game Information
            current_waste = np.count_nonzero(self.world_map == WASTE)
            waste_density = 0
            if self.potential_waste_area > 0:
                waste_density = current_waste / self.potential_waste_area

            game_info = ""
            if waste_density >= THRESHOLD_DEPLETION:
                game_info = "River severely polluted, apples cannot grow."
            elif waste_density <= THRESHOLD_RESTORATION:
                game_info = "River is clean, apples can grow well."
            else:
                # More nuanced info could be added here
                game_info = f"River pollution level moderate (density: {waste_density:.2f})."

            # 2. Process Info and Get Commands for each agent active at step start
            agents_cumulative_rewards = {
                 agent_id: self._agents[agent_id].get_cumulative_reward()
                 for agent_id in agents_at_step_start #if agent_id in self._agents
            }

            for agent_id in agents_at_step_start:
                if agent_id in self.llm_modules:
                    command = self.llm_modules[agent_id].process_game_info(game_info, self.llm_type, agents_cumulative_rewards)
                    self.llm_commands[agent_id] = command
                    #llm_outputs_this_step[agent_id] = command
                else:
                    # Handle case where agent might not have an LLM module? Default to None.
                    self.llm_commands[agent_id] = None
                    #llm_outputs_this_step[agent_id] = None
                
        
        # 8. Generate observations for all agents active at the start of the step
        observations = {}
        for agent_id in agents_at_step_start:
            #if agent_id in self._agents: # Check if agent object still exists
            observations[agent_id] = self._get_observation(agent_id)
            #else:
                # Handle cases where agent might have been unexpectedly removed
                # Maybe return a default observation or log an error
                # For now, we assume _get_observation handles missing agents gracefully if needed,
                # or simply don't add the key if the agent object is gone.
                # Let's assume _get_observation needs a valid agent_id from _agents
                # If agent terminates/truncates, they might still need an obs for this final step
                # The most robust way is perhaps calling _get_observation even if agent terminates this turn
                #  try:
                #      observations[agent_id] = self._get_observation(agent_id)
                #  except KeyError:
                #       print(f"Warning: Agent {agent_id} not found in self._agents when getting observation, though active at step start.")
                #       # Decide how to handle this: skip, add default, etc.
                #       # Skipping for now.
                #       pass

        # --- Modification Start: Update self.agents list based on term/trunc flags ---
        # Determine the agents who will be active in the *next* step
        next_agents = []
        for agent_id in agents_at_step_start:
            if not terminations[agent_id] and not truncations[agent_id]:
                next_agents.append(agent_id)
        self.agents = next_agents
        # --- Modification End ---


        if self.render_mode == "human":
            self.render()

        # PettingZoo expects dicts for all return values, keyed by agent ID
        # Ensure all returned dicts have keys from agents_at_step_start
        # (Observations dict is already handled. Rewards/Terms/Truncs/Infos were initialized based on it)

        return observations, rewards, terminations, truncations, infos



    def render(self) -> np.ndarray | None:
        """Renders the environment."""
        rgb_map = self._map_to_colors()

        if self.render_mode == "human":
            if self.fig is None:
                plt.ion() # Interactive mode on
                self.fig, self.ax = plt.subplots(1, 1)
                self.render_im = self.ax.imshow(rgb_map, interpolation='nearest')
                plt.title("Cleanup Social Dilemma")
            else:
                self.render_im.set_data(rgb_map)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events() # Update display
            return None
        elif self.render_mode == "rgb_array":
            return rgb_map
        return None


    def close(self):
        """Closes the rendering window."""
        if self.fig is not None:
            plt.ioff() # Interactive mode off
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.render_im = None


    # --- Helper Methods ---

    def _reset_map_features(self):
        """Places initial waste, river, and stream tiles."""
        for r, c in self.waste_init_points:
             self.world_map[r, c] = WASTE
        for r, c in self.river_points:
             self.world_map[r, c] = RIVER
        for r, c in self.stream_points:
             self.world_map[r, c] = STREAM # Place stream tiles


    def _ascii_to_numpy(self, ascii_map):
        """Converts the ASCII map list to a numpy byte array."""
        return np.array([[c.encode('ascii') for c in row] for row in ascii_map])

    def _find_points(self, char_to_find):
        """Finds all coordinates (row, col) of a given character in the base map."""
        return np.argwhere(self.base_map == char_to_find).tolist()

    def _get_map_with_agents(self):
        """Creates a temporary map view with agents and beams placed."""
        map_view = np.copy(self.world_map)
        # Place agents
        for agent in self._agents.values():
            pos = agent.get_pos()
            # Check bounds just in case
            if 0 <= pos[0] < self.map_height and 0 <= pos[1] < self.map_width:
                 # Check if beam is already there, beams have render priority
                 if map_view[pos[0], pos[1]] not in [PENALTY_BEAM, CLEAN_BEAM]:
                     map_view[pos[0], pos[1]] = agent.get_agent_char()

        # Place beams (render priority over agents)
        for r, c, beam_char in self.beam_pos:
             if 0 <= r < self.map_height and 0 <= c < self.map_width:
                 map_view[r, c] = beam_char
        return map_view

    def _map_to_colors(self) -> np.ndarray:
        """Converts the current world map state (with agents) to an RGB array."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' ']) # Default to black if char not found
        return rgb_map


    def _map_to_colors_mask_apple(self) -> np.ndarray:
        """Generates RGB map masking Apples ('A') as Grass ('B')."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        grass_color = DEFAULT_COLOURS[APPLE_SPAWN] # Color of 'B'

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                if char == APPLE:
                    rgb_map[r, c, :] = grass_color
                else:
                    rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' '])
        return rgb_map

    def _map_to_colors_mask_waste(self) -> np.ndarray:
        """Generates RGB map masking Waste ('H') as River ('R')."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        river_color = DEFAULT_COLOURS[RIVER] # Color of 'R'

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                if char == WASTE:
                    rgb_map[r, c, :] = river_color
                else:
                    rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' '])
        return rgb_map


    def _get_agent_view(self, agent: CleanupAgent, full_rgb_map: np.ndarray) -> np.ndarray:
        """Extracts the agent's egocentric view from the full RGB map."""
        pos = agent.get_pos()
        view_size = CLEANUP_VIEW_SIZE
        padded_map = np.pad(full_rgb_map, ((VIEW_PADDING, VIEW_PADDING), (VIEW_PADDING, VIEW_PADDING), (0, 0)), mode='constant', constant_values=0)

        # Agent's position in the padded map
        padded_r, padded_c = pos[0] + VIEW_PADDING, pos[1] + VIEW_PADDING

        # Extract the square view centered on the agent
        view = padded_map[
            padded_r - view_size : padded_r + view_size + 1,
            padded_c - view_size : padded_c + view_size + 1,
            :
        ]

        # Rotate the view based on agent's orientation
        orientation = agent.get_orientation()
        if orientation == "UP":
            rotated_view = view
        elif orientation == "RIGHT": # 90 deg clockwise
            rotated_view = np.rot90(view, k=3)
        elif orientation == "DOWN": # 180 deg clockwise
            rotated_view = np.rot90(view, k=2)
        elif orientation == "LEFT": # 270 deg clockwise (or 90 counter-clockwise)
            rotated_view = np.rot90(view, k=1)
        else:
            rotated_view = view # Should not happen

        return rotated_view


    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Generates the observation for a specific agent, potentially masked by LLM command.
        """
        agent = self._agents[agent_id]
        command = self.llm_commands.get(agent_id) if self.use_llm else None

        # Determine which map rendering function to use
        if self.use_llm and command == "clean up":
            # Mask apples (show as grass 'B')
            map_rgb_for_view = self._map_to_colors_mask_apple()
        elif self.use_llm and command == "collect apples":
            # Mask waste (show as river 'R')
            map_rgb_for_view = self._map_to_colors_mask_waste()
        else:
            # Default: no masking or LLM not used
            map_rgb_for_view = self._map_to_colors()

        # Get the egocentric view from the chosen map
        agent_view_rgb = self._get_agent_view(agent, map_rgb_for_view)
        return agent_view_rgb


    def _rotate_vector(self, vector: np.ndarray, orientation: str) -> np.ndarray:
        """Rotates a move vector based on the agent's orientation."""
        # Check if the input vector is the 'STAY' action first
        if np.array_equal(vector, MOVE_ACTIONS["STAY"]):
            return np.array([0, 0])

        # If orientation is UP, no rotation is needed
        if orientation == "UP":
            return vector

        # For other orientations (RIGHT, DOWN, LEFT), calculate rotation
        orientation_vec = ORIENTATION_VECTORS[orientation]

        # Determine the relative move type
        if np.array_equal(vector, MOVE_ACTIONS["MOVE_UP"]): # Relative Forward
            return orientation_vec
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_DOWN"]): # Relative Backward
            return -orientation_vec
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_LEFT"]): # Relative Strafe Left
            return np.array([-orientation_vec[1], orientation_vec[0]]) # 
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_RIGHT"]): # Relative Strafe Right
            return np.array([orientation_vec[1], -orientation_vec[0]])
        else:
            # Should not happen if vector is a valid move action from MOVE_ACTIONS
            print(f"Warning: Unexpected move vector {vector} in _rotate_vector")
            return np.array([0, 0]) # Return STAY as a safe default


    def _is_position_valid(self, pos: np.ndarray) -> bool:
        """Checks if a position is within map bounds."""
        return 0 <= pos[0] < self.map_height and 0 <= pos[1] < self.map_width

    def _is_tile_walkable(self, pos: np.ndarray) -> bool:
        """Checks if the tile at the given position is walkable."""
        if not self._is_position_valid(pos):
            return False
        # Check against non-walkable tile types
        tile = self.world_map[pos[0], pos[1]]
        if tile in NON_WALKABLE:
            return False
        # Check if another agent is already there (this is handled in conflict resolution)
        return True

    def _resolve_movement_conflicts(self, intended_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Resolves conflicts where multiple agents intend to move to the same cell."""
        final_positions = {}
        agent_current_positions = {agent_id: agent.get_pos() for agent_id, agent in self._agents.items()}
        
        # Agents who didn't request a move stay put
        for agent_id, current_pos in agent_current_positions.items():
            if agent_id not in intended_positions:
                final_positions[agent_id] = current_pos

        # Identify conflicting moves
        target_cells = {} # { (r, c): [agent_id1, agent_id2, ...], ... }
        for agent_id, target_pos_tuple in intended_positions.items():
             target_pos = tuple(target_pos_tuple)
             # Check basic validity (bounds and non-walkable static tiles)
             if not self._is_position_valid(target_pos_tuple) or \
                self.world_map[target_pos[0], target_pos[1]] in NON_WALKABLE:
                 # Invalid move, agent stays put
                 final_positions[agent_id] = agent_current_positions[agent_id]
                 continue # Skip this agent

             if target_pos not in target_cells:
                 target_cells[target_pos] = []
             target_cells[target_pos].append(agent_id)

        processed_agents = set(final_positions.keys()) # Track agents whose moves are decided

        # Resolve conflicts for cells targeted by multiple agents
        for target_pos, agents_targeting in target_cells.items():
            if len(agents_targeting) > 1:
                # Conflict! Randomly choose one agent to succeed
                winner = random.choice(agents_targeting)
                final_positions[winner] = np.array(target_pos)
                processed_agents.add(winner)
                # Losers stay in their original positions
                for loser in agents_targeting:
                    if loser != winner:
                        final_positions[loser] = agent_current_positions[loser]
                        processed_agents.add(loser)

        # Process non-conflicting moves (agents targeting unique, valid cells)
        for target_pos, agents_targeting in target_cells.items():
            if len(agents_targeting) == 1:
                agent_id = agents_targeting[0]
                if agent_id not in processed_agents: # Ensure not already processed as loser
                     # Check for swap conflicts (A->B, B->A) - Simplified: just allow if cell is targeted by only one
                     # Check if target cell is occupied by another agent *that is also moving*
                     is_occupied_by_moving_agent = False
                     occupying_agent_id = None
                     for other_agent_id, other_current_pos in agent_current_positions.items():
                         if other_agent_id != agent_id and tuple(other_current_pos) == target_pos:
                             # Check if the occupying agent has an intended move
                             if other_agent_id in intended_positions:
                                  is_occupied_by_moving_agent = True
                             occupying_agent_id = other_agent_id
                             break

                     if not is_occupied_by_moving_agent:
                          final_positions[agent_id] = np.array(target_pos)
                     else:
                          # Occupied by an agent that is also trying to move. Prevent move.
                          # More complex resolution (like the original's loop) could be added here.
                          # For simplicity now, the mover stays put if target is occupied by *any* other agent.
                          is_occupied = False
                          for other_id, other_pos in agent_current_positions.items():
                              if other_id != agent_id and tuple(other_pos) == target_pos:
                                   is_occupied = True
                                   break
                          if not is_occupied:
                               final_positions[agent_id] = np.array(target_pos)
                          else:
                               final_positions[agent_id] = agent_current_positions[agent_id]

                     processed_agents.add(agent_id)


        # Ensure all agents have a final position assigned
        for agent_id in self.agents:
            if agent_id not in final_positions:
                 # This agent must have intended an invalid move or stayed put implicitly
                 final_positions[agent_id] = agent_current_positions[agent_id]

        return final_positions


    def _update_map_tile(self, row: int, col: int, char: bytes):
        """Updates a single tile on the world map."""
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            self.world_map[row, col] = char


    def _fire_beam(self, start_pos: np.ndarray, orientation: str, length: int,
                   beam_char: bytes, cell_types: list[bytes], update_char: list[bytes],
                   blocking_cells: list[bytes], beam_width: int) -> list[tuple[int, int, bytes]]:
        """Fires a beam, potentially hitting agents or changing tiles."""
        
        firing_direction = ORIENTATION_VECTORS[orientation]
        # Simplified: Assume beam width is 1 (originates from agent's front)
        # Original code had logic for width 3 starting slightly offset.
        # We'll start the beam from the cell directly in front.
        
        updates = [] # List of (row, col, new_char) for map updates

        # current_pos = start_pos + firing_direction

        if beam_width == 3:
            if orientation == "UP" or orientation == "DOWN":
                init_pos_all = [tuple(start_pos), tuple(start_pos + [0, 1]), tuple(start_pos + [0, -1])]
            elif orientation == "RIGHT" or orientation == "LEFT":
                init_pos_all = [tuple(start_pos), tuple(start_pos + [1, 0]), tuple(start_pos + [-1, 0])]
        elif beam_width == 1:
            init_pos_all = [tuple(start_pos)]

        agent_positions = {tuple(agent.get_pos()): agent_id for agent_id, agent in self._agents.items()}

        for pos in init_pos_all:
            current_pos = pos
            affect_num = CLEAN_BEAM_LENGTH_VALID
            for _ in range(length):
                # Move beam forward
                current_pos += firing_direction
                if not self._is_position_valid(current_pos):
                    break # Hit map boundary

                row, col = current_pos[0], current_pos[1]
                tile_char = self.world_map[row, col]

                # Add beam to render list
                self.beam_pos.append((row, col, beam_char))

                # Check if beam hits an agent
                if tuple(current_pos) in agent_positions:  
                    hit_agent_id = agent_positions[tuple(current_pos)]
                    if beam_char == PENALTY_BEAM:  # 说明是fire 不是clean
                        self._agents[hit_agent_id].add_reward(-PENALTY_HIT) #reward=0
                        self._agents[hit_agent_id].immobilize(IMMOBILIZE_DURATION_HIT)
                    # Beam stops when hitting an agent
                    break

                # Check if the tile blocks the beam
                if tile_char in blocking_cells:
                    break # Beam stops by wall

                # Check if the tile is affected by the beam (e.g., cleaning waste)
                if tile_char in cell_types:
                    try:
                        type_index = cell_types.index(tile_char)
                        new_char = update_char[type_index]
                        affect_num -= 1
                        updates.append((row, col, new_char)) # Record the change needed
                        # If cleaning waste, the beam might stop or continue based on rules
                        # Original 'CLEAN' had blocking_cells=[b'H'], so it stops here.
                        if beam_char == CLEAN_BEAM and tile_char == WASTE and affect_num == 0:
                            break
                    except (ValueError, IndexError):
                        # Should not happen if cell_types and update_char match
                        pass


        return updates

    def _compute_probabilities(self):
        """Computes the apple and waste spawn probabilities based on waste density."""
        current_waste = np.count_nonzero(self.world_map == WASTE)
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = current_waste / self.potential_waste_area

        if waste_density >= THRESHOLD_DEPLETION:
            self.current_apple_spawn_prob = 0
            if waste_density >= 0.55:
                self.current_waste_spawn_prob = 0
            
        else:
            self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY
            if waste_density <= THRESHOLD_RESTORATION:
                self.current_apple_spawn_prob = APPLE_RESPAWN_PROBABILITY
            else:
                # Linear interpolation between restoration and depletion thresholds 
                # 恢复阈值和耗尽阈值之间的线性插值
                prob = (1.0 - (waste_density - THRESHOLD_RESTORATION) / (THRESHOLD_DEPLETION - THRESHOLD_RESTORATION)) * APPLE_RESPAWN_PROBABILITY
                self.current_apple_spawn_prob = max(0, prob) # Ensure non-negative


    def _spawn_apples_and_waste(self) -> list[tuple[int, int, bytes]]:
        """Attempts to spawn apples and waste based on current probabilities."""
        spawn_updates = []
        agent_pos_list = [tuple(a.get_pos()) for a in self._agents.values()]

        # Try to spawn apples
        for r, c in self.apple_spawn_points:
            if (self.world_map[r, c] == APPLE_SPAWN or self.world_map[r, c] == EMPTY) and tuple([r, c]) not in agent_pos_list:
                if random.random() < self.current_apple_spawn_prob:
                    spawn_updates.append((r, c, APPLE))

        # Try to spawn waste 
        for r, c in self.waste_spawn_points: # waste_spawn_points includes original H and R locations
            if self.world_map[r, c] == EMPTY or self.world_map[r, c] == RIVER: # Can spawn on empty or river tiles
                 if tuple([r, c]) not in agent_pos_list:
                    if random.random() < self.current_waste_spawn_prob:
                        spawn_updates.append((r, c, WASTE))

        return spawn_updates

# --- PettingZoo AEC Wrapper --- (Optional but common)
# def env(**kwargs):
#     """Creates a PettingZoo AEC environment."""
#     parallel_env = CleanupEnv(**kwargs)
#     aec_env = parallel_to_aec(parallel_env)
#     #aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env) # Good for debugging
#     #aec_env = wrappers.OrderEnforcingWrapper(aec_env)   # Ensures order
#     return aec_env
# --- 新的定义 ---
def env(render_mode=None, **kwargs):
    """
    Creates a PettingZoo AEC environment.

    Args:
        render_mode: The rendering mode ('human', 'rgb_array', or None).
        **kwargs: Other arguments to pass to the CleanupEnv constructor
                  (e.g., num_agents, max_cycles).
    """
    # 将 render_mode 和其他 kwargs 传递给 CleanupEnv
    parallel_env = CleanupEnv(render_mode=render_mode, **kwargs)
    aec_env = parallel_to_aec(parallel_env)
    #aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env) # Good for debugging
    #aec_env = wrappers.OrderEnforcingWrapper(aec_env)   # Ensures order
    return aec_env
# --- 修改结束 ---


# # # --- Example Usage ---
# if __name__ == "__main__":
#     num_agents = 2
#     render_mode = "human" # "rgb_array" or None
#     env = CleanupEnv(num_agents=num_agents, render_mode=render_mode, max_cycles=500)
#     observations = env.reset()

#     for _ in range(env.max_cycles):
#         actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
#         observations, rewards, terminations, truncations, infos = env.step(actions)
#         #print(f"Step: {env.num_cycles}, Agents: {env.agents}, Rewards: {rewards}")

#         if not env.agents: # Episode ended (all terminated or truncated)
#             print("Episode Finished!")
#             observations = env.reset()

#     env.close()
#     print("Cleanup Env Example Done.")
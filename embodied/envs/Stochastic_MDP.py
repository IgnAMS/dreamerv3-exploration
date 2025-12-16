import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import embodied

class SimpleStochasticMDP(embodied.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        max_steps: int = 2_000, 
        hall_size: int = 5,
        N_TIPS: int = 3, 
        seed=None,
    ):
        super().__init__()
        assert hall_size >= 2, "hall size must be at least 2"
        assert N_TIPS >= 2, "N_TIPS must be at least 2"
        self.N_TIPS = N_TIPS
        self.hall_size = hall_size
        self.num_states = self.N_TIPS * self.hall_size + 1
        
        self.V = [
            i for i in range(self.N_TIPS * hall_size + 1)
        ]
        # 0 -> middle
        # [1, 2, ..., hall_size] first hall
        # [hall_size+1, hall_size+2, ..., 2 * hall_size] second hall
        # [2*hall_size+1, 2*hall_size+2, ..., 3 * hall_size] third hall
        # ...
        # [(N_TIPS-1) * hall_size+1, (N_TIPS-1) * hall_size+2, ..., N_TIPS * hall_size] # N hall
        
        self.Edges = {
            0: {
                # start of the hall
                # 0 -> 1, 1 -> hall_size+1, 1 -> 2*hall_size+1, ... 
                i: hall_size * i + 1 for i in range(self.N_TIPS)
            }, 
            **{
                # From the start of the hall you can go to 0 by executing 0
                # From the start of the hall you can advance in the hall by executing 1
                i * hall_size + 1: {
                    0: 0, 
                    1: i * hall_size + 2
                }
                for i in range(self.N_TIPS) 
            },
            **{
                # From the middle of the hall you can take one step back of the hall by executing 0
                # From the middle of the hall you can go to the next path of the hall by executing 1
                i: {
                    0: i - 1, 
                    1: i + 1
                }
                for hall in range(self.N_TIPS)
                for i in range(hall_size * hall + 2, hall_size * hall + hall_size)
            },
            **{
                # At the end of the hall you can take one step back of the hall by executing 0
                (hall + 1) * hall_size: {
                    0: (hall + 1) * hall_size - 1
                }
                for hall in range(self.N_TIPS)
            }
        }
        
        self.observation_space = spaces.Dict({
            "is_first": spaces.Discrete(2),
            "is_last": spaces.Discrete(2),
            "is_terminal": spaces.Discrete(2),
            "reward": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            "image": spaces.Box(low=0, high=1, shape=(self.num_states,), dtype=np.uint8),
        })

        self.max_degree = max(len(self.Edges[u]) for u in self.V)
        self.action_space = {
            "reset": spaces.Discrete(2),
            "action": spaces.Discrete(self.max_degree),
        }
        
        # estado interno
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.rng = random.Random(seed)
        self.state = 0
        self.cookie_pos = self._random_initial_cookie()
        self._first = True

    def _random_initial_cookie(self) -> int:
        ends = [(h + 1) * self.hall_size for h in range(self.N_TIPS)]
        return self.rng.choice(ends)

    def _respawn_cookie_excluding(self, excluded: int) -> int:
        ends = [(h + 1) * self.hall_size for h in range(self.N_TIPS)]
        choices = [e for e in ends if e != excluded]
        return self.rng.choice(choices)

    def _one_hot_state(self, s: int) -> np.ndarray:
        arr = np.zeros((self.num_states,), dtype=np.uint8)
        arr[s] = 1
        return arr

    @property
    def obs_space(self):
        return self._obs_space

    @obs_space.setter
    def obs_space(self, v):
        self._obs_space = v

    @property
    def act_space(self):
        return self._act_space

    @act_space.setter
    def act_space(self, v):
        self._act_space = v

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
            
        self.state = 0
        self.cookie_pos = self._random_initial_cookie()
        self.step_count = 0
        self._first = True
        obs = {
            "is_first": np.array(1, dtype=np.int8),
            "is_last": np.array(0, dtype=np.int8),
            "is_terminal": np.array(0, dtype=np.int8),
            "reward": np.array(0.0, dtype=np.float32),
            "image": self._one_hot_state(self.state),
        }
        return obs


    def step(self, action):
        if action["reset"]:
            return self.reset()
        
        action = int(action["action"])
        self._first = False
        self.step_count += 1
        reward = 0.0
        
        if action not in self.Edges[self.state]:
            self.state = self.state
        else:
            print(self.state, self.Edges[self.state][action])
            self.state = self.Edges[self.state][action]
            
        if self.state == self.cookie_pos:
            reward = 1
            self.cookie_pos = self._respawn_cookie_excluding(excluded=self.state)
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        is_last = 1 if truncated else 0
        
        obs = {
            "is_first": np.array(0, dtype=np.int8),
            "is_last": np.array(is_last, dtype=np.int8),
            "is_terminal": np.array(1 if terminated else 0, dtype=np.int8),
            "reward": np.array(float(reward), dtype=np.float32),
            "image": self._one_hot_state(self.state),
        }
        
        return obs
    
    def render(self, mode: str = "human"):
        # impresión sencilla del grafo
        center = "(C)"
        rows = []
        for h in range(self.N_TIPS):
            start = h * self.hall_size + 1
            nodes = [str(i) for i in range(start, (h + 1) * self.hall_size + 1)]
            # put markers
            nodes_marked = []
            for n in nodes:
                ni = int(n)
                mark = ""
                if ni == self.state:
                    mark += "A"
                if ni == self.cookie_pos:
                    mark += "G"
                nodes_marked.append(f"{n}{mark}")
            rows.append(" - ".join(nodes_marked))
            
        s = f"State center={self.state==0 and 'A' or ''} | " + " || ".join(rows)
        if mode == "human":
            print(s)
        else:
            return s

    def close(self):
        pass

if __name__ == "__main__":
    # python3 -m embodied.envs.Stochastic_MDP   
    env = SimpleStochasticMDP(max_steps=2000, hall_size=3)
    while True:
        env.render()
        tecla = int(input("ingrese número: "))
        obs = env.step(action={"action": tecla, "reset": False})
        
        
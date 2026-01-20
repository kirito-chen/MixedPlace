import numpy as np
from collections import deque


# Mock PerPromptStatTracker (与原论文源码一致)
class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {} # key: prompt string, value: deque of rewards

    def update(self, prompts, rewards):
        prompts = np.array(prompts) 
        rewards = np.array(rewards)
        unique_prompts = np.unique(prompts) 
        advantages = np.empty_like(rewards) 

        for prompt_str in unique_prompts:
            prompt_rewards_in_batch = rewards[prompts == prompt_str]

            if prompt_str not in self.stats:
                self.stats[prompt_str] = deque(maxlen=self.buffer_size)
            
            self.stats[prompt_str].extend(prompt_rewards_in_batch)

            if len(self.stats[prompt_str]) < self.min_count:
                # Fallback to global mean/std if not enough history for this prompt
                # Note: This version of `mean` and `std` is from the *current* batch,
                # which is how the original DDPO code does it.
                mean = np.mean(rewards) 
                std = np.std(rewards) + 1e-8
            else:
                mean = np.mean(self.stats[prompt_str])
                std = np.std(self.stats[prompt_str]) + 1e-8
            
            advantages[prompts == prompt_str] = (prompt_rewards_in_batch - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Solution():
    def __init__(self, split="small"):
        self.batch_size = 16
        self.split = split
        self.data, self.action_space = self.read_csv()

        self.Q = nn.ModuleList(
            nn.Linear(2, 15),
            nn.LeakyRELU(),
            nn.Linear(15, len(self.action_space)),
            nn.Softmax()
        )
        self.lr = 0.01
        self.discount_rate = 0.9

    def read_csv(self):
        data = np.read_text(f"data/{self.split}.csv", dtype=int)
        num_actions = 4
        if self.split == "medium":
            num_actions = 7
        elif self.split == "large":
            num_actions = 9
        action_space = torch.tensor(list(range(1, num_actions+1)))

        return data, action_space

    def choose_action(self, curr_state):
        epsilon = 0.1
        if np.rand() < epsilon:
            actions = np.random.choice(self.action_space)
            return actions

        curr_state = curr_state.repeat([len(self.action_space)])
        print(curr_state)
        inputs = torch.cat([curr_state, self.action_space], dim=1)
        print(inputs)
        values = self.Q(inputs)
        print(values)
        actions = torch.argmax(values, dim=1)
        print(actions)
        return actions

    def save_policy(self):
        all_poss_states = 0
        outputs = self.Q(all_poss_states)
        best_actions = torch.argmax(outputs, dim=1)
        


    def train(self):
        for s,a,r,sp in self.data:
            # choose actions
            next_action = self.choose_action(sp)

            # update parameters
            next_step_input = torch.cat([sp, next_action])
            curr_step_input = torch.cat([s, a])
            curr_step_output = self.Q(curr_step_input)
            update = r + self.discount_rate*self.Q(next_step_input)
            update = update - curr_step_output
            update = self.lr * update * torch.autograd.grad(self.Q.parameters, curr_step_output)
            self.Q.parameters += update




if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim

# Define the generative model
class GenerativeModel(nn.Module):
    def __init__(self, num_states, num_obs):
        super(GenerativeModel, self).__init__()
        self.state_transition = nn.Linear(num_states, num_states)
        self.obs_emission = nn.Linear(num_states, num_obs)

    def forward(self, prev_state):
        state = self.state_transition(prev_state)
        obs = self.obs_emission(state)
        return state, obs

# Define the recognition model
class RecognitionModel(nn.Module):
    def __init__(self, num_states, num_obs):
        super(RecognitionModel, self).__init__()
        self.obs_to_state = nn.Linear(num_obs, num_states)

    def forward(self, obs):
        state = self.obs_to_state(obs)
        return state

# Generate synthetic data
def generate_data(num_samples, num_states, num_obs):
    true_states = torch.randint(0, num_states, (num_samples,))
    obs_probs = torch.rand(num_states, num_obs)
    obs_probs /= obs_probs.sum(dim=1, keepdim=True)
    obs = torch.distributions.Categorical(probs=obs_probs[true_states]).sample()
    return true_states, obs

# Define the training loop
def train(generative_model, recognition_model, data, num_epochs, lr):
    optimizer = optim.Adam(list(generative_model.parameters()) + list(recognition_model.parameters()), lr=lr)

    for epoch in range(num_epochs):
        true_states, obs = data
        
        # Infer approximate posterior states using the recognition model
        approx_states = recognition_model(obs)
        
        # Compute the variational free energy
        state, pred_obs = generative_model(approx_states)
        log_likelihood = torch.distributions.Categorical(logits=pred_obs).log_prob(obs)
        kl_divergence = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=approx_states),
                                                          torch.distributions.Categorical(logits=state))
        free_energy = -log_likelihood + kl_divergence
        
        # Minimize the variational free energy
        optimizer.zero_grad()
        free_energy.mean().backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Free Energy: {free_energy.mean().item():.4f}")

# Set up the model and data
num_states = 5
num_obs = 10
num_samples = 1000
num_epochs = 10
lr = 0.01

generative_model = GenerativeModel(num_states, num_obs)
recognition_model = RecognitionModel(num_states, num_obs)
data = generate_data(num_samples, num_states, num_obs)

# Train the model
train(generative_model, recognition_model, data, num_epochs, lr)
import jax
import jax.numpy as jnp
import optax

# Define the generative model
def generative_model(prev_state, params):
    state = jnp.dot(prev_state, params['state_transition'])
    obs = jnp.dot(state, params['obs_emission'])
    return state, obs

# Define the recognition model
def recognition_model(obs, params):
    state = jnp.dot(obs, params['obs_to_state'])
    return state

# Generate synthetic data
def generate_data(num_samples, num_states, num_obs, key):
    true_states = jax.random.randint(key, (num_samples,), 0, num_states)
    obs_probs = jax.random.uniform(key, (num_states, num_obs))
    obs_probs /= obs_probs.sum(axis=1, keepdims=True)
    obs = jax.random.categorical(key, jnp.log(obs_probs[true_states]))
    return true_states, obs

# Define the loss function
def loss_fn(params, data):
    true_states, obs = data
    approx_states = recognition_model(obs, params)
    state, pred_obs = generative_model(approx_states, params)
    log_likelihood = jax.nn.log_softmax(pred_obs)[jnp.arange(obs.shape[0]), obs]
    kl_divergence = jnp.sum(jax.nn.softmax(approx_states) * (jax.nn.log_softmax(approx_states) - jax.nn.log_softmax(state)))
    free_energy = -log_likelihood + kl_divergence
    return jnp.mean(free_energy)

# Define the training loop
@jax.jit
def train_step(params, opt_state, data):
    loss, grads = jax.value_and_grad(loss_fn)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train(num_epochs, num_states, num_obs, num_samples, lr, key):
    init_fn = jax.nn.initializers.glorot_normal()
    params = {
        'state_transition': init_fn(key, (num_states, num_states)),
        'obs_emission': init_fn(key, (num_states, num_obs)),
        'obs_to_state': init_fn(key, (num_obs, num_states)),
    }
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    data = generate_data(num_samples, num_states, num_obs, key)
    
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state, data)
        print(f"Epoch [{epoch+1}/{num_epochs}], Free Energy: {loss:.4f}")
    
    return params

# Set up the model and data
num_states = 5
num_obs = 10
num_samples = 1000
num_epochs = 10
lr = 0.01
key = jax.random.PRNGKey(0)

# Train the model
params = train(num_epochs, num_states, num_obs, num_samples, lr, key)
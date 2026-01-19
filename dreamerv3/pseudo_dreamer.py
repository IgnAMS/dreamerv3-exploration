def WorldModel():
    pass
def PolicyNetwork():
    pass
def ValueNetwork():
    pass
def ReplayBuffer():
    pass
def make_env():
    pass
def reconstruction_loss():
    pass
def dynamics_loss():
    pass
def reward_loss():
    pass
def continuation_loss():
    pass
def lambda_return():
    pass
def log_prob():
    pass
def entropy_bonus():
    pass


num_envs = 1
TOTAL_ENV_STEPS = 1_000_000

world_model = WorldModel()      # encoder + RSSM + decoder
actor       = PolicyNetwork()
critic      = ValueNetwork()
replay      = ReplayBuffer()

step = 0
envs = [make_env() for _ in range(num_envs)]

# Hiperparámetros clave (orden de magnitud típico)
BATCH_SIZE      = 32
BATCH_LENGTH    = 64          # pasos reales por secuencia
IMAG_HORIZON    = 333         # pasos imaginados
TRAIN_RATIO     = 4           # updates por paso real


# ------------------------------------------------------------
# Loop principal (interacción real)
# ------------------------------------------------------------

while step < TOTAL_ENV_STEPS:

    # ========================================================
    # 1. Interacción con el mundo REAL
    # ========================================================

    for env in envs:
        obs = env.observe()
        action = actor(obs)            # policy entrenada SOLO en imaginación
        next_obs, reward, done = env.step(action)

        replay.store(
            obs=obs,
            action=action,
            reward=reward,
            done=done
        )

        step += 1


    # ========================================================
    # 2. Entrenamiento (solo si hay suficientes datos)
    # ========================================================

    if replay.size < BATCH_SIZE * BATCH_LENGTH:
        continue


    # --------------------------------------------------------
    # 🔁 2.1 Múltiples updates por paso real
    # --------------------------------------------------------

    for update in range(TRAIN_RATIO):

        # ====================================================
        # 2.2 Sampleo desde replay (datos REALES)
        # ====================================================

        batch = replay.sample(
            batch_size=BATCH_SIZE,
            sequence_length=BATCH_LENGTH
        )

        # Tamaño real del batch:
        #   32 secuencias × 64 pasos = 2048 pasos reales


        # ====================================================
        # 2.3 Entrenamiento del WORLD MODEL
        # ====================================================

        latent_states = world_model.encode(batch.obs)

        model_loss = (
            reconstruction_loss(batch.obs, latent_states)
          + dynamics_loss(latent_states, batch.actions)
          + reward_loss(latent_states, batch.rewards)
          + continuation_loss(latent_states, batch.dones)
        )

        world_model.optimize(model_loss)

        # 👉 Aprende dinámica, reward, termination
        # 👉 SOLO con datos reales


        # ====================================================
        # 2.4 IMAGINACIÓN (rollouts en el mundo latente)
        # ====================================================

        imagined_states = []
        state = latent_states[:, -1]   # último estado real

        for t in range(IMAG_HORIZON):
            action = actor(state)
            state = world_model.imagine_step(state, action)
            imagined_states.append(state)

        # Tamaño efectivo de imaginación:
        #   32 trayectorias × 333 pasos
        # = 10,656 pasos imaginados POR UPDATE


        # ====================================================
        # 2.5 Entrenamiento del CRITIC (value)
        # ====================================================

        returns = lambda_return(imagined_states)

        value_loss = critic.loss(imagined_states, returns)
        critic.optimize(value_loss)


        # ====================================================
        # 2.6 Entrenamiento del ACTOR (policy)
        # ====================================================

        advantages = returns - critic(imagined_states)

        policy_loss = (
            - advantages * log_prob(actor.actions)
            - entropy_bonus(actor)
        )

        actor.optimize(policy_loss)


# ------------------------------------------------------------
# Fin del entrenamiento
# ------------------------------------------------------------

import gymnasium as gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from gymnasium import spaces
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from configs import MODEL_SIZES

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- 1. Utilidades y Normalización ---
class RunningMeanStd:
    """Rastrea la media y la varianza para normalizar entradas."""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        x = x.to(device)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot

        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x):
        x = x.to(device)
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- 2. Redes Neuronales ---
class Encoder(nn.Module):
    def __init__(self, obs_shape, depth):
        super().__init__()
        C, H, W = obs_shape
        # Escalamos los filtros según el 'depth' del config
        d = depth 
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(C, d, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(d, d*2, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(d*2, d*2, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            self.out_dim = self.net(dummy).shape[1]

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions, size_cfg):
        super().__init__()
        hidden_dim = size_cfg["fc_units"]
        gru_hidden = size_cfg["gru_units"]
        self.encoder = Encoder(obs_shape, size_cfg["cnn_depth"])
        self.fc = layer_init(nn.Linear(self.encoder.out_dim, hidden_dim))
        
        # GRU compartida
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.gru_hidden = gru_hidden

        # Heads
        self.actor = layer_init(nn.Linear(gru_hidden, n_actions), std=0.01)
        self.critic_ext = layer_init(nn.Linear(gru_hidden, 1), std=1)
        self.critic_int = layer_init(nn.Linear(gru_hidden, 1), std=1)

    def get_initial_state(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def forward(self, x, h_state):
        # x: [Batch, Seq, C, H, W]
        B, S, C, H, W = x.shape
        x = x.reshape(B * S, C, H, W) 
        
        # Normalización básica de imagen (0-255 -> 0-1)
        x = x / 255.0 
        
        features = self.encoder(x)
        features = F.relu(self.fc(features))
        
        # Preparar para GRU [Batch, Seq, Features]
        features = features.view(B, S, -1)
        
        gru_out, new_h = self.gru(features, h_state)
        
        # Aplanar para las heads
        gru_out_flat = gru_out.reshape(B * S, -1)
        
        logits = self.actor(gru_out_flat)
        v_ext = self.critic_ext(gru_out_flat)
        v_int = self.critic_int(gru_out_flat)
        
        return logits, v_ext, v_int, new_h

class RNDModel(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        C, H, W = obs_shape
        
        # Arquitectura idéntica al paper
        def make_net():
            return nn.Sequential(
                layer_init(nn.Conv2d(C, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(self.feature_dim, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, out_dim))
            )

        # Calculamos dimensión de features intermedia
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            dummy_enc = nn.Sequential(
                nn.Conv2d(C, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.Flatten()
            )
            self.feature_dim = dummy_enc(dummy).shape[1]

        self.target = make_net()
        self.predictor = make_net()

        # Target fijo
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, next_obs_norm):
        # next_obs_norm ya debe venir normalizada y recortada por RunningMeanStd
        target_feature = self.target(next_obs_norm)
        predict_feature = self.predictor(next_obs_norm)
        return predict_feature, target_feature

# --- 3. El Agente PPO + RND ---

class PPO_RND_Agent:
    def __init__(self, envs, config):
        self.envs = envs
        self.obs_shape = envs.observation_space.shape[-3:] # (4, 1, H, W) -> (1, H, W)
        # self.n_actions = envs.action_space.n
        if hasattr(envs.action_space, 'n'):
            self.n_actions = envs.action_space.n
        else:
            # Para MultiDiscrete, tomamos el tamaño de la primera dimensión de acciones
            self.n_actions = envs.action_space.nvec[0]
        
        # Hiperparámetros
        self.n_steps = config['n_steps']
        self.n_envs = envs.num_envs
        self.lr = config['lr']
        self.gamma = config['gamma']
        self.gamma_int = config['gamma_int'] # Generalmente 0.99
        self.gae_lambda = config['gae_lambda']
        self.coef_int = config['coef_int'] # Beta (peso de recompensa intrinseca)
        self.clip_coef = 0.1
        self.ent_coef = 0.001
        
        # Modelos
        self.ac = ActorCritic(self.obs_shape, self.n_actions, config["size_cfg"]).to(device)
        self.rnd = RNDModel(self.obs_shape).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.ac.parameters(), 'lr': self.lr},
            {'params': self.rnd.predictor.parameters(), 'lr': self.lr}
        ])
        
        # Normalizadores
        self.obs_rms = RunningMeanStd(shape=(1, *self.obs_shape))
        self.reward_rms = RunningMeanStd(shape=()) # Para retorno intrinseco

        # Estado global
        self.global_step = 0
        self.obs = torch.tensor(envs.reset()[0], dtype=torch.float32, device=device)
        self.h_state = self.ac.get_initial_state(self.n_envs).to(device)

        print_model_summary(self.ac, "Actor-Critic (GRU)")
        print_model_summary(self.rnd, "RND (Target & Predictor)")

    def get_intrinsic_reward(self, next_obs):
        # 1. Normalizar observación para RND (crítico)
        # Se recorta a [-5, 5] como en el paper original
        self.obs_rms.update(next_obs)
        next_obs_norm = self.obs_rms.normalize(next_obs).clamp(-5, 5)
        
        # 2. Forward RND
        with torch.no_grad():
            pred, target = self.rnd(next_obs_norm)
            # Recompensa es el MSE (Mean Squared Error) por batch
            loss = (pred - target).pow(2).mean(dim=1)
        return loss.detach().unsqueeze(1), next_obs_norm # [n_envs, 1]

    def rollout(self):
        # Almacenamiento
        buffer = {
            'obs': [], 'actions': [], 'logprobs': [], 'rewards_ext': [], 'rewards_int': [],
            'dones': [], 'values_ext': [], 'values_int': [], 'h_states': []
        }
        
        for _ in range(self.n_steps):
            self.global_step += self.n_envs
            
            with torch.no_grad():
                # Nota: Pasamos obs [N_envs, 1, C, H, W]
                logits, v_ext, v_int, next_h = self.ac(self.obs.unsqueeze(1), self.h_state)
                dist = Categorical(logits=logits)
                action = dist.sample()

            # Ejecutar paso en el entorno
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Convertir a tensores
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            reward_ext = torch.tensor(reward, dtype=torch.float32).to(device).view(-1, 1)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device).view(-1, 1)

            # --- RND: Recompensa Intrínseca ---
            # 1. Calculamos el error de predicción (MSE)
            intrinsic_reward, next_obs_norm_rnd = self.get_intrinsic_reward(next_obs_tensor)
            
            # 2. Normalización de recompensa intrínseca (Crucial para estabilidad)
            # RND paper: Normalizar dividiendo por la std dev de los retornos rodantes.
            # Aquí usamos una aproximación actualizando con el batch actual.
            self.reward_rms.update(intrinsic_reward.detach())
            intrinsic_reward = intrinsic_reward / torch.sqrt(self.reward_rms.var + 1e-8)

            # Almacenar en buffer
            buffer['obs'].append(self.obs)
            buffer['actions'].append(action)
            buffer['logprobs'].append(dist.log_prob(action))
            buffer['rewards_ext'].append(reward_ext)
            buffer['rewards_int'].append(intrinsic_reward)
            buffer['dones'].append(done_tensor)
            buffer['values_ext'].append(v_ext.detach())
            buffer['values_int'].append(v_int.detach())
            buffer['h_states'].append(self.h_state) # Guardamos el estado ANTES del step

            # Actualizar estado global
            self.obs = next_obs_tensor
            # Manejo de estado oculto: si done=1, reseteamos h a 0 para ese env
            self.h_state = next_h * (1 - done_tensor.view(1, -1, 1))

        # --- Bootstrapping (Valor del último estado) ---
        with torch.no_grad():
            _, next_val_ext, next_val_int, _ = self.ac(self.obs.unsqueeze(1), self.h_state)
            
        return buffer, next_val_ext, next_val_int

    def compute_advantages(self, buffer, next_val_ext, next_val_int):
        """Calcula GAE independientemente para stream extrínseco e intrínseco."""
        # Convertir listas a tensores: [Time, Batch, ...]
        rewards_ext = torch.stack(buffer['rewards_ext'])
        rewards_int = torch.stack(buffer['rewards_int'])
        values_ext = torch.stack(buffer['values_ext'])
        values_int = torch.stack(buffer['values_int'])
        dones = torch.stack(buffer['dones'])
        
        advantages_ext = torch.zeros_like(rewards_ext).to(device)
        advantages_int = torch.zeros_like(rewards_int).to(device)
        
        lastgaelam_ext = 0
        lastgaelam_int = 0
        
        # Bucle inverso para GAE
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues_e = next_val_ext
                nextvalues_i = next_val_int
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues_e = values_ext[t + 1]
                nextvalues_i = values_int[t + 1]

            # GAE Extrínseco
            delta_ext = rewards_ext[t] + self.gamma * nextvalues_e * nextnonterminal - values_ext[t]
            advantages_ext[t] = lastgaelam_ext = delta_ext + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam_ext
            
            # GAE Intrínseco (Gamma diferente usualmente)
            delta_int = rewards_int[t] + self.gamma_int * nextvalues_i * nextnonterminal - values_int[t]
            advantages_int[t] = lastgaelam_int = delta_int + self.gamma_int * self.gae_lambda * nextnonterminal * lastgaelam_int

        returns_ext = advantages_ext + values_ext
        returns_int = advantages_int + values_int
        
        return returns_ext, returns_int, advantages_ext, advantages_int

    def update(self, buffer, next_val_ext, next_val_int):
        # 1. Calcular ventajas y retornos
        ret_ext, ret_int, adv_ext, adv_int = self.compute_advantages(buffer, next_val_ext, next_val_int)
        
        # Combinar ventajas: A_total = A_ext + beta * A_int
        advantages = adv_ext + self.coef_int * adv_int
        
        # Aplanar tensores para PPO [Time * Batch, ...]
        # Nota: Para RNN, mantenemos estructura secuencial si usamos BPTT, 
        # pero aquí usaremos el estado oculto guardado en cada paso para simplificar (Burn-in strategy es mejor pero más compleja).
        
        # b_obs = torch.stack(buffer['obs']).reshape((-1,) + self.obs_shape)
        b_obs = torch.stack(buffer['obs']).reshape((-1,) + self.obs_shape)
        
        b_logprobs = torch.stack(buffer['logprobs']).reshape(-1)
        b_actions = torch.stack(buffer['actions']).reshape(-1)
        # b_h_states = torch.stack(buffer['h_states']).reshape(-1, self.n_envs, self.ac.gru_hidden)
        b_h_states = torch.stack(buffer['h_states']).reshape(-1, self.ac.gru_hidden)
        b_rewards_int = torch.stack(buffer['rewards_int']).reshape(-1)
        avg_intrinsic_reward = b_rewards_int.mean().item()
        b_advantages = advantages.reshape(-1)
        b_ret_ext = ret_ext.reshape(-1)
        b_ret_int = ret_int.reshape(-1)
        
        # Normalizar obs para RND Update (volvemos a normalizar el batch completo para consistencia)
        # Recortar obs a [-5, 5] es estándar en RND
        b_obs_norm_rnd = self.obs_rms.normalize(b_obs).clamp(-5, 5)

        # Optimization epochs
        batch_size = self.n_steps * self.n_envs
        inds = np.arange(batch_size)
        
        for _ in range(4): # 4 épocas de PPO
            np.random.shuffle(inds)
            # Minibatch loop (simplified)
            # Para GRU real, idealmente procesaríamos secuencias completas, 
            # pero aquí usamos el h_state guardado en cada paso.
            
            minibatch_size = batch_size // 4
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]
                
                # Reconstruir datos del minibatch
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_h = b_h_states[mb_inds] # [Minibatch, N_envs, Hidden] -> Necesitamos ajustar dimensión
                
                # OJO: Con GRU y shuffling aleatorio, el hidden state no es contiguo.
                # Para simplificar este script educativo, haremos el forward pass 
                # tratando cada sample como inicio de secuencia (seq_len=1).
                # En producción, se usan secuencias de largo L (ej. 8 pasos).
                
                # Forward Pass ActorCritic
                # mb_obs needs [Minibatch, Seq=1, C, H, W]
                # mb_h needs [1, Minibatch, Hidden] -> Flatten env dim
                
                # Truco para GRU sin secuencias largas en PPO:
                # Usamos el hidden state almacenado y pasamos seq_len=1
                # mb_h original shape: [Minibatch_size (mezclado de time y env), 1 (layer), Hidden]
                mb_h_in = mb_h.view(1, minibatch_size, -1) 
                
                new_logits, new_v_ext, new_v_int, _ = self.ac(mb_obs.unsqueeze(1), mb_h_in)
                
                new_v_ext = new_v_ext.view(-1)
                new_v_int = new_v_int.view(-1)
                
                new_dist = Categorical(logits=new_logits)
                new_logprob = new_dist.log_prob(mb_actions)
                entropy = new_dist.entropy().mean()
                
                # --- PPO Losses ---
                ratio = torch.exp(new_logprob - b_logprobs[mb_inds])
                mb_adv = b_advantages[mb_inds]
                # Normalizar ventajas del minibatch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss_ext = 0.5 * ((new_v_ext - b_ret_ext[mb_inds]) ** 2).mean()
                v_loss_int = 0.5 * ((new_v_int - b_ret_int[mb_inds]) ** 2).mean()
                v_loss = v_loss_ext + v_loss_int
                
                # --- RND Loss (Distillation) ---
                # Entrenamos el predictor para imitar al target en las observaciones visitadas
                pred_feat, target_feat = self.rnd(b_obs_norm_rnd[mb_inds])
                rnd_loss = (pred_feat - target_feat.detach()).pow(2).mean()
                
                # --- Total Loss ---
                # RND mask: A veces se enmascara una proporción de experiencias, aquí usamos todo
                loss = pg_loss - self.ent_coef * entropy + 0.5 * v_loss + rnd_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), 0.5)
                self.optimizer.step()

        return loss.item(), avg_intrinsic_reward


def plot_learning_curve(ext_history, int_history):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eje para Recompensa Extrínseca (El éxito en el juego)
    color = 'tab:blue'
    ax1.set_xlabel('Iteración de Entrenamiento')
    ax1.set_ylabel('Reward Extrínseco (Media 10 eps)', color=color)
    ax1.plot(ext_history, color=color, label='Extrinsic Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    # Eje para Recompensa Intrínseca (La curiosidad)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Reward Intrínseco (Curiosidad)', color=color)
    ax2.plot(int_history, color=color, alpha=0.3, label='Intrinsic Reward (RND)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Progreso de PPO + RND en MiniGrid')
    fig.tight_layout()
    plt.savefig("entrenamiento_ppo_rnd.png")
    plt.show()
    
def print_model_summary(model, model_name="Model"):
    print(f"\n{'='*20} {model_name} Summary {'='*20}")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
        # Imprime cada capa, su forma y cantidad de parámetros
        print(f"Layer: {name:35} | Shape: {str(list(param.shape)):20} | Params: {params:,}")
    
    print(f"{'-'*75}")
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    # Cálculo aproximado en MB (cada parámetro float32 ocupa 4 bytes)
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB")
    print(f"{'='*75}\n")

# --- 4. Loop Principal ---

def main(config):
    # Crear entorno (vectorizado)
    def make_env():
        env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode="rgb_array")
        env = RGBImgObsWrapper(env) # Necesitamos pixeles para la CNN
        env = ImgObsWrapper(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env) # (H, W) -> (1, H, W)
        new_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.expand_dims(obs, axis=0), 
            observation_space=new_space # <--- Aquí está el fix
        )
        
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(config["n_envs"])])
    
    agent = PPO_RND_Agent(envs, config)
    extrinsic_rewards_history = []
    intrinsic_rewards_history = []
    episode_rewards = np.zeros(config["n_envs"]) # Para acumular reward por cada env
    final_rewards_list = [] # Solo rewards de episodios terminados
    
    
    print("Iniciando entrenamiento PPO + RND...")
    for iteration in range(config["iters"]):
        # 1. Colectar datos
        buffer, next_val_ext, next_val_int = agent.rollout()
        
        # 2. Actualizar redes
        loss, avg_int_reward = agent.update(buffer, next_val_ext, next_val_int)
        
        batch_rewards = torch.stack(buffer['rewards_ext']).cpu().numpy()
        batch_dones = torch.stack(buffer['dones']).cpu().numpy()
        for t in range(config['n_steps']):
            episode_rewards += batch_rewards[t].flatten()
            for idx, done in enumerate(batch_dones[t].flatten()):
                if done:
                    final_rewards_list.append(episode_rewards[idx])
                    episode_rewards[idx] = 0
        # Guardar promedio de los últimos episodios para graficar
        if len(final_rewards_list) > 0:
            extrinsic_rewards_history.append(np.mean(final_rewards_list[-10:]))
        else:
            extrinsic_rewards_history.append(0)
        
        intrinsic_rewards_history.append(avg_int_reward)
        
        if iteration % 10 == 0:
            print(f"Iter: {iteration} | Loss: {loss:.3f} | Avg Intrinsic Reward: {avg_int_reward:.5f}")
    
    plot_learning_curve(extrinsic_rewards_history, intrinsic_rewards_history)
    envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO + RND Training")
    
    # --- Parámetros de Ejecución ---
    parser.add_argument("--iters", type=int, default=1000, help="Número de actualizaciones (rollouts)")
    parser.add_argument("--n_envs", type=int, default=4, help="Entornos en paralelo")
    parser.add_argument("--n_steps", type=int, default=128, help="Pasos por rollout")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    # --- Hiperparámetros RL ---
    parser.add_argument("--size", type=str, default="small", choices=["tiny", "small", "large"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento extrínseco")
    parser.add_argument("--gamma_int", type=float, default=0.99, help="Factor de descuento intrínseco")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda para GAE")
    parser.add_argument("--coef_int", type=float, default=0.1, help="Peso de la curiosidad (Beta)")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="Coeficiente de Entropía")

    args = parser.parse_args()
    config = vars(args)
    config["size_cfg"] = MODEL_SIZES[config["size"]]    
    
    
    main(config)
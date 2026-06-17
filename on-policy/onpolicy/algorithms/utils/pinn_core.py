import torch
from torch import nn
from torch.autograd import Variable

# ================= 基础多层感知机与注意力机制组件 =================
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class MLP2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class Attention_LEMURS(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()
        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)

    def forward(self, x):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)
        Q = self.activation_swish(torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))
        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)
        
        Q = self.activation_swish(torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))
        return self.mlp_out(x.mean(dim=1)).reshape(-1, self.na, self.output_dim)

class Att_R(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()
        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na
        # 参数初始化
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        
        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)
        self.register_buffer('_eye2', torch.eye(2, device=device))

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)
        Q = self.activation_swish(torch.bmm(self.Aq_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))
        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)
        Q = self.activation_swish(torch.bmm(self.Aq_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))
        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)
        batch = int(x.shape[0] / x.shape[2])
        j12_raw = x.sum(1).sum(1).reshape(batch, self.na)
        j12 = torch.abs(j12_raw) + 0.01
        j21 = -j12
        J12 = torch.diag_embed(j12)
        J21 = torch.diag_embed(j21)
        zeros = torch.zeros_like(J12)
        J = torch.cat((torch.cat((zeros, J21), dim=1), torch.cat((J12, zeros), dim=1)), dim=2)
        J_expanded = J.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        eye_pattern = self._eye2.unsqueeze(0).expand(J.shape[0], -1, -1)
        eye_tiled = eye_pattern.repeat(1, J.shape[1], J.shape[2])
        return J_expanded * eye_tiled

class Att_J(Att_R):
    pass # 结构完全一致，可复用 Att_R 的代码

class Att_H(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()
        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)

    def forward(self, x, na):
        self.na = na
        x = self.mlp_in(x).unsqueeze(dim=1)
        Q = self.activation_swish(torch.bmm(self.Aq_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_4.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))
        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1)
        Q = self.activation_swish(torch.bmm(self.Aq_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(torch.bmm(self.Ak_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(torch.bmm(self.Av_7.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(0).expand(x.shape[0], -1, -1))
        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))
        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).unsqueeze(dim=1).transpose(1, 2)

        x_sq = x ** 2
        M11 = x_sq[:, 0:5, :].sum(1).repeat_interleave(2, dim=-1)
        M12 = x_sq[:, 5:10, :].sum(1).repeat_interleave(2, dim=-1)
        M21 = x_sq[:, 10:15, :].sum(1).repeat_interleave(2, dim=-1)
        M22 = x_sq[:, 15:20, :].sum(1).repeat_interleave(2, dim=-1)
        Mpp = x_sq[:, 20:25, :].sum(1)

        Mupper11 = torch.diag_embed(M11)
        Mupper12 = torch.diag_embed(M12)
        Mupper21 = torch.diag_embed(M21)
        Mupper22 = torch.diag_embed(M22)
        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)
        q = x[:, :4, :]
        return torch.bmm(q.transpose(1, 2), torch.bmm(M, q)).sum(2) + Mpp.sum(1).unsqueeze(1)

class SoftBarrierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.mlp_shared = MLP(input_dim, [hidden_dim, hidden_dim]).to(device)
        self.mlp_k = MLP(2 * hidden_dim, [hidden_dim, 1]).to(device)
        self.log_smoothness = nn.Parameter(torch.tensor(0.0, device=device))
        self.softplus = nn.Softplus(beta=1.0)
    def forward(self, x, adj):
        b, n, d = x.shape
        z = self.mlp_shared(x)
        z_i = z.unsqueeze(2).expand(-1, -1, n, -1)
        z_j = z.unsqueeze(1).expand(-1, n, -1, -1)
        z_combined = torch.cat([z_i, z_j], dim=-1)
        k_ij_raw = torch.clamp(self.mlp_k(z_combined).squeeze(-1), min=-10.0, max=10.0)
        smoothness = self.softplus(self.log_smoothness) + 0.1
        k_ij = torch.clamp(self.softplus(k_ij_raw) * smoothness, min=0.0, max=10.0)
        return k_ij * adj

# ================= Safe PINN 核心重构（去除 TorchRL/TensorDict 依赖） =================
class SafePinnPPOActorCore(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agents, device, scenario_name="vrx_navigation_obs"):
        super().__init__()
        self.observation_dim_per_agent = obs_dim
        self.action_dim_per_agent = action_dim
        self.n_agents = n_agents # 用于自适应船只数量(2艘或3艘)
        self.device = device
        self.scenario_name = scenario_name
        # self.r_communication, self.agent_collision_radius = 0.45, 0.17
        # self.barrier_epsilon, self.f_max = 0.06, 0.8
        
        # self.use_lidar_barrier = True
        # self.lidar_start_idx, self.n_lidar_rays, self.lidar_max_range = 6, 12, 0.35
        # self.obstacle_barrier_weight = 0.45
        
        # self.task_weight, self.barrier_weight, self.barrier_weight_max = 1.3, 0.12, 0.20
        # self.barrier_warmup_steps, self.barrier_decay_start, self.barrier_decay_rate = 200, 400, 0.50
        # self.drag = 0.25
        self.r_communication = 50
        # 两船中心点之间的安全距离阈值。环境里另一艘船半径约 4m，表面安全余量约 5m，所以中心距取 9m。
        self.agent_collision_radius = 12.0
        # 船到静态障碍物/其他船表面的安全距离阈值，对应观测 [6:18] 的 range sector 距离。
        self.obstacle_safe_distance = 8.0
        self.barrier_epsilon, self.f_max = 0.1, 5
        
        self.use_lidar_barrier = True
        self.lidar_start_idx, self.n_lidar_rays, self.lidar_max_range = 6, 12, 50
        self.obstacle_barrier_weight = 0.65
        
        self.task_weight, self.barrier_weight, self.barrier_weight_max = 1.3, 0.2, 0.4
        self.barrier_warmup_steps, self.barrier_decay_start, self.barrier_decay_rate = 2000, 4000, 0.80
        self.drag = 0.25
        self.register_buffer('_training_steps', torch.tensor(0, dtype=torch.long))

        self.R_mean = Att_R(obs_dim, 16, 8, obs_dim, scenario_name, device).to(device)
        self.J_mean = Att_J(obs_dim, 16, 8, obs_dim, scenario_name, device).to(device)
        self.H_task = Att_H(obs_dim, 25, 8, obs_dim, device).to(device)
        self.H_barrier_head = SoftBarrierHead(obs_dim, 16, device).to(device)
        self.std_net = Attention_LEMURS(obs_dim + action_dim, action_dim, obs_dim, n_agents, device).to(device)
        
        dim = self.action_dim_per_agent * self.n_agents
        zeros, eye = torch.zeros(dim, dim, device=device), torch.eye(dim, device=device)
        self.F_sys_pinv = torch.cat((zeros, eye), dim=1)
        self.J_sys = torch.cat((torch.cat((zeros, eye), dim=1), torch.cat((-eye, zeros), dim=1)), dim=0)
        self.R_sys = torch.cat((torch.cat((zeros, zeros), dim=1), torch.cat((zeros, self.drag * eye), dim=1)), dim=0)

    def laplacian(self, q_agents):
        Q = torch.cdist(q_agents, q_agents, p=2)
        L = Q.le(self.r_communication).float()
        return L * torch.sigmoid(-(2.0) * (Q - self.r_communication))

    def _get_barrier_weight(self):
        steps = self._training_steps.float()
        if steps < self.barrier_warmup_steps:
            progress = steps / max(1.0, self.barrier_warmup_steps)
            return self.barrier_weight_max * 0.5 * (1.0 - torch.cos(progress * 3.14159))
        elif steps < self.barrier_decay_start:
            return self.barrier_weight_max
        else:
            decay_progress = torch.clamp((steps - self.barrier_decay_start) / 500.0, max=1.0)
            target = self.barrier_weight * self.barrier_decay_rate
            return target + (self.barrier_weight_max - target) * 0.5 * (1.0 + torch.cos(decay_progress * 3.14159))

    def forward(self, obs):
        if self.training: self._training_steps += 1
        batch_size = obs.shape[0]
        state = obs
        state_h_mean = torch.clone(state).reshape(-1, self.observation_dim_per_agent)

        F_sys_pinv = self.F_sys_pinv.unsqueeze(0).expand(batch_size, -1, -1)
        J_sys = self.J_sys.unsqueeze(0).expand(batch_size, -1, -1)
        R_sys = self.R_sys.unsqueeze(0).expand(batch_size, -1, -1)

        laplacian_base = self.laplacian(state[:, :, 0:2])
        laplacian = laplacian_base.unsqueeze(-1).repeat_interleave(self.observation_dim_per_agent, dim=-1)
        laplacian = laplacian.reshape(-1, self.n_agents, self.observation_dim_per_agent)

        state_expanded = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(-1, self.n_agents, self.observation_dim_per_agent)
        state_masked = laplacian * state_expanded
        std_input = state_masked.detach().clone()

        R_mean = self.R_mean(state_masked.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean(state_masked.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            state_batch = state_h_mean.reshape(batch_size, self.n_agents, -1)
            H_task_sum = self.H_task(state_h_mean.to(torch.float32), self.n_agents).sum()
            
            q_batch = state_batch[:, :, 0:2]
            dist = torch.sqrt(torch.sum((q_batch.unsqueeze(2) - q_batch.unsqueeze(1))**2, dim=-1) + 1e-6)
            k_ij = self.H_barrier_head(state_batch, laplacian_base)
            mask = laplacian_base * (1 - torch.eye(self.n_agents, device=self.device).unsqueeze(0))
            
            agent_gap = torch.clamp(
                dist - self.agent_collision_radius,
                min=max(self.barrier_epsilon, 0.02),
            )
            agent_ratio = torch.clamp(
                agent_gap / (self.agent_collision_radius + self.barrier_epsilon),
                min=0.01,
                max=100.0,
            )
            H_barrier_ij = torch.clamp(k_ij * torch.nn.functional.softplus(torch.clamp(-torch.log(agent_ratio), min=-20.0, max=20.0), beta=2.0) * mask, min=0.0, max=50.0)
            H_barrier_agent_sum = (H_barrier_ij.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).sum()

            H_barrier_obs_sum = torch.tensor(0.0, device=self.device)
            if self.use_lidar_barrier and self.observation_dim_per_agent > self.lidar_start_idx:
                lidar_end_idx = min(self.lidar_start_idx + self.n_lidar_rays, self.observation_dim_per_agent)
                if lidar_end_idx > self.lidar_start_idx:
                    lidar_data = state_batch[:, :, self.lidar_start_idx:lidar_end_idx]
                    safe_dist = torch.clamp(
                        torch.clamp(lidar_data, min=0.01, max=self.lidar_max_range) - self.obstacle_safe_distance,
                        min=self.barrier_epsilon,
                    )
                    H_barrier_obs_per_ray = torch.nn.functional.softplus(torch.clamp(-torch.log(torch.clamp(safe_dist / (self.obstacle_safe_distance + self.barrier_epsilon), min=0.01, max=10.0)), min=-20.0, max=20.0), beta=3.0)
                    H_barrier_obs_sum = torch.clamp(H_barrier_obs_per_ray.sum() * self.obstacle_barrier_weight, min=0.0, max=50.0)
            
            H_kin_sum = 0.5 * torch.sum(state_batch[:, :, 2:4]**2)
            H_goal_sum = 0.5 * torch.sum((q_batch - (q_batch - state_batch[:, :, 4:6]).detach())**2, dim=-1).sum() * 10.0
            H_total = self.task_weight * (H_goal_sum + H_task_sum + H_kin_sum) + self._get_barrier_weight() * (H_barrier_agent_sum + H_barrier_obs_sum)
            
            grad_H_total = torch.autograd.grad(H_total, state_h_mean, only_inputs=True, create_graph=self.training)[0]
            dH_mean_combined = torch.clamp(torch.nan_to_num(grad_H_total, nan=0.0, posinf=1.0, neginf=-1.0), min=-self.f_max, max=self.f_max)
            
        dHdx_mean = torch.cat((dH_mean_combined[:, :self.action_dim_per_agent].reshape(-1, self.n_agents * self.action_dim_per_agent),
                               dH_mean_combined[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1, self.n_agents * self.action_dim_per_agent)), dim=1)
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)
        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2), dx_mean[:, :self.action_dim_per_agent * self.n_agents].unsqueeze(dim=2)), dim=1)
        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)
        u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, 1, u_mean.shape[2]).expand(-1, self.n_agents, -1)), dim=2))

        # return torch.nan_to_num(torch.clamp(u_mean, min=-10.0, max=10.0), nan=0.0), torch.exp(u_log_std)
        # 强制钳制 log_std，防止 torch.exp() 运算后产生 inf 和 NaN
        u_log_std_clamped = torch.clamp(torch.nan_to_num(u_log_std, nan=-5.0), min=-20.0, max=0.5)
        
        return torch.nan_to_num(torch.clamp(u_mean, min=-10.0, max=10.0), nan=0.0), torch.exp(u_log_std_clamped)

class PinnActorCore(SafePinnPPOActorCore):
    # 这里为了简便，你可以直接用 SafePinnPPOActorCore 替代普通 PINN，或者在此实现去安全屏障的基础 PINN
    pass

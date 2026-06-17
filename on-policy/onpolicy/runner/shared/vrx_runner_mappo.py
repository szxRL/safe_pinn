import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    """辅助函数：将 PyTorch Tensor 转换为 Numpy 数组"""
    return x.detach().cpu().numpy()

class VRXRunner(Runner):
    """
    VRX 多智能体无船 (USV) 导航的 Runner。
    继承自 on-policy 的基础 Runner 类，处理环境交互、数据收集与模型更新。
    """
    def __init__(self, config):
        super(VRXRunner, self).__init__(config)

    def run(self):
        # 1. 环境预热，获取初始观测值
        print(">>> 获取初始观测值...")
        self.warmup()   
        print(">>> 预热环境...")

        start = time.time()
        
        # 计算总回合数
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            print("第 {} 回合开始...".format(episode + 1))
            # 学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 在一个 Episode 内循环收集数据
            for step in range(self.episode_length):
                # 1. 网络推断：根据当前观测采用动作
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # 2. 与环境交互：执行动作，获取下一步状态和奖励
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # 3. 数据打包存入缓存 (Replay Buffer)
                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            # 4. 一个回合结束，计算优势函数 (Advantage) 并更新网络 (PPO Update)
            self.compute()
            train_infos = self.train()

            # 5. 后处理与日志记录
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # 保存模型
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # 打印与记录 Tensorboard
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}."
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                # 计算并记录本回合的平均团队奖励 (Team Reward)
                # 之前我们将 reward 放缩到了 +/- 15 左右，这里将其乘以 episode_length 得到回合总分
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print(">>> 当前回合平均奖励 (Average Episode Reward): {:.2f}".format(train_infos["average_episode_rewards"]))
                
                # 将训练指标写入 TensorBoard
                self.log_train(train_infos, total_num_steps)

    def warmup(self):
        """预热函数：初始化环境并存入第一帧状态"""
        # 复位环境
        obs, share_obs, available_actions = self.envs.reset()
        print(">>> 环境已复位！开始收集数据...")

        # 如果没有开启中心化 Critic，就把局部观测当成全局观测
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def collect(self, step):
        """收集函数：调用 Actor-Critic 网络获取动作"""
        self.trainer.prep_rollout()
        
        # 核心：将所有并行环境、所有智能体的数据展平传入网络
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
            
        # 拆分回 [envs, agents, dim] 的形状
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        
        # 将动作传递给环境 (连续动作空间直接使用)
        actions_env = actions

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def _get_agent_info(self, infos, env_id, agent_id):
        env_infos = infos[env_id]
        if isinstance(env_infos, np.ndarray):
            env_infos = env_infos.tolist()
        if isinstance(env_infos, (list, tuple)):
            return env_infos[agent_id]
        if isinstance(env_infos, dict):
            return env_infos
        return {}

    def _build_active_masks(self, infos, dones):
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for env_id in range(self.n_rollout_threads):
            if np.all(dones[env_id]):
                active_masks[env_id] = 1.0
                continue

            for agent_id in range(self.num_agents):
                agent_info = self._get_agent_info(infos, env_id, agent_id)
                if agent_info.get("policy_active") is False or agent_info.get("station_keeping_active", False):
                    active_masks[env_id, agent_id, 0] = 0.0

        return active_masks

    def insert(self, data):
        """插入函数：处理数据并放入 Buffer，特别处理 RNN 状态的截断"""
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # 【重点】如果环境触发了 Done，RNN 的记忆必须被清空，防止把上一局的记忆带到下一局
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        # Mask 用于告诉网络该步是否是有效的转换 (Done 时 Mask 为 0)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks = self._build_active_masks(infos, dones)

        if not self.use_centralized_V:
            share_obs = obs

        # 将整理好的数据插入底层 Buffer
        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks=active_masks,
        )

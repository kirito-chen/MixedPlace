import torch

class CosineScheduler():
    def __init__(self, clip = False, _eta = 0.3):
        self.clip = clip
        self.timesteps = None
        self._eta = _eta

    def add_noise(self, input, eps, t):
        # takes:
        # clean input (B, ...) 
        # eps (B, ...)
        # t (B,)
        # returns noised input at timestep t
        B, = t.shape
        input_dims = len(input.shape[1:])
        alpha = self.alpha(t).view((B, *([1] * input_dims)))
        sigma = self.sigma(t).view((B, *([1] * input_dims)))
        output = input * alpha + eps * sigma
        return output
    
    def set_timesteps(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(1-1e-4, 1e-4, num_timesteps+1)


    def step(
            self, 
            eps_prediction,  # 模型输出的噪声预测 
            t,               # 当前时间步和前一时间步
            t_minus_one, 
            xt,             # 当前样本 
            z,              # 额外噪声，用于随机采样
            mask = None, # where mask is true, predicted x0 does not change from sample
            guidance_fn = None,
            ):
        # eps_prediction: epsilon_hat (B, ...)
        # t: float
        # t_minus_one: float
        # xt: (B, ...)
        # guidance function: x_hat -> guidance
        alpha_t_minus_one = self.alpha(t_minus_one)
        alpha_t = self.alpha(t)
        sigma_t_minus_one = self.sigma(t_minus_one)
        sigma_t = self.sigma(t)
        eta = self.eta(t, t_minus_one)
        # (xt - sigma_t * eps_prediction) / alpha_t 是模型对 x0 的预测
        predicted_x0 = (xt - sigma_t * eps_prediction)/alpha_t   # 是add_noise的反向，xt = alpha_t * x_0 + eps_pre * sigma_t

        # apply mask
        predicted_x0 = torch.where(mask, xt, predicted_x0) if mask is not None else predicted_x0

        if self.clip: # consider moving this 
            predicted_x0 = torch.clip(predicted_x0, min=-4, max=4)
        
        # compute guidance if necessary
        g = guidance_fn(predicted_x0) if guidance_fn is not None else 0   # 引导采样，引导力带来一个偏移值
        x0_guided = predicted_x0 + g
        
        xt_minus_one_direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_prediction
        extra_noise = eta * z  # 额外的随机噪声

        xt_minus_one = alpha_t_minus_one * (x0_guided) + xt_minus_one_direction + extra_noise
        # apply mask
        xt_minus_one = torch.where(mask, xt, xt_minus_one) if mask is not None else xt_minus_one

        return xt_minus_one, predicted_x0

    def step_ddpo(
            self, 
            eps_prediction, 
            t, 
            t_minus_one, 
            xt, 
            z,
            mask = None, # where mask is true, predicted x0 does not change from sample
            guidance_fn = None,
            return_mean_and_std = False, # 新增参数
            ):
        # eps_prediction: epsilon_hat (B, ...)
        # t: float
        # t_minus_one: float
        # xt: (B, ...)
        # guidance function: x_hat -> guidance
        # 确保 t 和 t_minus_one 是标量或与 batch 兼容的 tensor

        alpha_t_minus_one = self.alpha(t_minus_one).to(xt.device)
        alpha_t = self.alpha(t).to(xt.device)
        sigma_t_minus_one = self.sigma(t_minus_one).to(xt.device)
        sigma_t = self.sigma(t).to(xt.device)
        eta = self.eta(t, t_minus_one).to(xt.device) # 确保 eta 也在设备上

        # (xt - sigma_t * eps_prediction) / alpha_t 是模型对 x0 的预测
        predicted_x0 = (xt - sigma_t * eps_prediction)/alpha_t

        # apply mask
        if mask is not None:
            # 确保 mask 的维度与 predicted_x0 匹配，并将其移动到正确的设备
            predicted_x0 = torch.where(mask.to(xt.device), xt, predicted_x0)
        
        if self.clip:
            predicted_x0 = torch.clip(predicted_x0, min=-4, max=4)
        
        # compute guidance if necessary
        # 关键修正：确保x0_guided保持梯度
        if guidance_fn is not None:
            # ❌ 错误：guidance_fn可能断开梯度
            # g = guidance_fn(predicted_x0)  
            
            # ✅ 正确：使用detach的输入，但保持predicted_x0的梯度
            with torch.no_grad():
                g = guidance_fn(predicted_x0.detach())  # guidance不参与梯度计算
            x0_guided = predicted_x0 + g
        else:
            x0_guided = predicted_x0
        # x0_guided = predicted_x0

        # 计算策略均值 mu_theta
        # mu_theta = alpha_t_minus_one * x0_guided + sqrt(sigma_t_minus_one^2 - eta^2) * eps_prediction
        # 这里 sqrt(clip(...)) 确保了平方根内部非负
        sqrt_term = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0.)) # 使用 min=0. 来处理浮点精度问题
        mu_theta = alpha_t_minus_one * x0_guided + sqrt_term * eps_prediction
        # mu_theta = xt_minus_one
        # 策略标准差 sigma_s
        sigma_s = eta # 采样噪声的标准差就是 eta

        # 实际的 xt_minus_one 计算
        xt_minus_one = mu_theta + sigma_s * z

         # apply mask for xt_minus_one
        if mask is not None:
            xt_minus_one = torch.where(mask.to(xt.device), xt, xt_minus_one)

        if return_mean_and_std:
            # 返回实际生成的 xt_minus_one, 以及用于 log_prob 计算的 mu_theta 和 sigma_s
            return xt_minus_one, mu_theta, sigma_s
        else:
            return xt_minus_one, predicted_x0 # 保持原始返回 (xt_minus_one, predicted_x0) 兼容性
        
    # 反向扩散时下一步应该加多少噪声
    def eta(self, t, t_minus_one):
        # DDIM
        # return torch.tensor(0)
        a = self.sigma(t_minus_one) / self.sigma(t)
        b = torch.sqrt(1-torch.square(self.alpha(t)/self.alpha(t_minus_one)))
        return a * b  * self._eta

    # alpha 表示保留多少模型
    def alpha(self, t):
        return torch.cos((torch.pi/2)*t)
    
    # sigma 表示保留多少噪声
    def sigma(self, t):
        return torch.sin((torch.pi/2)*t)
    
    # sigma 表示保留多少噪声
    def beta(self, t):
        return torch.sin((torch.pi/2)*t)
    
    # 新增：获取alpha_t值
    def get_alpha_t(self, t):
        """
        获取时间步t的alpha值
        t: 可以是标量或tensor
        """
        return self.alpha(t)
     # 新增：获取sigma_t值  
    def get_sigma_t(self, t):
        """
        获取时间步t的sigma值
        t: 可以是标量或tensor
        """
        return self.sigma(t)
     # 新增：批量获取alpha和sigma
    def get_alpha_sigma_batch(self, timesteps):
        """
        批量获取多个时间步的alpha和sigma值
        timesteps: tensor of shape (B,)
        """
        alpha_values = self.alpha(timesteps)
        sigma_values = self.sigma(timesteps)
        return alpha_values, sigma_values
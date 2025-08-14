import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm
from main.arquitetura.stable_vc.modelo.unet.models.ema import EMA
from main.arquitetura.stable_vc.modelo.unet.training.grad_scaler import NativeScalerWithGradNormCount
from torch.nn.parallel import DistributedDataParallel
import gc
from torchmetrics import MeanMetric
from flow_matching.path import CondOTProbPath
from torch import nn

from main.arquitetura.stable_vc.modelo.unet.models.unet import UNetModel

class FlowMatchingWithLinear(nn.Module):
    def __init__(self, mel_dim, d_model, d_style, hidden_dim=64):
        """
        mel_dim: dimensão do mel espectrograma achatado (ex.: shape_mel * shape_mel)
        d_model: dimensão da saída do DiTBlock (conteúdo)
        d_style: dimensão do embedding de estilo (prosody_code)
        hidden_dim: dimensão oculta da rede interna
        """
        super(FlowMatchingWithLinear, self).__init__()
        input_dim = mel_dim + 1 + d_model + mel_dim + d_style
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mel_dim)  # Prediz um vetor que terá a mesma dimensão de X_t (velocidade)
        )


    def forward(self, x_t, t, content, mel_ref, style):
        """
        x_t: Tensor [batch, mel_dim] – o estado atual (ruidoso)
        t: Tensor [batch, 1] – tempo atual (valores entre 0 e 1)
        content: Tensor [batch, d_model] – saída do DiTBlock
        mel_ref: Tensor [batch, mel_dim] – mel espectrograma limpo (referência)
        style: Tensor [batch, d_style] – embedding de estilo (prosody_code)
        """
        # if t.dim() == 1:
        #     t = t.unsqueeze(1)
        inp = torch.cat([x_t, t, content, mel_ref, style], dim=-1)

        velocity = self.net(inp)

        return velocity

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, content, mel_ref, style ) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start,content, mel_ref, style) * (t_end - t_start) / 2, t_start + (t_end - t_start) / 2,content, mel_ref, style )


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def forward_padded(
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        extra: dict,
        mult: int = 16,
        orig_shape: tuple[int, int] = (80, 259),
):
    """
    Forward wrapper that accepts either:
      - x: [B, L] flattened vector where L = H*W
      - x: [B, C, H, W] image tensor
    t: [B]            timesteps tensor
    extra: dict       conditioning dict
    mult: int         pad length to multiple of this value
    orig_shape: tuple original image (H, W)
    """
    H, W = orig_shape
    # Detect input format
    if x.dim() == 2:
        # flattened vector
        B, L = x.shape
        expected = H * W
        if L != expected:
            raise ValueError(f"Esperado vetor achatado de tamanho {expected}, mas recebeu {L}")
        x_flat = x
        # Padflat to multiple of 'mult'
        length = x_flat.shape[1]
        pad_len = (mult - (length % mult)) % mult
        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, pad_len))

        # Reshape for 1D UNet: [B, 1, L_pad]
        x_in = x_flat.unsqueeze(1)
    elif x.dim() == 4:
        B, C, H, W = x.shape
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult

        # pad = (left, right, top, bottom)
        x_in = F.pad(x, (0, pad_w, 0, pad_h))

    else:
        raise ValueError(f"Input x deve ser 2-D ou 4-D, recebeu {x.dim()}-D")

    # Forward through model (must be UNetModel with dims=1)
    out_pred = model(x_in, t, extra=extra)
    # out_padded: [B, C_out, L_pad]

    if x.dim() == 4:
        return out_pred[:, :, :H, :W]
    else:
        return out_pred


def redimensionalizacao(samples, dims,device="cuda"):
    if dims == 1:
        samples = samples.reshape(1, -1).to(device, non_blocking=True)
    elif dims == 2:
        # canal
        samples = samples.unsqueeze(0).to(device, non_blocking=True)

    return samples


def train_flow_matching(
        model: UNetModel,
        data_loader,
        optimizer,
        num_epochs: int = 20,
        dims: int = 1,
        class_drop_prob: float = 0.1,
        labels=1,
        accum_iter: int = 1,
        skewed_timesteps=False,
        view_plot: bool = False,
        test_run: bool = True,
        learning_rate=1e-2,
        loss_scaler=NativeScalerWithGradNormCount()
):
    loss_list = []
    torch.optim.Adam(model.parameters(), learning_rate)
    lr_schedule: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # monitora a métrica para redução
        factor=0.1,  # redução por multiplicação
        patience=1,  # quantas épocas sem melhoria antes de reduzir
        verbose=True  # se quiser printar quando reduzir
    )

    for epoch in range(num_epochs):
        # samples muda de imagem para imagem como usando apenas uma imagem precisa mantar a original
        # samples = samples_original

        gc.collect()
        model.train(True)
        optimizer.zero_grad()

        # Inicializa as métricas de loss
        batch_loss = MeanMetric().to(device, non_blocking=True)
        epoch_loss = MeanMetric().to(device, non_blocking=True)

        path = CondOTProbPath()

        for data_iter_step, samples in tqdm(enumerate(data_loader)):

            if data_iter_step % accum_iter == 0:
                optimizer.zero_grad()
                batch_loss.reset()
                # Testa apenas uma imagem por epoca
                if data_iter_step > 0 and test_run:
                    break

            samples = redimensionalizacao(samples, dims)

            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            if skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.torch.rand(samples.shape[0]).to(device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            # Define o condicionamento com base na probabilidade
            if False:  # torch.rand(1) < class_drop_prob:
                conditioning = {}
            else:
                conditioning = {"label": labels}

            # with torch.cuda.amp.autocast():
            pred = forward_padded(model, x_t, t, conditioning, mult=16)

            if view_plot:
                plotar_flow_recontrucao(samples[0][0].detach().cpu().numpy(), pred[0][0].detach().cpu().numpy())

            loss = torch.pow(pred - u_t, 2).mean().to(device)

            loss_value = loss.item()
            batch_loss.update(loss)
            epoch_loss.update(loss)

            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")

            # Ajusta o loss para a acumulação
            loss /= accum_iter
            apply_update = True

            # Atualiza os gradientes utilizando o loss scaler
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=apply_update,
                clip_grad=1.0,
            )
            # Atualiza a média exponencial do modelo se aplicável
            if apply_update:
                if isinstance(model, EMA):
                    model.update_ema()
                elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
                    model.module.update_ema()

            # Registra logs a cada PRINT_FREQUENCY iterações
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: loss = {batch_loss.compute()}, lr = {lr}")

        val_loss = float(epoch_loss.compute().detach().cpu())
        lr_schedule.step(val_loss)
        print(f"loss: {val_loss:.5f}")
        loss_list.append(val_loss)

    return model, loss_list


def plotar_flow_recontrucao(img_1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_1 - img_1.min())  # normaliza [0,1]
    axes[0].axis("off")
    axes[0].set_title("Tensor 1")

    axes[1].imshow(img2 - img2.min())
    axes[1].axis("off")
    axes[1].set_title("Tensor 2")

    plt.tight_layout()
    plt.show()
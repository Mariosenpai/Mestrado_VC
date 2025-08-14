from safetensors import torch
from tqdm import tqdm

from main.arquitetura.auto_encoder.fit.treinamento.CVMPT import pre_processa_batch
from main.metricas.similaridade import calculate_similarity, distancia_euclidiana, media_desvio_padrao


def validacao(val_loader, generator, loss, num_epoca, device='cpu'):
    total_ssim = []
    total_psnr = []
    total_distancia_euclidiana = []
    generator.eval()
    with torch.no_grad():
        # print(f'Epocas : {1} / {num_epoca} ')
        for i, data_batch in enumerate(tqdm(val_loader)):
            batch_sem_ruido, batch_com_ruido = pre_processa_batch(data_batch, device)

            gen_natural = generator(batch_com_ruido).to(device)

            ssim_score, psnr_value = calculate_similarity(batch_sem_ruido, gen_natural)

            # Transformar imagem em auido
            # processado = transformar_imagem_em_audio(gen_natural[0][0].detach().cpu().numpy(), data_batch[0][0])
            # original = transformar_imagem_em_audio(batch_sem_ruido[0][0].detach().cpu().numpy(), data_batch[0][0])

            distancia = distancia_euclidiana(batch_sem_ruido, gen_natural)

            total_distancia_euclidiana.append(distancia)
            total_ssim.append(ssim_score)
            total_psnr.append(psnr_value)

        ssim_media, ssim_std = media_desvio_padrao(total_ssim)
        psnr_media, psnr_std = media_desvio_padrao(total_psnr)
        euclidiana_media, euclidiana_std = media_desvio_padrao(total_distancia_euclidiana)

        informacoes = f"----EPOCA {num_epoca}----\nLoss:{loss}\nSSIM médio (validação): {ssim_media:.2f} dB, devio padrao:{ssim_std}\nPSNR médio (validação): {psnr_media:.4f}, devio padrão: {psnr_std}\nDistancia Euclidiana Media: {euclidiana_media}, devio padrao: {euclidiana_std}"

        with open("auto_encoder/pesos/info_treinamento.txt", "a") as arquivo:
            arquivo.write(informacoes)

        print(f"SSIM medio (validacao): {ssim_media:.2f}, devio padrao: {ssim_std}")
        print(f"PSNR medio (validacao): {psnr_media:.4f}, devio padrão: {psnr_std:}")
        print(f"Euclidiana media (validacao): {euclidiana_media}, devio padrao: {euclidiana_std}")
import math

import librosa
import numpy as np

from src.common.pre_processamento.noise import f0, f0_log


def __f0_rmse_calculater_deprecated(wavs_natural, f0s_org,f0s_synth):
    min_cost_tot = []

    for i in range(len(wavs_natural)):
        frame_len = 0

        def logf0_rmse(x, y):
            log_spec_db_const = 1 / len(frame_len)
            diff = x - y
            return log_spec_db_const * math.sqrt(np.inner(diff, diff))

        if len(f0s_org[i]) < len(f0s_synth[i]):
            frame_len = f0s_org[i]
        else:
            frame_len = f0s_synth[i]

        cost_function = logf0_rmse
        min_cost, _ = librosa.sequence.dtw(f0s_org[i][:].T, f0s_synth[i][:].T, metric=cost_function)

        min_cost_tot.append(np.mean(min_cost))
        f0_rmse_value = sum(min_cost_tot) / len(min_cost_tot)

        return f0_rmse_value

def __f0_rmse_calculater(wavs_natural, f0s_org, f0s_synth):

    valores = []

    for i in range(len(wavs_natural)):

        org = np.asarray(f0s_org[i]).squeeze()
        synth = np.asarray(f0s_synth[i]).squeeze()

        # DTW
        _, wp = librosa.sequence.dtw(
            X=org.reshape(1, -1),
            Y=synth.reshape(1, -1),
            metric="euclidean"
        )

        # wp possui os índices alinhados pelo DTW
        idx_org = wp[:, 0]
        idx_synth = wp[:, 1]

        # F0 alinhados
        org_aligned = org[idx_org]
        synth_aligned = synth[idx_synth]

        # RMSE
        rmse = np.sqrt(
            np.mean((org_aligned - synth_aligned) ** 2)
        )

        valores.append(rmse)

    return np.mean(valores)
def _f0_rmse(wavs_noise: list[np.array],wavs_natural: list[np.array], sample_rate:int) -> float:
    f0s_org = f0(wavs_natural, sample_rate)
    f0s_synth = f0(wavs_noise, sample_rate)

    return __f0_rmse_calculater(wavs_natural, f0s_org, f0s_synth)

def _f0_rmse_log(wavs_noise: list[np.array],wavs_natural: list[np.array], sample_rate:int) ->float:
    f0s_org = f0_log(wavs_natural, sample_rate)
    f0s_synth = f0_log(wavs_noise, sample_rate)

    return __f0_rmse_calculater(wavs_natural, f0s_org, f0s_synth)
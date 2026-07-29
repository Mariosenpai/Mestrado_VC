import math

import numpy as np
import pysptk
import pyworld


def _compute_static_features(x:np.array, sample_rate:float):
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x,sample_rate,frame_period=5.0)
    f0 = pyworld.stonemask(x,f0, timeaxis, sample_rate)
    spectrogram = pyworld.cheaptrick(x,f0, timeaxis, sample_rate)
    aperiodicity = pyworld.d4c(x,f0, timeaxis, sample_rate)
    alpha = pysptk.util.mcepalpha(sample_rate)
    mc = pysptk.sp2mc(spectrogram, order=24,alpha=alpha)
    c0, mc = mc[:,0], mc[:,1]
    return mc


def modspec(x,n=4096, norm=None,return_phase=False):

    s_complex = np.fft.rfft(x, n=n,axis=0, norm=norm)
    assert s_complex.shape[0] == n//2+1
    R, im = s_complex.real, s_complex.imag
    ms = R * R + im * im

    if return_phase:
        return ms,np.exp(1.0j*np.angle(s_complex))
    else:
        return ms

def mean_modspec(wavs:list[np.ndarray], sample_rate:float):
    mss =[]

    for wav in wavs:
        mgc = _compute_static_features(wav, sample_rate)

        ms = np.log(modspec(mgc))
        mss.append(ms)

    return np.mean(np.array(mss), axis=(0,))


def _msd(wavs_sythn:list[np.ndarray], wavs_org:list[np.array],sample_rate:float) -> float:

    ms_into2out_orig = mean_modspec(wavs_org,sample_rate)
    ms_into2out_synth = mean_modspec(wavs_sythn,sample_rate)

    new =0
    for i in range(24):
        a = ms_into2out_orig[i]
        b = ms_into2out_synth[i]
        diff = np.mean(np.absolute(a-b))
        diff=(np.inner(diff,diff))
        new+=diff
    MSD = math.sqrt(1/len(mean_modspec(wavs_org,sample_rate)))*math.sqrt(new)
    return MSD
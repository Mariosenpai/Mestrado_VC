import numpy as np
import pyworld
import pyworld as pw
import librosa


def f0_constante(x, sr, f0_floor=25.0, f0_ceil=90.0, frame_period=1.0, speed=1, f0_contante=60.0):
    # 1. A convient way
    f0, sp, ap = pw.wav2world(x, sr)  # use default options
    y = pw.synthesize(f0, sp, ap, sr, pw.default_frame_period)
    # 2. Step by step
    # 2-1 Without F0 refinement
    _f0, t = pw.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
                    channels_in_octave=2,
                    frame_period=frame_period,
                    speed=speed)
    _f0[_f0 == 0] = f0_contante
    _sp = pw.cheaptrick(x, _f0, t, sr)
    _ap = pw.d4c(x, _f0, t, sr)
    _y = pw.synthesize(_f0, _sp, _ap, sr, frame_period)
    # librosa.output.write_wav('test/y_without_f0_refinement.wav', _y, fs)
    return _y



def f0(wavs:list[np.array], sample_rate:int, frame_period=5.0) -> list[np.array]:

    f0s = []
    for i in range(len(wavs)):
        wav = wavs[i]
        wav = wav.astype(np.float64)
        f0, _ = pyworld.harvest(wav,sample_rate, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
        f0s.append(f0)

    return f0s


def f0_log(wavs:list[np.array], sample_rate:int, frame_period=5.0) -> list[np.array]:

    f0s = f0(wavs, sample_rate, frame_period)
    log_f0s_concatenate = []
    for f in f0s:
        log_f0s_concatenate.append(np.ma.log(f))

    return log_f0s_concatenate


if __name__ == '__main__':
    x, fs = librosa.load(
        r'/dataset/2024_AUDIOS_PROJETO_LARINGE/SEM_TRAQUEOSTOMIA/DALLETE_FONO/NATURAL_MP3/2_n.mp3',
        dtype=np.float64)
    y = f0_constante(x, fs)
    print(y)

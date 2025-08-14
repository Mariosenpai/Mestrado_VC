import numpy as np
import pyworld as pw
import librosa


def f0_constante(x, sr, f0_floor=25.0, f0_ceil=500.0, frame_period=5.0, speed=1, f0_contante=80.0):
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


if __name__ == '__main__':
    x, fs = librosa.load(
        r'/dataset/2024_AUDIOS_PROJETO_LARINGE/SEM_TRAQUEOSTOMIA/DALLETE_FONO/NATURAL_MP3/2_n.mp3',
        dtype=np.float64)
    y = f0_constante(x, fs)
    print(y)

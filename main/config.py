path_save_auto = "arquitetura/auto_encoder/pesos"
path_save_GAN = "arquitetura/GAN/pesos"

image_size = 1024
tamanho_tensor = 1318970 #1420000
batch_size = 1
epocas = 1
porcentagem_teste = 0.1

arquivos_ignorados= [r"dataset/fma_small\108\108925.mp3",
                     r"dataset/fma_small\133\133297.mp3",
                     r"dataset/fma_small\099\099134.mp3",
                     r"dataset/fma_small\107\107535.mp3"]

arquivos_ignorados_CVMPT = [r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989458.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989415.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39988638.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989473.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39988641.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39983836.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989427.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989478.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_33781142.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989431.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989413.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_36111494.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39988639.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39983834.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989451.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39988636.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989476.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39983824.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989474.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989430.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989460.mp3",
 r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_27026967.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_33781143.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989449.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989412.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_30470112.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39983826.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989448.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_33781144.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_39989429.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_36461709.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_27677507.mp3",
r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_36461707.mp3"
 ]
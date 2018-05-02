from ASAMs.GAUSSIAN.GaussianSAM_ps import GaussianSAM as gsam_ps
from ASAMs.GAUSSIAN.GaussianSAM_pnr import GaussianSAM as gsam_pnr
from ASAMs.GAUSSIAN.GaussianSAM_snn import GaussianSAM as gsam_snn

gsam_pitch_sel = gsam_ps(10,10,3)
gsam_pnote_rest = gsam_pnr(10,10,3)
gsam_sus_note = gsam_snn(10,10,3)

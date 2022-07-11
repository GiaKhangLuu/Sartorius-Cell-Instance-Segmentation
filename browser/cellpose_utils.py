import sys
sys.path.append('../models/cellpose')

from cellpose import models

channels = [0, 0]
CP_MODEL = '../models/weight/cellpose_250622_concat_epoch_500'
SZ_MODEL = '../models/weight/cellpose_250622_concat_epoch_500_size.npy'

def init_models():
    cp_model = models.CellposeModel(gpu=False, pretrained_model=CP_MODEL, 
                                    concatenation=True)
    sz_model = models.SizeModel(cp_model=cp_model, pretrained_size=SZ_MODEL)

    return cp_model, sz_model

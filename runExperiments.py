from experiments import *
from dataloader import *

# this file runs experiments to obtain model and ensemble metrics (and plots)
baseline_models = [r"model%i_final" % i for i in range(69,79)]
experimentBaselines(baseline_models)

SWA_models = [r"model_%i_SWA_final" % i for i in range(69,79)]
experimentSWA(SWA_models)

swagDiag()
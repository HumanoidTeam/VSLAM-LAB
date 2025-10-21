import os.path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class KIMERA_baseline_dev(BaselineVSLAMLab):
    def __init__(self):
        default_parameters = {'verbose': 1, 'mode': 'mono-vi'}
        
        # Initialize the baseline
        super().__init__(baseline_name='kimera-dev', baseline_folder='Kimera-VIO-DEV', 
                        default_parameters=default_parameters)
        self.color = 'purple'
        self.modes = ['mono-vi', 'stereo-vi']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_kimera_mono_vi'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')



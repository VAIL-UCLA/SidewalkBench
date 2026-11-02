from sidewalkbench.models.base_model import ControllerCfg
model_types = ['CityWalker', 'FlowPilot', 'Cosmos3', 'InternVLA_N1', 'MBRA', 'MIMIC', 'NoMaD', 'S2E', 'ViNT', 'Keyboard']

def get_model(model_type: str, controller: ControllerCfg | None=None):
    if model_type not in model_types:
        raise ValueError(f'Unsupported model type: {model_type}')
    if model_type == 'CityWalker':
        from sidewalkbench.models.CityWalker.inference import CityWalkerNavigator
        return CityWalkerNavigator(controller=controller)
    elif model_type == 'FlowPilot':
        from sidewalkbench.models.FlowPilot.inference import FlowPilotNavigator
        return FlowPilotNavigator(controller=controller)
    elif model_type == 'Cosmos3':
        from sidewalkbench.models.Cosmos3.inference import Cosmos3Navigator
        return Cosmos3Navigator(controller=controller)
    elif model_type == 'InternVLA_N1':
        from sidewalkbench.models.InternVLA_N1.inference import DualVLNNavigator
        return DualVLNNavigator(controller=controller)
    elif model_type == 'MBRA':
        from sidewalkbench.models.MBRA.inference import MBRAPGNavigator
        return MBRAPGNavigator(controller=controller)
    elif model_type == 'MIMIC':
        from sidewalkbench.models.MIMIC.inference import MIMICNavigator
        return MIMICNavigator(controller=controller)
    elif model_type == 'NoMaD':
        from sidewalkbench.models.NoMaD.inference import NoMaDGLNavigator
        return NoMaDGLNavigator(controller=controller)
    elif model_type == 'S2E':
        from sidewalkbench.models.S2E.inference import S2ENavigator
        return S2ENavigator(controller=controller)
    elif model_type == 'ViNT':
        from sidewalkbench.models.ViNT.inference import VintGLOfficialNavigator
        return VintGLOfficialNavigator(controller=controller)
    elif model_type == 'Keyboard':
        from sidewalkbench.pg.inference_scripts.keyboard import KeyboardNavigator
        return KeyboardNavigator()
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

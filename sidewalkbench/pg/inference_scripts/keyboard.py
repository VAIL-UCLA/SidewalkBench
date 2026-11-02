from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

config = Se2KeyboardCfg(v_x_sensitivity=1.5, v_y_sensitivity=0.65)
model = Se2Keyboard(config)


class KeyboardNavigator:
    def __init__(self):
        self.model = Se2Keyboard(config)
        self._info = {"image_resolution": [288, 512], "context_size": 21}

    def reset(self, env_idx=None):
        pass

    def inference_vw(self, obs, robot=None):
        command = self.model.advance()
        action = command[:2].unsqueeze(0).repeat(obs.shape[0], 1).cuda()
        traj = action.unsqueeze(1).cpu().numpy()
        print(action)
        return action, traj

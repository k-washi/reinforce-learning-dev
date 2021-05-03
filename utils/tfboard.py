from tensorboardX import SummaryWriter
import os
from datetime import datetime, timedelta, timezone

# タイムゾーンの生成
JST = timezone(timedelta(hours=+9), 'JST')
TIME_FORMAT = "%y-%m-%d-%H-%M-%S"

class RLSummaryWriter():
    def __init__(self) -> None:
        os.makedirs("./logs", exist_ok=True)
        self._writer = SummaryWriter(log_dir=f"./logs/{datetime.now(JST).strftime(TIME_FORMAT)}")
        self._frame_id = 0
    def __delattr__(self, name: str) -> None:
        self._writer.close()
    
    def set_frame_id(self, x):
        self._frame_id = x

    def add_return(self, x):
        self._writer.add_scalar("Return", x, self._frame_id)

    def add_loss(self, x):
        self._writer.add_scalar("loss", x, self._frame_id)

class RLModelSaver():
    def __init__(self, model_dir) -> None:
        self._max_reward = -1
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir = f"{model_dir}/{datetime.now(JST).strftime(TIME_FORMAT)}"
        os.makedirs(self._model_dir, exist_ok=True)

    def reward_save(self, model, epi, reward):
        if reward > self._max_reward:
            self._max_reward = reward
            model_path = os.path.join(self._model_dir, f"reward_{str(epi).zfill(5)}_{str(reward).zfill(6)}")
            model.save_weights(model_path)
            return model_path

        return None
    
    def save(self, model, epi):
        model_path = os.path.join(self._model_dir, f"checkpoints_{str(epi).zfill(5)}")
        model.save_weights(model_path)
        return model_path

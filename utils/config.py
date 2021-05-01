import sys
import os
from hydra.experimental import compose, initialize_config_dir


class Config():
    """
    hydraによる設定値の取得 (conf)
    """
    @staticmethod
    def get_cnf(conf_path: str):
        """
        設定値の辞書を取得
        @param
            conf_path: str
        @return
            cnf: OmegaDict
        """
        conf_dir = os.path.join(os.getcwd(), "conf")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir):
            cnf = compose(config_name="default.yaml")
            return cnf
import yaml
import os
import sys

# 현재 스크립트 파일의 경로를 얻습니다.
script_directory = os.path.dirname(os.path.abspath(__file__))

# config.json 파일의 상대 경로를 구성합니다.
config_file_path = os.path.join(script_directory, 'config.yaml')

# config.yaml 파일을 읽습니다.
with open(config_file_path) as config_file:
    config = yaml.safe_load(config_file)
    
# test
if __name__ == '__main__':
    actor_lr = config["agent"]["actor_lr"]
    critic_lr = config["agent"]["critic_lr"]

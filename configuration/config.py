import json
import os

# 현재 스크립트 파일의 경로를 얻습니다.
script_directory = os.path.dirname(os.path.abspath(__file__))

# config.json 파일의 상대 경로를 구성합니다.
config_file_path = os.path.join(script_directory, 'config.json')

# config.json 파일을 읽습니다.
with open(config_file_path) as config_file:
    config = json.load(config_file)
    
# test
if __name__ == '__main__':
    learning_rate = float(config['learning_rate'])
    batch_size = int(config['batch_size'])
    epochs = int(config['epochs'])
    input_size = int(config['input_size'])
    hidden_size = int(config['hidden_size'])
    output_size = int(config['output_size'])
    
    
    max_processing_time = int(config['max_processing_time'])

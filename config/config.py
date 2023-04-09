import json

with open('config.json') as config_file:
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
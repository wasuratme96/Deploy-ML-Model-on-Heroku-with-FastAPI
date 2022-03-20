import yaml

config = yaml.safe_load(open("./params.yml"))
raw_data_path = config['data']['raw_data']
clean_data_path = config['data']['clean_data']


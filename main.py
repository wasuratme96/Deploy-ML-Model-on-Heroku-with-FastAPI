import yaml
import argparse
import logging

from src.data_cleaning import execute_clean_data
from src.preprocess_data import execute_data_split, execute_process_data
from src.train_model import execute_modeling
from src.model_validation import execute_model_validation
#import src.visualization

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

config = yaml.safe_load(open("./params.yml"))

def execute_pipeline(args):
    """
    Execute each pipeline for machine learning model training.
    Available steps:
        - data_cleaning
        - feature_engineering
        - traing_test_model
        - 

    """
    if args.step == "all" or args.step == "data_cleaning":
        logger.info("Perform Data Cleaning")
        execute_clean_data(config['data'])

    if args.step == "all" or args.step == "data_split":
        logger.info("Perform Data Splitting")
        execute_data_split(config['data'], config['model'])

    if args.step == "all" or args.step == "process_data":
        logger.info("Perform Preprocessing Data")
        execute_process_data(config['data'],
                             config['process_data'],
                             config['process_data']['train_mode'],
                             config['model'])

    if args.step == "all" or args.step == "train_model":
        logger.info("Perform Model Development - Train")
        execute_modeling(config['data']['train'], config['model'])

    if args.step == 'all' or args.step == "model_validation":
        logger.info("Perform model validation")
        execute_model_validation(config['data']['test'], config['process_data'], config['model'])


if __name__ == "__main__":
    # Main entry point of pipeline
    parser = argparse.ArgumentParser(description= "CICD Model Training")

    parser.add_argument(
        "--step",
        type = str,
        choices = ["data_cleaning",
                   "data_split",
                   "process_data",
                   "traing_test_model",
                   "validation",
                   "all"], 
        default = "all",
        help = "ML Pipeline actions to execute"
    )
    args = parser.parse_args()

    execute_pipeline(args)

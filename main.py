import argparse
import os

from src import train, evaluate, config, utils

def main():
    parser = argparse.ArgumentParser(description="MultiZoo Animal Classification Project")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate"],
                        help="Operating mode: 'train' or 'evaluate'")
    parser.add_argument("--model_path", type=str, default=os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME),
                        help="Path to the model file for evaluation or for continuing training (if supported). ")
    parser.add_argument("--data_dir", type=str,
                        help="Directory of the data to use. For training, defaults to config.DATA_DIR. For evaluation, defaults to config.TEST_DATA_DIR.")
   
    args = parser.parse_args()

    

    utils.set_seed(config.RANDOM_SEED) 

    if args.mode == "train":
        print("Starting training mode...")
       
        config.DATA_DIR = args.data_dir if args.data_dir else config.DATA_DIR
        print(f"Training data will be loaded from: {config.DATA_DIR}")
        train.run_training()
        print("Training finished.")

    elif args.mode == "evaluate":
        print("Starting evaluation mode...")
        model_to_evaluate = args.model_path
        
        
        eval_data_dir = args.data_dir if args.data_dir else config.TEST_DATA_DIR
        print(f"Evaluation data will be loaded from: {eval_data_dir}")

        if not os.path.exists(model_to_evaluate):
            print(f"Error: Model file not found at {model_to_evaluate}. Cannot evaluate.")
            print("Please ensure you have a trained model or provide the correct path via --model_path.")
            return
        if not os.path.exists(eval_data_dir):
            print(f"Error: Evaluation data directory not found at {eval_data_dir}. Cannot evaluate.")
            return
            
        evaluate.run_evaluation(model_path=model_to_evaluate, data_dir=eval_data_dir, data_type="Test")
        print("Evaluation finished.")

if __name__ == "__main__":
    main() 
import yaml
import os
import subprocess
import copy
from run_ablations import update_recursive # On réutilise la fonction helper

experiments = {
    "baseline": {
        "updates": {}, 
        "output_dir": "outputs_imdb/baseline"
    },
    "exp_1_light": {
        "updates": {
            "model": {"hidden_dim": 64, "bidirectional": False}
        },
        "output_dir": "outputs_imdb/exp_1_light"
    },
    "exp_2_high_lr": {
        "updates": {
            "train": {"lr": 0.005}
        },
        "output_dir": "outputs_imdb/exp_2_high_lr"
    },
    # Ajout d'une expérience spécifique à l'Overfitting fréquent sur IMDb
    "exp_3_high_dropout": {
        "updates": {
            "model": {"dropout": 0.5},
            "train": {"weight_decay": 0.05}
        },
        "output_dir": "outputs_imdb/exp_3_high_dropout"
    }
}

base_config_path = "configs/nlp_imdb.yaml"

def run_experiment(name, exp_data):
    print(f"\n{'='*60}\n🚀 IMDB Experiment: {name}\n{'='*60}")
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    current_config = copy.deepcopy(base_config)
    update_recursive(current_config, exp_data["updates"])
    current_config["output_dir"] = exp_data["output_dir"]
    
    os.makedirs("configs/ablations_imdb", exist_ok=True)
    temp_config_path = f"configs/ablations_imdb/{name}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(current_config, f)
    
    cmd = f"python src/train.py --config {temp_config_path}"
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    for exp_name, exp_data in experiments.items():
        run_experiment(exp_name, exp_data)
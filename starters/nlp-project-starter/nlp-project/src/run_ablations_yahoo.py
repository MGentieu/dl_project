import yaml
import os
import subprocess
import copy
from run_ablations import update_recursive

# Expériences sur le dataset COMPLET
experiments = {
    "baseline": {
        "updates": {}, 
        "output_dir": "outputs_yahoo/baseline"
    },
    # Exp 1 : Embedding plus grand pour capturer plus de sémantique
    "exp_1_large_embed": {
        "updates": {
            "model": {"emb_dim": 256}
        },
        "output_dir": "outputs_yahoo/exp_1_large_embed"
    },
    # Exp 2 : Deep LSTM (2 couches) - utile pour 1.4M de données
    "exp_2_deep_lstm": {
        "updates": {
            "model": {"num_layers": 2, "dropout": 0.4}
        },
        "output_dir": "outputs_yahoo/exp_2_deep_lstm"
    }
}

base_config_path = "configs/nlp_yahoo.yaml"

def run_experiment(name, exp_data):
    print(f"\n{'='*60}\n🚀 Yahoo Experiment (Full Dataset): {name}\n{'='*60}")
    
    if not os.path.exists(base_config_path):
        print(f"❌ Erreur: {base_config_path} introuvable.")
        return

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    current_config = copy.deepcopy(base_config)
    update_recursive(current_config, exp_data["updates"])
    current_config["output_dir"] = exp_data["output_dir"]
    
    os.makedirs("configs/ablations_yahoo", exist_ok=True)
    temp_config_path = f"configs/ablations_yahoo/{name}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(current_config, f)
    
    # On lance l'entraînement
    cmd = f"python src/train.py --config {temp_config_path}"
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    for exp_name, exp_data in experiments.items():
        run_experiment(exp_name, exp_data)
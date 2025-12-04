import yaml
import os
import subprocess
import copy

# Fonction utilitaire pour mettre à jour des dictionnaires imbriqués (nested dicts)
def update_recursive(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict:
            update_recursive(base_dict[key], value)
        else:
            base_dict[key] = value

# --- DÉFINITION DES EXPÉRIENCES ---
# Structure : "Nom_Expérience": { "updates": { ...paramètres... }, "output_dir": "..." }
experiments = {
    # 0. Baseline (Configuration par défaut)
    "baseline": {
        "updates": {}, 
        "output_dir": "outputs/baseline"
    },

    # 1. Architecture Légère (Modèle simple)
    # On réduit la taille cachée et on passe en unidirectionnel
    "exp_1_light": {
        "updates": {
            "model": {"hidden_dim": 64, "bidirectional": False}
        },
        "output_dir": "outputs/exp_1_light"
    },

    # 2. Optimization - High Learning Rate
    # On garde AdamW mais on augmente le LR pour voir si ça converge plus vite ou diverge
    "exp_2_high_lr": {
        "updates": {
            "train": {"lr": 0.005}  # 5x le LR de base (1e-3)
        },
        "output_dir": "outputs/exp_2_high_lr"
    },

    # 3. Optimization - SGD + Momentum
    # On change l'optimiseur et on adapte le LR (SGD a souvent besoin d'un LR plus élevé)
    "exp_3_sgd": {
        "updates": {
            "train": {
                "optimizer": "sgd",
                "lr": 0.01,
                "momentum": 0.9
            }
        },
        "output_dir": "outputs/exp_3_sgd"
    },

    # 4. Heavy Model & Regularization (Modifications multiples)
    # Modèle plus gros (512) mais avec un fort Dropout (0.5) pour régulariser
    "exp_4_heavy_reg": {
        "updates": {
            "model": {"hidden_dim": 512, "dropout": 0.5}
        },
        "output_dir": "outputs/exp_4_heavy_reg"
    }
}

base_config_path = "configs/nlp_agnews.yaml"

def run_experiment(name, exp_data):
    print(f"\n{'='*60}")
    print(f"🚀 Lancement de l'expérience : {name}")
    print(f"{'='*60}")
    
    # 1. Charger la config de base
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 2. Préparer la nouvelle config
    # On fait une copie profonde pour ne pas modifier l'original pour les tours suivants
    current_config = copy.deepcopy(base_config)
    
    # Appliquer les mises à jour récursives
    update_recursive(current_config, exp_data["updates"])
    
    # Forcer le dossier de sortie
    current_config["output_dir"] = exp_data["output_dir"]
    
    # 3. Sauvegarder le fichier YAML temporaire
    # On crée un dossier configs/temp pour ne pas polluer
    os.makedirs("configs/ablations", exist_ok=True)
    temp_config_path = f"configs/ablations/{name}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(current_config, f)
    
    # 4. Lancer l'entraînement
    # On peut réduire epochs à 5 pour gagner du temps lors des tests, 
    # sinon on laisse la valeur par défaut du yaml (10)
    cmd = f"python src/train.py --config {temp_config_path}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Expérience {name} terminée avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur critique sur l'expérience {name}.")

if __name__ == "__main__":
    for exp_name, exp_data in experiments.items():
        run_experiment(exp_name, exp_data)
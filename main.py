import numpy as np

# --- Fonctions d'activation ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(v): return v * (1 - v)
def tanh(x): return np.tanh(x)
def tanh_deriv(v): return 1 - v**2

def formater_poids(W, m):
    """Affiche les poids avec la notation W(m, ij)"""
    print(f"\n--- Poids de la couche m={m} ---")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            print(f"W({m}, {i+1}{j+1}) = {W[i, j]:.6f}")

def saisir_poids_manuellement(n_lignes, n_colonnes, m):
    """Permet de saisir chaque poids W(m, ij) un par un"""
    W = np.zeros((n_lignes, n_colonnes))
    print(f"\n--- Saisie des poids pour la couche m={m} ({n_lignes}x{n_colonnes} liaisons) ---")
    for i in range(n_lignes):
        for j in range(n_colonnes):
            W[i, j] = float(input(f"  Entrez W({m}, {i+1}{j+1}) : "))
    return W

# --- Configuration ---
print("=== CONFIGURATION DU RÉSEAU RNA ===")
mode = input("Fonction d'activation (1: Sigmoide, 2: Tanh) : ")
f, f_deriv = (tanh, tanh_deriv) if mode == "2" else (sigmoid, sigmoid_deriv)

# Saisie des dimensions
n_entree = int(input("Nombre d'unités en couche 1 (entrée) : "))
n_cache = int(input("Nombre d'unités en couche 2 (cachée) : "))
n_sortie = int(input("Nombre d'unités en couche 3 (sortie) : "))

# --- INITIALISATION DYNAMIQUE DES POIDS ---
init_type = input("\nInitialisation (1: Saisie manuelle, 2: Aléatoire) : ")

if init_type == "1":
    # On saisit les poids pour chaque couche en fonction des dimensions
    W2 = saisir_poids_manuellement(n_cache, n_entree, 2)
    W3 = saisir_poids_manuellement(n_sortie, n_cache, 3)
else:
    # Initialisation aléatoire
    W2 = np.random.uniform(-0.5, 0.5, (n_cache, n_entree))
    W3 = np.random.uniform(-0.5, 0.5, (n_sortie, n_cache))

# --- Prototype et paramètres ---
print("\n--- Données du Prototype ---")
inp_str = input(f"Entrez les {n_entree} valeurs de V(1,i) (ex: 1,0,1) : ")
V1 = np.array([float(x) for x in inp_str.split(",")])
target_str = input(f"Entrez les {n_sortie} valeurs désirées d(u,i) (ex: 1) : ")
d = np.array([float(x) for x in target_str.split(",")])
eta = float(input("Pas d'apprentissage (eta) : "))

# --- TRAITEMENT (Étapes 3 à 6) ---

# 3. Propagation Avant
net2 = np.dot(W2, V1)
V2 = f(net2)
net3 = np.dot(W3, V2)
V3 = f(net3)

# 4. Erreur en sortie (m=3)
delta3 = f_deriv(V3) * (d - V3)

# 5. Rétropropagation (m=2)
delta2 = f_deriv(V2) * np.dot(W3.T, delta3)

# 6. Mise à jour des poids
W3 += eta * np.outer(delta3, V2)
W2 += eta * np.outer(delta2, V1)

# --- SORTIE AMÉLIORÉE ---
print("\n" + "="*50)
print("             RÉSULTATS APRÈS MISE À JOUR")
print("="*50)
print(f"Sortie finale V(3,1)  : {V3[0]:.6f}")
print(f"Erreur résiduelle     : {(d[0] - V3[0]):.6f}")

formater_poids(W2, 2)
formater_poids(W3, 3)
print("\n" + "="*50)
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
            # i+1 et j+1 pour coller à la notation humaine (commence à 1)
            print(f"W({m}, {i+1}{j+1}) = {W[i, j]:.6f}")

# --- Configuration ---
print("=== CONFIGURATION DU RÉSEAU RNA ===")
mode = input("Fonction d'activation (1: Sigmoide, 2: Tanh) : ")
f, f_deriv = (tanh, tanh_deriv) if mode == "2" else (sigmoid, sigmoid_deriv)

# Saisie dynamique des dimensions
n_entree = int(input("Nombre d'unités en couche 1 (entrée) : "))
n_cache = int(input("Nombre d'unités en couche 2 (cachée) : "))
n_sortie = int(input("Nombre d'unités en couche 3 (sortie) : "))

# Initialisation des poids
init_type = input("Initialisation (1: Ton exemple fixe, 2: Aléatoire) : ")
if init_type == "1" and n_entree==3 and n_cache==2 and n_sortie==1:
    W2 = np.array([[0.2, 0.1, 0.1], [0.3, 0.2, 0.3]])
    W3 = np.array([[0.2, 0.3]])
else:
    # Initialisation aléatoire entre -0.5 et 0.5
    W2 = np.random.uniform(-0.5, 0.5, (n_cache, n_entree))
    W3 = np.random.uniform(-0.5, 0.5, (n_sortie, n_cache))

# Prototype
inp_str = input(f"Entrez les {n_entree} valeurs de V(1,i) séparées par virgules : ")
V1 = np.array([float(x) for x in inp_str.split(",")])
target_str = input(f"Entrez les {n_sortie} valeurs désirées d(u,i) : ")
d = np.array([float(x) for x in target_str.split(",")])
eta = float(input("Pas d'apprentissage (eta) : "))

# --- TRAITEMENT ---

# 3. Propagation Avant
net2 = np.dot(W2, V1)
V2 = f(net2)
net3 = np.dot(W3, V2)
V3 = f(net3)

# 4. Erreur en sortie (m=3)
delta3 = f_deriv(V3) * (d - V3)

# 5. Rétropropagation (m=2)
delta2 = f_deriv(V2) * np.dot(W3.T, delta3)

# 6. Mise à jour (en une étape)
W3 += eta * np.outer(delta3, V2)
W2 += eta * np.outer(delta2, V1)

# --- SORTIE AMÉLIORÉE ---
print("\n" + "="*40)
print("             RÉSULTATS")
print("="*40)
print(f"Sortie obtenue V(3,1) : {V3[0]:.6f}")
print(f"Erreur (d - V)        : {(d[0] - V3[0]):.6f}")

print("\n--- NOUVELLES VALEURS DES POIDS ---")
formater_poids(W2, 2)
formater_poids(W3, 3)
print("\n" + "="*40)
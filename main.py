import numpy as np

# --- Définition des fonctions et de leurs dérivées ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(v):
    return v * (1 - v)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(v):
    return 1 - v**2

# --- 1. Configuration dynamique ---
print("--- Configuration du Réseau ---")
choice = input("Choisissez la fonction d'activation (1: Sigmoide, 2: Tanh) : ")

if choice == "2":
    f, f_deriv = tanh, tanh_deriv
    print("Mode : Tangente Hyperbolique activé.")
else:
    f, f_deriv = sigmoid, sigmoid_deriv
    print("Mode : Sigmoïde activé.")

# Saisie du prototype
print("\n--- Saisie du Prototype u=1 ---")
inp_str = input("Entrez les valeurs de la couche 1 (ex: 1,0,1) : ")
V1 = np.array([float(x) for x in inp_str.split(",")])

target = float(input("Entrez la sortie désirée (d) : "))
eta = float(input("Entrez le pas d'apprentissage (eta, ex: 0.1) : "))

# --- 2. Initialisation des poids (valeurs de ton exemple) ---
W2 = np.array([
    [0.2, 0.1, 0.1],
    [0.3, 0.2, 0.3]
])
W3 = np.array([
    [0.2, 0.3]
])

# --- 3. Propagation Avant ---
net2 = np.dot(W2, V1)
V2 = f(net2)

net3 = np.dot(W3, V2)
V3 = f(net3)

# --- 4 & 5. Calcul des Deltas ---
# Couche de sortie (M=3)
delta3 = f_deriv(V3) * (target - V3)

# Couche cachée (m=2)
delta2 = f_deriv(V2) * np.dot(W3.T, delta3)

# --- 6. Mise à jour ---
W3 += eta * np.outer(delta3, V2)
W2 += eta * np.outer(delta2, V1)

# --- 7. Affichage précis à 6 chiffres ---
print("\n--- RÉSULTATS ---")
print(f"Sortie V(3,1) : {V3[0]:.6f}")
print(f"Erreur brute  : {(target - V3[0]):.6f}")
print("\nNouveaux poids W(3,ij) :")
print(np.round(W3, 6))
print("\nNouveaux poids W(2,ij) :")
print(np.round(W2, 6))
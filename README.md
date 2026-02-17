
# Apprentissage par Descente de Gradient pour Réseau Multicouches (MLP)

Ce projet implémente un algorithme d'apprentissage pour un réseau de neurones artificiels (RNA) multicouches, codé "from scratch" en Python. Il a été conçu pour illustrer le fonctionnement mathématique de la rétropropagation du gradient (backpropagation).

## Fonctionnalités

- **Architecture Dynamique** : Configuration libre du nombre d'unités pour la couche d'entrée ($m=1$), la couche cachée ($m=2$) et la couche de sortie ($m=3$).
- **Double Activation** : Support des fonctions **Sigmoïde** et **Tangente Hyperbolique (Tanh)**.
- **Initialisation Flexible** : Saisie manuelle des poids (idéal pour vérifier des exercices théoriques) ou génération aléatoire.
- **Sortie Précise** : Affichage des résultats avec 6 chiffres après la virgule, utilisant la notation formelle $W(m, ij)$.

## Installation & Configuration (Ubuntu)

Cloner le dépôt :
    Bash
      git clone git@github.com:Tsitohainiavo/lgorithme-d-apprentissage-par-descente-de-gradient-pour-reseau-multicouches-.git
      cd lgorithme-d-apprentissage-par-descente-de-gradient-pour-reseau-multicouches-

    Créer l'environnement virtuel :
    Bash
      python3 -m venv algoRNA
      source algoRNA/bin/activate

Installer les dépendances :
   Bash
    pip install numpy

## Utilisation

Lancez le script principal et suivez les instructions interactives :
Bash

python3 main.py

Notation utilisée

    V(m,i) : sortie de la i-ème unité dans la m-ème couche.

    W(m,ij) : poids de connexion de V(m−1,j) à V(m,i).

    net(m,i) : somme pondérée reçue par l'unité i de la couche m.

## Algorithme (Les 7 Étapes)

L'implémentation suit rigoureusement ce cycle :

    Initialisation des poids W(m,ij).

    Présentation du prototype u sur la couche d'entrée (m=1).

    Propagation avant : Calcul de V(m,i)=f(net(m,i)) pour m=2 et m=3.

    Calcul du delta de sortie : δ(M,i)=f′(net(M,i))(d(u,i)−V(M,i)).

    Rétropropagation : δ(m,i)=f′(net(m,i))∑k​δ(m+1,k)W(m+1,ki) pour m=2.

    Mise à jour des poids : W(m,ij)=W(m,ij)+η⋅δ(m,i)⋅V(m−1,j).

    Itération : Retour à l'étape 2 pour les prototypes suivants.

**Auteur**

Tsitohainiavo
EOF

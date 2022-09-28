# Deep_learning_pytorch_intro

# Explication détaillée du code :

On commence par définir notre classe Net du module NeuralNet de Pytorch, qui représente le réseau de neuronnes.

Pour ceci, on a la fonction _init()_ qui nous permet d'initialiser les hyperparamètres de notre modèle, à savoir le nombre des entrées et sorties. Dans notre cas, on a deux couches cachées linéaires.

La fonction _forward()_ décrit comment l'information se propage dans le réseau de neuronnes. Après la première couche, on applique une fonction d'activation linéaire _tanh()_ ; c'est une fonction principalement utilisée pour la classification entre deux classes.

Les fonctions _train()_ et _test()_ sont deux fonctions utilisées réciproquement pour entrainer et évaluer le modèle:

    La fonction train() prend comme paramètres le nombre d'époques c'est à dire le nombre d'itération et le learning rate qu'on verra par le suite son impact sur les résultats et surtout sur le calcul de la perte (loss function).
    Pendant cette phase, on prend comme algorithme d'optimisation. En effet l'optimiseur est utilisé pour diminuer les taux d'erreur lors de la formation des réseaux neuronaux. L'optimiseur Adam est défini comme un processus utilisé comme optimiseur de remplacement pour la descente de gradient. Il est très efficace pour les problèmes de grande taille qui comprennent beaucoup de données.On utilise comme critère de perte l'entropie croisée qui calcule la perte d'entrepie entre la cible et la valeur de sortie (prédite)

    On convertit nos entrées et cibles en tenseur puis pour chaque itération on fait un shuffle de nos données; généralement on fait du shuffle pour s'assurer que notre modèle overfit moins .On calcule ensuite l'accuracy et la perte en faisant appel à la fonction _test()_ puis on fait de la backpropagation pour mettre à jour les poids.

La fonction _gendata()_ a comme objectif la génération des données; ici on choisit comme entrée f(x,y)=log(x+y) et comme cible y2.
x et y étant des valeurs aléatoires générées grace à la distribution uniforme.
et y2 une matrice de 1 et 0. (On suppose qu'on a une classification de deux classes).

Dans le main, on fait appel à _toytest()_ la fonction qui permet l'entrainement du modèle :

    - On construit un réseau avec la classe Net
    - On modifie le *learning_rate*
    - On finit par avoir un graphe qui montre les courbes de la perte et de l'accuracy

On remarque qu'en changeant le learning rate, la vitesse d'apprentissage changent.
En effet, plus lr est petit plus la courbe du loss prend du temps à décroitre.

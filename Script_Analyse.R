#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  PROJET : ANALYSE DES VENTES ET SEGMENTATION CLIENT AVEC R
#   Programmation Statistique avec R - Bachelor 2 Business Data Science
#
#
#  Contenu du script :
#    Étape 1 : Importation et préparation des données
#    Étape 2 : Analyse exploratoire (EDA)
#    Étape 3 : Segmentation client (RFM + k-means)
#    Étape 4 : Modélisation prédictive 
#
#    Auteur: MADJITESSEM Bénédicte
#           
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# 0. PACKAGES
# =============================================================================
# 

library(readxl)      
library(dplyr)       
library(ggplot2)     
library(lubridate)   
library(factoextra)  
library(pROC)        
library(caret)       


# =============================================================================
# ÉTAPE 1 : IMPORTATION ET PRÉPARATION DES DONNÉES
# =============================================================================

# ---- 1.1 Chargement des données ----
donnees_ventes <- read_excel("historique_ventes.xlsx")

# ---- 1.2 Premier aperçu ----
head(donnees_ventes, 10)    # 10 premières lignes
str(donnees_ventes)         # structure : type de chaque colonne
dim(donnees_ventes)         # nombre de lignes et de colonnes

# ---- 1.3 Valeurs manquantes ----
colSums(is.na(donnees_ventes))

# ---- 1.4 Renommage des colonnes ----
# Copie dans "Ventes" : on garde "donnees_ventes" intact comme sauvegarde

Ventes <- donnees_ventes

names(Ventes) <- c("id_client", "date_achat", "montant",
                   "categorie", "frequence", "region")
str(Ventes)

# ---- 1.5 Contrôle qualité ----

summary(Ventes)                   # Statistique descriptive
sum(duplicated(Ventes))           # lignes dupliquées
sum(Ventes$montant <= 0)          # montants nuls ou négatifs
sum(Ventes$frequence <= 0)        # fréquences impossibles
table(Ventes$categorie)           # modalités de la catégorie
table(Ventes$region)              # modalités de la région
range(Ventes$date_achat)          # plage de dates
length(unique(Ventes$id_client))  # clients uniques (200 = 1 ligne par client)

# ---- 1.6 Transformation des variables ----
# Toutes les lignes modifient le même tableau : Ventes
Ventes$categorie  <- as.factor(Ventes$categorie)                # texte -> facteur
Ventes$region     <- as.factor(Ventes$region)                   # texte -> facteur
Ventes$date_achat <- as.Date(Ventes$date_achat)                 # date-heure -> date

# Extraire l'année et le mois de la date
Ventes$annee      <- year(Ventes$date_achat)                    # année
Ventes$mois       <- month(Ventes$date_achat, label = TRUE)     # mois (nom abrégé)

str(Ventes)


# =============================================================================
# ÉTAPE 2 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =============================================================================

# ---- 2.1 Distribution des montants dépensés ----
# Histogramme 

ggplot(Ventes, aes(x = montant)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  labs(title = "Distribution des montants dépensés",
       x = "Montant dépensé (€)", y = "Nombre de clients") +
  theme_minimal()

# ---- 2.2 Fréquence d'achat par région ----
# Boxplot 
ggplot(Ventes, aes(x = region, y = frequence)) +
  geom_boxplot(fill = "steelblue", alpha = 0.6) +
  labs(title = "Fréquence d'achat par région",
       x = "Région", y = "Nombre d'achats") +
  theme_minimal()


# ---- 2.3 Tendance des ventes dans le temps ----
# floor_date() ramène chaque date au 1er jour de son mois (15/08/2024 -> 01/08/2024)
# On évite ainsi de mélanger les mêmes mois de deux années différentes

Ventes$mois_annee <- floor_date(Ventes$date_achat, unit = "month")

ventes_mois <- Ventes %>%
  group_by(mois_annee) %>%
  summarise(total = sum(montant, na.rm = TRUE))

ggplot(ventes_mois, aes(x = mois_annee, y = total)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  labs(title = "Évolution des ventes mensuelles",
       x = "Mois", y = "Montant total (€)") +
  theme_minimal()


# ---- 2.4 Statistiques descriptives ----

# Dépense moyenne par client 
depense_moyenne <- mean(Ventes$montant, na.rm = TRUE);depense_moyenne

# Répartition des ventes par catégorie (CA total et part en %)
repartition_cat <- Ventes %>%
  group_by(categorie) %>%
  summarise(ca_total = sum(montant)) %>%
  mutate(part = round(ca_total / sum(ca_total) * 100, 1)) %>%
  arrange(desc(part))

repartition_cat

# Graphique de la répartition du chiffre d'affaires par catégorie
ggplot(repartition_cat, aes(x = reorder(categorie, part), y = part)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Part du chiffre d'affaires par catégorie",
       x = "Catégorie", y = "Part du CA (%)") +
  theme_minimal()


# =============================================================================
# ÉTAPE 3 : SEGMENTATION CLIENT
# =============================================================================

# #############################################################################
# PARTIE A : Appliquer la méthode RFM (Récence, Fréquence, Montant)
#            pour créer des segments clients
# #############################################################################

# -----------------------------------------------------------------------------
# A1. Calculer les scores R, F et M pour chaque client
# -----------------------------------------------------------------------------
# R = Récence   : nombre de jours depuis le dernier achat
# F = Fréquence : nombre d'achats (colonne "frequence")
# M = Montant   : somme dépensée (colonne "montant")
# Date de référence = lendemain du dernier achat du jeu de données
date_ref <- as.Date("2024-12-11")

rfm <- Ventes %>%
  mutate(recence = as.numeric(date_ref - date_achat)) %>%
  select(id_client, recence, frequence, montant)

# Scores de 1 à 5 : ntile(x, 5) découpe les 200 clients en 5 groupes de 40
# Récence : petit = bon, donc on inverse avec 6 - ntile()
rfm <- rfm %>%
  mutate(
    R = 6 - ntile(recence, 5),
    F = ntile(frequence, 5),
    M = ntile(montant, 5)
  )

# -----------------------------------------------------------------------------
# A2. Catégoriser les clients en fonction des scores
#     
# -----------------------------------------------------------------------------
# case_when lit les règles de haut en bas : la première vraie l'emporte
rfm <- rfm %>%
  mutate(
    segment = case_when(
      R >= 4 & F >= 4 & M >= 4 ~ "Champions",           # récents, fréquents, gros montants
      F >= 4 & R >= 3          ~ "Clients fidèles",     # achètent souvent, assez récents
      R >= 4 & F <= 2          ~ "Nouveaux / récents",  # récents mais peu d'achats
      R <= 2 & F >= 4          ~ "À risque",            # fréquents mais plus actifs récemment
      R <= 2 & F <= 2          ~ "Dormants",            # anciens et peu d'achats
      TRUE                     ~ "Clients moyens"       # tous les autres
    )
  )


# #############################################################################
# PARTIE B : Implémenter un algorithme de clustering comme k-means
# #############################################################################

# -----------------------------------------------------------------------------
# B1. Normaliser les données avec scale()
# -----------------------------------------------------------------------------
# Moyenne 0 et écart-type 1 : sans cela, le montant (jusqu'à 1000) écraserait
# la fréquence (1 à 20) dans le calcul des distances
rfm_scaled <- scale(rfm[, c("recence", "frequence", "montant")])

# -----------------------------------------------------------------------------
# B2. Utiliser factoextra pour choisir le nombre optimal de clusters
# -----------------------------------------------------------------------------
# Méthode du coude : chercher le k où la courbe cesse de baisser fortement
fviz_nbclust(rfm_scaled, kmeans, method = "wss", nstart = 25) +
  labs(title = "Méthode du coude")

# Indice de silhouette : le meilleur k est celui où la valeur est la plus haute
fviz_nbclust(rfm_scaled, kmeans, method = "silhouette", nstart = 25) +
  labs(title = "Indice de silhouette")

# Choix de k : à ajuster après lecture des deux graphiques ci-dessus
k <- 4

# Lancement de l'algorithme (nstart = 25 : 25 départs aléatoires, R garde le meilleur)
km <- kmeans(rfm_scaled, centers = k, nstart = 25)

# -----------------------------------------------------------------------------
# B3. Visualiser les clusters obtenus
# -----------------------------------------------------------------------------
fviz_cluster(km, data = rfm_scaled, geom = "point",
             ellipse.type = "convex", ggtheme = theme_minimal()) +
  labs(title = paste("Clusters k-means (k =", k, ")"))

# =============================================================================
# ÉTAPE 4 : MODÉLISATION PRÉDICTIVE (optionnelle)
# =============================================================================

# -----------------------------------------------------------------------------
# Préparation : construire la variable à prédire
# -----------------------------------------------------------------------------

modele_df <- Ventes[, c("frequence", "montant", "region", "categorie")]
modele_df$achat_3m <- factor(as.numeric(Ventes$date_achat >= as.Date("2024-09-11")))


# -----------------------------------------------------------------------------
# 1. Diviser les données en ensemble d'entraînement et de test
# -----------------------------------------------------------------------------

set.seed(123)

idx <- sample(1:nrow(modele_df), 0.7 * nrow(modele_df))

train <- modele_df[idx, ]
test <- modele_df[-idx, ]


# -----------------------------------------------------------------------------
# 2. Ajuster le modèle avec glm()
# -----------------------------------------------------------------------------

modele <- glm(achat_3m ~ frequence + montant + region + categorie,
              data = train, family = binomial)


# -----------------------------------------------------------------------------
# 3. Évaluer la performance (précision, courbe ROC)
# -----------------------------------------------------------------------------

# Probabilité prédite
proba <- predict(modele, newdata = test, type = "response")

# Classement des clients
classe <- 0
classe[proba >= 0.5] <- 1
classe <- factor(classe, levels = c(0, 1))

# Précision
mean(classe == test$achat_3m)

# Courbe ROC
roc_obj <- roc(test$achat_3m, proba, levels = c("0", "1"), direction = "<")

plot(roc_obj, main = "Courbe ROC - achat dans les 3 derniers mois")
# =============================================================================
# FIN DU SCRIPT
# =============================================================================


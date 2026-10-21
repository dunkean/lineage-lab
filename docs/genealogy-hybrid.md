# Simulation du noyau et généalogies à la demande

`full` reste le mode par défaut, avec son simulateur et ses archives existantes.
`hybrid` se choisit dans le studio (**Population**, préréglage **Hybride**),
en YAML ou par `--population-model hybrid`. L'architecture HTML existante est
conservée ; ce document décrit l'option supplémentaire.

```powershell
.\.venv\Scripts\python.exe -m cousinade.cli studio --scenario config/genealogy/hybrid.yaml
.\.venv\Scripts\python.exe -m cousinade.cli generate config/genealogy/hybrid.yaml --output output/genealogy/hybrid.sqlite
.\.venv\Scripts\python.exe -m cousinade.cli explore output/genealogy/hybrid.sqlite
```

Choisir un nouveau chemin pour chaque archive persistante. Le scénario fourni
représente 2 000 fondateurs sur 75 ans ; il illustre le fonctionnement, pas le
seuil de rentabilité du moteur hybride.

## Construction : aucune vie ordinaire simulée

Le premier prototype était plus lent parce qu'il simulait tous les habitants
par marchés puis rejouait certaines simulations pour retenir les fiches.
Ce mécanisme a été supprimé. Le format actuel est `cohort-runtime-v2` ; les
archives expérimentales du prototype précédent ne sont pas réinterprétées.

1. Une seule instance du moteur natif simule les fondateurs du noyau :
   `floor(initial_population * core_target_share)`, soit 5 % par défaut.
   Leurs naissances, décès, unions, parentés et migrations sont enregistrés.
2. Le reste est représenté par des effectifs entiers par localité, peuple,
   année de naissance et sexe. Chaque année avance ces nombres, sans boucle
   sur les personnes, sans fiche personnelle et sans table de liens familiaux.
3. Les recensements, cartes et distributions additionnent les deux populations.
   Les données conservées pour le fond sont des effectifs de cohortes, des
   facteurs annuels de régulation et des paramètres déterministes.

Le résumé expose `core_founders_simulated`, `background_persons_simulated: 0`
et `simulation_runs: 1` (ou 0 si le budget ne permet aucun couple de noyau).
Le coût du fond dépend surtout du nombre de cohortes, donc des lieux, peuples
et années, et non du nombre d'habitants. Un nombre élevé de localités coûte
plus cher même si chaque localité ne contient que quelques personnes.

Le noyau favorise les rangs élevés : pour les quatre rangs sociaux standards,
les poids sont `[0.12, 0.18, 0.60, 0.10]`, configurables par
`hybrid.core_status_weights`. Les familles nobles et bourgeoises, les notables,
les aubergistes et quelques héros communs sont identifiés par leurs rôles.
Les quotas et priorités sont configurables dans `promotion_rules` ; les rôles
n'ajoutent pas de bâtiment ni de règles de comportement. Les membres de soutien
et descendants du noyau restent simulés pour préserver les familles.

Les plafonds de vivants et de personnes historiques suivies sont contrôlés
**chaque année**, avec un maximum configurable de 10 %. Le noyau n'est pas
tronqué arbitrairement pour respecter le budget : un scénario dépassant le
plafond est refusé, et il faut diminuer sa part initiale. Les interactions entre
personnes importantes passent toutes par le même simulateur natif.

## Consultation : procédure déterministe, sans replay de population

Un habitant ordinaire possède un ID stable
`h:<empreinte du monde>:<cohorte>:<case>`. La cohorte 0 désigne le noyau.
Les IDs du fond représentent des cases virtuelles : aucun objet personnel n'a
besoin d'exister avant sa consultation.

La règle de naissance associe un ordinal d'enfant à un ordinal de couple dans
une cohorte parentale compatible. Des permutations bijectives indépendantes
répartissent les cases masculines, féminines et les enfants. La même règle
s'inverse pour retrouver les enfants d'un parent : interroger d'abord un enfant
ou son père donne exactement les mêmes relations. L'espacement des naissances
est garanti par la progression des ordinaux admissibles. L'infertilité et le
choix sans enfant sont fixés pour la vie, même si les périodes changent ensuite.

Le runtime reconstruit, si nécessaire, la courbe d'effectifs de quelques
cohortes concernées. Il ne relance jamais `Engine`, ne simule pas leurs habitants
et ne génère pas les fiches de toute la population. Seules les fiches demandées
et les proches nécessaires sont construits. Le cache de réponses personnelles
est borné par `projection_cache_mebibytes` ; les petits caches de courbes et de
quotas sont bornés à 256 cohortes chacun et sont vidés avec le cache personnel.
Ils ne sont pas des histories personnelles persistantes.

Pour le fond procédural, la profondeur est limitée à **2 générations
ascendantes** et **5 générations descendantes**, dans l'API et le studio.
`depth=1` demande uniquement le premier rang. Le noyau conserve les possibilités
d'exploration de l'historique simulé. Les ascendants précédant les fondateurs
à `start_year` restent inconnus dans les deux modes.

Les fiches restent identiques après réouverture, éviction du cache, changement
d'ordre de consultation et accès concurrents. Cela vaut pour une configuration
et une version d'algorithme données ; modifier le scénario crée un autre monde.
La sauvegarde SQLite vérifie le checksum et la version. Les anciennes archives
`full` restent lisibles. L'export dense `compact` concerne uniquement `full`.

Les routes existantes acceptent les IDs opaques du mode hybride.
`/api/core?limit=100&role=noble_house` découvre les personnages importants.
Les réponses indiquent `source: core|projected` et l'importance du noyau.
`residents` pagine avec l'ID complet du dernier résultat comme curseur `after`.
L'atlas compte toute la population et ne matérialise que son échantillon demandé.

Une ancre explicite désigne une personne déjà simulée dans le noyau :

```yaml
hybrid:
  core_target_share: 0.05
  core_max_share: 0.10
  core_historical_max_share: 0.10
  anchors:
    - {segment: 0, person: 17, role: hero, year: 1030}
```

La case doit exister et être vivante à la date choisie. Le changement d'ancre
modifie l'identité du monde, mais conserve les faits démographiques à paramètres
identiques. Promouvoir rétrospectivement n'importe quelle case du fond en une
nouvelle trajectoire causale simulée n'est pas implémenté.

## Approximation du fond

Les personnes importantes utilisent les règles détaillées existantes. Pour
réduire le coût du fond, ses couples restent dans la même localité, le même
peuple et la même année de naissance. Ils ont au plus une union ; un divorce
n'entraîne pas de remariage. Les décès sont répartis par ordinaux déterministes.
L'activité ordinaire est l'activité dominante de la localité et le rang est 0.
L'identifiant familial du fond désigne une famille proche, pas une maison noble.

Les cohortes intègrent les taux d'âge, mortalité masculine et maternelle,
fertilité, périodes, crises ciblées, divorce et régulation démographique.
Elles n'appliquent pas les migrations individuelles, croisements de peuples,
transitions sociales ni le contrôle détaillé des cousins du moteur complet.
Les deux populations n'échangent pas de partenaires. Les héros communs déjà
prévus font partie du noyau avec leurs familles simulées. Ces simplifications
expliquent que les recensements des deux modes puissent différer ; `hybrid`
ne prétend pas reproduire les vies d'une simulation `full` de même graine.
Les règles Python personnalisées ne sont pas prises en charge dans ce mode.

## Performances mesurées

Mesure locale du 1er octobre 2026, Windows, Python 3.12.13, NumPy 2.5.3,
Numba 0.67.0 ; graine 42, 16 localités, 60 ans, 4 threads.
Les durées mesurent la construction complète en mémoire, incluant compression
et préparation de l'archive hybride, sans écriture disque ni consultations.
Le JIT est préchauffé pour les deux modes ; valeurs médianes de 3 exécutions.

| Fondateurs | Full | Hybrid | Réduction du temps |
| ---: | ---: | ---: | ---: |
| 2 000 | 0,108 s | 0,139 s | −29 % (plus lent) |
| 10 000 | 0,205 s | 0,171 s | 16 % |
| 100 000 | 1,263 s | 0,266 s | 79 % |
| 1 000 000 | 15,504 s | 0,834 s | **94,6 %** |

La cible d'au moins 85 % est atteinte sur cette mesure à un million de
fondateurs. Elle n'est pas garantie pour toute taille ni tout scénario : les
coûts fixes du simulateur du noyau et des cohortes dominent les petites villes.
Le mode `full` reste préférable pour une petite population si seul le temps
compte. Ces chiffres ne mesurent ni une réduction de RAM ni la latence des
consultations ; ne pas les réutiliser comme garanties de stockage ou de mémoire.

Pour reproduire la comparaison :

```powershell
.\.venv\Scripts\python.exe -m cousinade.benchmark --compare --population 1000000 --places 16 --years 60 --repeats 3
```

Le benchmark individuel, sans `--compare`, vérifie les bilans démographiques et
budgets et mesure des consultations à froid. Les tests de régression vérifient
notamment qu'un seul `Engine` reçoit uniquement les 5 % du noyau et qu'aucune
consultation ou réouverture ne relance ce moteur. Les autres tests couvrent les
liens inverses, profondeur, espacement, traits reproductifs, ordre des lectures,
concurrence, pagination, archives, annulation et les deux modes du studio.
Le test navigateur `tests/browser/genealogy_hybrid.cjs` produit ses captures
dans `output/genealogy/screenshots/`, répertoire ignoré.

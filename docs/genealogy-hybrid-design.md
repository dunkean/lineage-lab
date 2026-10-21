# Généalogie à deux échelles : simulation complète ou hybride

Le moteur disponible simule uniquement le noyau important, utilise des cohortes
agrégées pour le reste et calcule les généalogies ordinaires à la demande.
Le [guide d'utilisation](genealogy-hybrid.md) décrit son périmètre et les mesures
réelles. Le prototype de replay par marchés a été remplacé, car il était plus lent.
Pour les personnages procéduraux, la demande utilisateur limite les ascendants
à 2 générations et les descendants à 5 ; un rang 1 reste possible. Les exigences
plus ambitieuses du design ci-dessous restent des pistes, sans prévaloir sur
ces limites ni constituer des fonctionnalités déjà réalisées.

**Statut :** design cible, avec une première implémentation expérimentale.
**Base examinée pour la conception :** `main`, commit
`b6e9142` du 1er octobre 2026 (« Document genealogy architecture and simulation
rules in standalone HTML »), ainsi que
[l'anatomie du prototype](genealogy-architecture.html). Le module concerné est
`src/cousinade/`, indépendant du pipeline LLM de `src/pipeline.py`. Les noms de
classes et champs ci-dessous décrivent la cible ; le guide d'utilisation
distingue ceux disponibles de ceux restant à implémenter.

## Décision et garanties

Ajouter un choix explicite entre :

| Mode | Source des fiches | Usage |
| --- | --- | --- |
| `full` (défaut, actuel) | Toute personne est simulée et archivée ; parents et événements sont des faits enregistrés. | Mondes où l'exactitude individuelle complète justifie le coût. |
| `hybrid` (proposé) | Démographie entière en cohortes ; un noyau de personnes importantes est simulé explicitement ; les autres ont des identités et des liens **fixés par un plan déterministe**, reconstruits à la demande. | Grandes populations où l'on veut explorer n'importe quel habitant sans garder une fiche par habitant. |

Une même graine et un même scénario en mode hybride doivent rendre le même monde,
quel que soit le nombre, l'ordre ou la concurrence des requêtes, le redémarrage du
serveur, la taille des lots de lecture et l'état du cache. Une relation publiée
ne change jamais au gré des consultations. La garantie porte sur une version
précise des algorithmes et des paramètres, **pas** sur l'identité des résultats
des modes `full` et `hybrid` : leurs taux réalisés et leurs histoires peuvent
différer. `backend: compiled|reference` désigne aujourd'hui deux exécutants du
mode complet ; il ne doit pas servir à sélectionner le nouveau modèle.

« Généalogie complète » signifie tous les parents, unions, enfants et
ascendants **dans la période simulée**, y compris les branches ordinaires
nécessaires à la lignée d'un notable. À `start_year`, les parents des fondateurs
sont inconnus, comme actuellement. Une histoire dynastique antérieure exige des
ancres historiques saisies et validées, ou un début de scénario plus ancien.
Les endpoints peuvent tronquer l'affichage à une profondeur et un nombre de
nœuds, avec indicateur et pagination ; ils ne doivent pas inventer un parent
pour combler la coupure.

## Ancrage dans le code présent

`Scenario` dans `src/cousinade/config.py` valide les scénarios Pydantic et
refuse les champs inconnus ; sa version de schéma est aujourd'hui `1`.
`Engine` dans `src/cousinade/engine.py` crée des IDs `int32` denses, conserve
chaque naissance et chaque mort dans `PersonColumns` (`schema.py`), puis traite
annuellement naissances, décès, divorces, unions, migrations et recensement.
`MemoryArchive` (`memory.py`) copie les colonnes de **toutes** les personnes,
les unions et migrations ; `Store`/`Archive` (`store.py`) offrent un SQLite
complet optionnel. `compact.py` suppose également des IDs denses et une ligne
par personne : ce format n'est pas une archive hybride. Les traits reproductifs
peuvent déjà être reconstruits par graine et identité (`reproduction.py`),
mais la généalogie actuelle reste entièrement stockée.

L'exploration passe par `Studio` (`studio.py`), `server.py` et `studio.js` :
`overview`, `person`, `lineage`, `residence`, `residents`, `resident_atlas`,
`map_at`, `distributions`. Les cartes et distributions viennent des recensements ;
`resident_atlas` calcule actuellement ses groupes exacts sur tous les IDs vivants
et n'échantillonne que les marqueurs. Le mode hybride doit fournir les mêmes
contrats de lecture avec des cohortes et des sélections d'identifiants, sans
faire semblant que l'échantillon constitue la population. La publication d'un
monde terminé, le choix d'archive et l'annulation en fin d'année dans `Studio`
restent les frontières transactionnelles utiles.

## Modèle de données hybride

Une **cohorte** est une partition disjointe de personnes par lieu, année de
naissance (ou classe d'âge des fondateurs), peuple, sexe et éventuellement niveau
social et activité. Sa taille, les décès, migrations, unions et naissances
alloués par année forment le registre démographique. Les recensements et cartes
sont calculés sur `noyau + cohortes`, avec un seul propriétaire pour chaque
personne ; une promotion déplace un compte du fond vers le noyau. Garder aussi
les marges par âge et lieu nécessaires aux filtres de l'atlas et au contrôle des
capacités. Une nationalité personnelle ne doit pas être déduite du propriétaire
politique du lieu : le prototype n'en a pas.

Une **case latente** est la place immuable d'une personne dans une cohorte et
un segment (par exemple 256 cases). Son identité opaque encode ou référence
`world_id`, `cohort_id` et `slot`. Les nombres de slots sont figés lors du scellement
du monde ; aucune requête ne crée de nouvel ID. Les IDs hybrides sont des chaînes
ASCII typées, par exemple `h:<world>:<cohort>:<slot>` ; les IDs numériques des
archives complètes restent acceptés. Ne pas convertir un ID hybride en `Number`
JavaScript ni le limiter à l'`int32` de `schema.py`. Une promotion conserve l'ID
de la case, elle ne crée pas un second personnage. Un identifiant de famille
désigne une lignée ou un graphe de parenté construit ; le champ `family` actuel
est seulement patrilinéaire et ne peut pas représenter à lui seul les deux
ascendances, les remariages ou les familles recomposées.

La clé de cohorte est un tuple canonique complet, non un hash tronqué ; le rang
de slot est unique dans cette clé. L'encodeur d'ID doit être injectif et le
lecteur vérifier l'appartenance au manifeste. Une empreinte courte de monde
peut être utilisée pour l'affichage, mais une collision avec deux manifestes
différents doit être détectée et rejetée. Les noms et patronymes sont dérivés
du lieu, de la culture ou famille et d'un domaine PRF dédié ; plusieurs
personnes peuvent légitimement porter le même nom. La recherche et les liens
utilisent toujours l'ID, jamais le nom. Un ancêtre commun conserve le même ID
dans toutes les branches et l'arbre déduplique les nœuds rencontrés à plusieurs
profondeurs sans masquer le chemin le plus court.

Le **plan de filiation** est un index compact immuable de quotas et de règles
d'appariement par lieu, année et segment. Il contient les ancres explicites du
noyau, les réservations de liens noyau↔fond, les comptes d'événements, les
versions d'algorithme, les permutations ou leurs graines et les exceptions
rares. Il ne contient pas une fiche ou une paire de parents stockée pour chaque
personne ordinaire. Les faits projetés d'une case (naissance, décès, parents,
épisodes d'union, résidences) sont dérivés de ce plan. Une projection peut être
gardée dans un cache borné, jamais utilisée comme nouvelle source de vérité.

Trois volumes doivent être publiés séparément dans les statistiques :

1. **Noyau vivant** : personnes dont la vie participe explicitement au cycle
   annuel, avec état individuel mutable pendant la construction du monde.
2. **Noyau historique** : personnes importantes décédées ou sorties du monde,
   dont les faits explicites demeurent dans l'archive. Les ascendants ordinaires
   affichés dans leur arbre restent des cases latentes.
3. **Cache de projections** : objets ordinaires reconstruits pour la lecture,
   limités par octets et nombre d'entrées, évictables sans effet sur les faits.

Les quotas, index de segments et exceptions ne sont pas des fiches personnelles,
mais leur taille RAM/disque doit être mesurée. Si l'on exigeait de conserver
physiquement tous les ancêtres ordinaires de toutes les familles importantes,
un plafond strict de 10 % de fiches serait impossible sur certains graphes.
La solution est de conserver leurs identités et liens **implicitement** dans le
plan et d'en projeter les fiches seulement pour une consultation.

## Construction canonique, indépendante des requêtes

Le scellement du monde a quatre responsabilités logiques, et non quatre passes
indépendantes sur toute la chronologie. Les trois premières avancent
**année par année** : leurs quotas sont proposés puis validés à chaque année,
et les comptes effectivement réalisés alimentent l'année suivante. On ne fige
pas tous les nombres de naissances futurs avant de connaître les unions et les
survivants qui les rendent possibles.

À chaque frontière annuelle, le noyau et les cohortes utilisent le même bilan
de lieux et les mêmes restrictions politiques. Les promotions préparées et
leurs réservations sont appliquées d'abord. Suivre ensuite l'ordre actuel :
naissances des unions déjà existantes (noyau et fond, puis ponts réservés),
décès avec nouveau-nés et décès maternels, dissolutions, nouvelles unions,
migrations de ménages, puis recensement. Les nouvelles unions ne produisent pas
d'enfant dans leur année de formation. À chaque sous-étape, soustraire les
personnes du fond lorsqu'elles sont promues, y compris un enfant promu dès sa
naissance. Un partenaire qui reste ordinaire lors d'un pont demeure dans sa
cohorte : réserver seulement sa case et ses événements, puis débiter les quotas
correspondants. Chaque vie et chaque événement est compté une seule fois.
Réconcilier le bilan des vivants (début + naissances − décès = fin) ainsi que
les flux par lieu avant d'avancer.

1. **Démographie** : proposer pour l'année les comptes par cohortes, lieux
   et catégories, en appliquant périodes, crises, capacité, frontières et cible
   de population. Définir les quotas de décès, naissances, unions et migrations
   avant d'allouer des personnes. Les contraintes de faisabilité peuvent réduire
   un quota demandé ; consigner le manque au lieu de créer un parent impossible.
2. **Marchés et familles** : découper les candidats en segments ordonnés par
   clés stables ; réserver d'abord les unions/filiations explicitement imposées
   par le noyau, puis apparier les cases ordinaires avec un tri et une
   permutation déterministes par année, lieu, type d'événement et segment.
   Résoudre les candidats dans un ordre canonique, jamais dans l'ordre des
   requêtes HTTP ni de parcours des threads. Les unions intersegments, hors
   lieu et noyau↔fond sont des ponts rares, explicitement inscrits dans le plan.
3. **Allocation inverse et validation annuelle** : pour chaque quota de
   naissance, choisir une mère éligible dans une union déjà établie, avec
   espacement depuis ses naissances
   précédentes, père vivant et politique de croisement valide. La case enfant
   est attribuée au couple par rang canonique. De même, allouer décès,
   dissolutions et déplacements sans contradiction d'âge ou de présence. Écrire
   les comptes réellement obtenus et index nécessaires pour recalculer *les deux
   sens* des liens. Réconcilier les cohortes avant l'année suivante.
4. **Scellement et audit** : vérifier les comptes, les contraintes de parenté
   et les ponts, calculer l'empreinte du manifeste, puis publier l'archive de
   façon atomique. L'exploration est alors purement en lecture. Un échec ou une
   annulation ne publie aucun monde partiel.

Il ne suffit **pas** de hacher chaque personne indépendamment pour lui choisir
des parents : deux enfants pourraient réclamer la même mère à une date
impossible, une mère pourrait nier son enfant et un mariage pourrait être
réécrit par une requête concurrente. Le tirage est une fonction d'un **marché
commun scellé**. Conceptuellement :

```text
P = couples_éligibles(marché, année, plan_antérieur)
P = tri_stable(P, PRF(graine, version, marché, année, "naissance", couple_id))
enfants_année = quotas_validés attribués à P, avec espacements
parents(enfant_slot) = attribution_inverse(marché, année, rang_enfant)
enfants(parent_slot) = union des rangs attribués à ce parent dans ses marchés
```

Le même mécanisme vaut pour les unions : un appariement bijectif des candidats
disponibles, avec exclusion de parenté et de doublons, puis épisodes datés. Un
candidat rejeté ne fait pas réapparier silencieusement toutes les autres cases
par une future lecture ; le résultat du marché est fixé à la construction.
Un index de segments donne les marchés et années à inspecter pour les enfants,
unions et migrations d'un ID. Les sorties volumineuses sont paginées. Construire
le plan peut examiner temporairement des candidats ordinaires par blocs ; la
promesse porte sur l'absence d'un **état et d'une archive individuels durables**
pour ces candidats, pas sur une absence absolue de calcul par individu.

Cette projection exige un état de dépendance implicite : disponibilité des
partenaires, naissance précédente de la mère, vie ou décès et résidence au
moment de l'événement. Pour une fiche, rejouer les marchés nécessaires depuis
la frontière de fondation ou un checkpoint **de segment** contenant quotas et
résumés compacts, puis ne garder que les projections demandées. Un checkpoint
facultatif peut contenir des bitmaps ou rangs d'état, mais son coût doit être
compté dans le plan ; s'il devient équivalent à une fiche par personne, il
invalide l'objectif de compacité. Les ponts entre segments forment des
dépendances explicites et indexées ; éviter une cascade non bornée en limitant
les marchés ordinaires et en mesurant la taille de leur fermeture de replay.
Le calcul et les exceptions pourraient malgré tout devenir `O(N)` et coûter
presque autant que le mode complet. Ce point est un **risque technique à
résoudre expérimentalement**, pas une économie de RAM déjà démontrée.

Utiliser une PRF spécifiée (par exemple BLAKE2b avec encodage canonique UTF-8
des champs, tailles et signe des entiers) et des domaines distincts
`fondateur`, `mariage`, `naissance`, `deces`, `migration`, `role`, `nom`.
Documenter l'ordre de tri et le départage. Ne pas utiliser `hash()` de Python
ni dépendre de l'ordre des dictionnaires, du parallélisme, du cache, du nombre
de requêtes ou des tirages de `Engine.rng`. `Engine._random_stream()` isole déjà
ses processus par année ; le nouveau moteur a besoin de cette isolation **et**
d'une adressabilité par case/marché. Même graine + scénario normalisé + version
du plan + ancres ⇒ mêmes résultats. Changer la version d'une règle crée un
nouveau monde identifié par une nouvelle empreinte.

## Réalisme et liens avec le noyau

Les règles de parenté contrôlent avant chaque union les ascendants communs
jusqu'à `society.kinship_depth` ; les IDs des fondateurs inconnus ne constituent
pas un ancêtre partagé. Le test suit le graphe des deux parents et les ponts,
pas seulement `family`. Une personne ne peut avoir qu'un père et une mère
biologiques dans ce modèle, avec dates cohérentes ; un remariage forme un nouvel
épisode sans réattribuer les enfants. Si adoption, père social ou unions de même
sexe sont ajoutés plus tard, les liens biologiques, sociaux et les règles de
reproduction devront être séparés dans un schéma versionné.

Pour les personnes ordinaires, utiliser les taux de `Demography` et `Society`
comme contraintes statistiques : survie par âge/peuple, fenêtres de fertilité,
infécondité, choix sans enfant, espacement, écarts d'âge, statut, compatibilité
des peuples, pression de capacité, distances, contacts politiques et crises.
La sélection par quotas respecte ces contraintes et borne les tentatives. Si
elles ne permettent pas le compte demandé, privilégier une histoire réalisable,
publier l'écart et réconcilier le recensement. Les décès néonataux peuvent avoir
année de naissance = année de décès ; les parents sont vivants au début de
l'année de naissance. Conserver l'ordre annuel présent du prototype. Les
fondateurs n'ont ni parents ni migrations fabriqués avant `start_year`.

Le noyau reçoit un **score de personnage** et un **score de famille** avec
raisons traçables : rang social, office, influence locale, métier clé, rôle
narratif, liens politiques, nombre de générations pertinentes et liens au noyau.
Les nobles, lignées bourgeoises, notables, aubergistes et héros issus du commun
peuvent ainsi y entrer sans assimiler « important » à « noble ». Sélectionner
les familles ancrées avant les personnages isolés, puis les partenaires et
descendants dont les événements doivent réellement influencer la simulation.
Une famille reste importante si son score franchit un seuil durant plusieurs
années ; prévoir une hystérésis afin d'éviter des promotions/déclassements
oscillants. Un décès retire l'état vivant mais conserve les faits historiques.
Le score, la règle et l'année de promotion sont archivés.

**Budget ferme :** cible configurable à 5 % des vivants, plafond de 10 % par
année, plus plafond de 10 % des personnes nées depuis le début pour les fiches
historiques explicites. Le dénominateur initial inclut tous les fondateurs ; le
dénominateur historique inclut fondateurs et naissances. Une promotion réserve
le budget **avant** tout nouveau fait. Priorité stable aux ancres imposées,
puis aux scores ; égalités par ID canonique. Les parents ordinaires nécessaires
à l'arbre d'un héros restent virtuels et ne consomment pas le budget de noyau.
Si des ancres imposées excèdent le plafond, refuser le scénario avec le compte
et les causes, plutôt que violer discrètement la promesse. Le cache a un autre
plafond, par exemple 64 Mio par monde, et son remplissage ne change aucun
quota. À petite population, utiliser `floor(0,10 × population)` et signaler
si aucun personnage ne peut être ancré ; une exemption « au moins un » doit
être explicite si souhaitée, car elle contredit le plafond strict.

Une **promotion préparée** lors de la construction reprend l'ID et tous les
faits déjà fixés dans le plan. Elle ajoute uniquement des états futurs du noyau
à partir d'une année de frontière ; les événements antérieurs et les parents
restent immuables. Les réservations des marchés futurs prennent l'acteur en
compte avant le scellement. Une personne choisie par un utilisateur **après**
publication peut être épinglée ou annotée dans une couche de présentation,
sans modifier la simulation. Si l'on veut que cette personne change causalement
des événements passés/futurs, créer une branche du monde à un checkpoint
antérieur, recalculer et lui donner un nouveau `world_id` ; une requête de
lecture ne doit jamais provoquer cette branche automatiquement.

Exemple : au village 42, la tenancière d'auberge est choisie comme rôle clé au
départ ; ses parents ordinaires sont des cases latentes, mais ses unions et
enfants sont consultables et ses affaires pèsent dans les événements futurs du
noyau. Un enfant de cultivateurs devient héros en 1034 : son ID, sa naissance,
ses parents et ses frères déjà fixés ne changent pas ; seul son état explicite
à partir de 1034 et son score évoluent. En 1041, il épouse une héritière noble :
le marché réserve le pont commun↔noble, vérifie les ascendants communs, inscrit
l'union des deux côtés et attribue leurs enfants au couple. Une requête « fiche
de l'héritière » avant « fiche du héros » renvoie exactement la même union,
les mêmes enfants et les mêmes ancêtres. Si cette histoire est imposée après
scellement sans pont prévu, elle exige un nouveau monde branché.

## Contrats, intégration et coexistence

Emplacements proposés, sous réserve du prototype algorithmique :

| Fichier | Responsabilité |
| --- | --- |
| Nouveau `src/cousinade/archive_protocol.py` | Contrat de lecture commun et type d'ID opaque. |
| Nouveau `src/cousinade/cohorts.py` | Comptes annuels, segments et réconciliation des recensements. |
| Nouveau `src/cousinade/hybrid_engine.py` | Cycle annuel, marchés, ponts et budget du noyau. |
| Nouveau `src/cousinade/projection.py` | PRF versionnée, replay borné, fiches et filiations dans les deux sens. |
| Nouveau `src/cousinade/hybrid_archive.py` | Manifeste, stockage RAM/disque, vérification et cache borné. |
| `src/cousinade/config.py`, `engine.py`, `cli.py` | Choix validé du mode, routage de génération et commandes. |
| `src/cousinade/studio.py`, `server.py`, `studio.js` | Publication, sélection d'archive, IDs chaînes et vues. |
| `src/cousinade/memory.py`, `store.py`, `compact.py` | Adaptation au protocole pour la lecture complète ; formats v1 préservés. |


Introduire un protocole `GenealogyArchive` couvrant les méthodes de lecture
actuelles et une propriété `identity_kind`/`population_model`. `FullArchive`
adapte `MemoryArchive` et `Archive` sans changer leurs données. Un
`HybridArchive` nouveau expose les mêmes agrégats et projette les personnes.
Un routeur de génération choisit `Engine` ou `HybridEngine` avant le calcul ;
il ne transforme jamais une archive complète en archive hybride pendant une
lecture. `Studio` garde sa sélection par ID de monde ; `server.py` délègue les
lectures au protocole et parse les IDs comme chaînes opaques dans les routes
personnelles. `studio.js` conserve ces chaînes de bout en bout, y compris
attributs DOM, sélection, arbre et pagination. La CLI `person` accepte aussi
une chaîne ; `generate`/`simulate` dispatchent selon le scénario. Garder
`src/cli.py`, `src/pipeline.py`, `src/cache/file_cache.py` et
`src/output/renderer.py` hors du chemin critique : ce sont l'autre pipeline,
son cache de réponses LLM et son rendu HTML. Un raccordement ultérieur des
figures clés du générateur de communautés doit être un adaptateur explicite,
avec mapping de lieux, rôles et IDs, sans dépendance implicite.

Les réponses existantes gardent leurs champs utiles. Ajouter `source` (`core` ou
`projected`), `history_known_from`, `lineage.truncated`, un curseur stable pour
les enfants/descendants, et les compteurs de couverture. `residents` parcourt
les slots vivants dans un ordre canonique, sans les matérialiser tous ;
`resident_atlas` calcule `total`, `matched` et `groups` sur les comptes exacts
du registre et choisit les marqueurs par rang stable. `map_at` et
`distributions` demeurent des lectures de snapshots complets. La page HTML de
documentation existante reste intacte ; l'interface du studio devra afficher
clairement « histoire complète » ou « cohorte + noyau », et la frontière des
fondateurs.

Exemple de **nouvelle** configuration (syntaxe indicative à implémenter dans
`Scenario`, pas encore acceptée par le parseur) :

```yaml
schema_version: 2
population_model: hybrid
seed: 42
initial_population: 1000000
years: 150
hybrid:
  core_target_share: 0.05
  core_max_share: 0.10
  core_historical_max_share: 0.10
  segment_size: 256
  projection_cache_mebibytes: 64
  promotion_rules:
    - {role: noble_house, priority: 100}
    - {role: bourgeois_lineage, priority: 80}
    - {role: local_notable, priority: 70}
    - {role: innkeeper, per_settlement: 1, priority: 60}
    - {role: common_hero, per_world: 4, priority: 90}
```

Les scénarios v1 restent `full` par défaut. `Scenario.schema_version` doit
accepter v2 avec validation des champs hybrides, bornes 0–0,10, politique
d'ancres et compatibilité des peuples. Refuser les paramètres hybrides quand
`population_model: full` plutôt que les ignorer. Conserver `backend` pour le
choix `compiled|reference` en mode `full` ; en mode `hybrid`, accepter sa valeur
implicite héritée `compiled`, mais refuser une valeur explicitement fournie et
documenter que ce champ n'y sélectionne aucun moteur.

Versionner séparément le schéma du scénario, le moteur de projection, la PRF,
le format du plan et l'archive. L'empreinte `world_id` doit dépendre de la
configuration normalisée, des ancres, des versions et de la graine ; elle ne
dépend ni du chemin de sortie ni de l'heure. Stocker le manifeste, ses sommes
de contrôle et les agrégats avant de déclarer `complete=true`. Pour une archive
hybride persistée, utiliser un format distinct, par exemple un dossier avec
manifeste et blocs binaires ou SQLite v2 aux tables distinctes. Ne pas la faire
passer pour le `Store` v1 ni pour `compact.py`, qui attendent une ligne par ID
dense. Le studio RAM peut suivre le même protocole ; cache de projection
éphémère par archive, verrouillé pour les requêtes concurrentes. Une
réouverture vérifie les versions et empreintes avant d'accepter les IDs.

**Ordre de lecture :** valider `world_id` et ID → trouver slot ou ancre → lire
plan immuable et ponts → projeter les faits → vérifier la cohérence locale →
mettre éventuellement en cache → sérialiser. Une entrée de cache est indexée
par `(world_id, projection_version, person_id, vue)` ; aucun cache négatif
durable pendant la construction. Le cache peut être vidé ou réchauffé dans
n'importe quel ordre sans changer les octets des réponses canoniques.

## Plan d'implémentation et vérification

1. Extraire le contrat de lecture et les tests de parité de réponse des
   archives complètes ; garder `full` comme défaut. Faire accepter les IDs
   opaques par serveur, CLI et UI avant d'ajouter le moteur hybride.
2. Ajouter `Scenario` v2, quotas et registre de cohortes, identités, PRF et
   manifeste. Vérifier totaux annuels, snapshots, filtres et politique des
   fondateurs sur petits scénarios à graine fixe.
3. Construire les marchés canoniques, l'allocation inverse des parents/enfants,
   unions, décès et migrations ; tester chaque invariant avant de permettre la
   projection publique. Mesurer le coût de reconstruction des descendants.
4. Ajouter le score et le budget du noyau, réservations et ponts ; brancher les
   promotions préparées, puis les vues `person`, `lineage`, `residence`,
   `residents` et `resident_atlas`. Refuser le dépassement d'ancres explicites.
5. Ajouter archive persistante versionnée et ouverture contrôlée ; intégrer au
   studio, à la CLI et enfin aux contrôles UI. Garder un chemin de repli complet
   et tester l'ouverture d'archives v1 sans migration destructive.

Tests nécessaires (`pytest`, graines fixes, `tmp_path`, sans service externe) :
ordre de requêtes inversé et concurrent, cache vide/plein/évincé, redémarrage ;
parent↔enfant dans les deux sens, ancêtre partagé, remariage, orphelin,
naissance et décès la même année, mariage noyau↔fond, frontière de fondation,
peuples croisés, migrations et frontières politiques ; comptes de cohortes =
cartes = atlas = recensements ; plafond du noyau pour chaque année et cumul
historique ; refus d'ancres impossibles ; versions et IDs incompatibles ;
annulation sans archive publiée. Ajouter un test de scénario aubergiste/héros/
héritière et un benchmark grand monde mesurant RAM maximale, taille du plan,
latence p50/p95 des fiches et arbres, temps de génération et taux de quotas
ajustés. Tester `full` pour non-régression avec les tests existants ; ne pas
exiger une parité bit à bit entre `full` et `hybrid`.

Jalon de décision après le prototype des marchés : mesurer un scénario d'un
million de fondateurs sur 150 ans à côté du mode complet, avec la même machine,
et publier RAM maximale (y compris checkpoints/cache), taille d'archive,
temps de scellement, latence p95 d'une fiche, d'un arbre à quatre générations
et d'une liste de résidents, ainsi que le nombre d'exceptions et la fermeture
de replay maximale. Critères de départ à viser : RAM et disque au plus 30 % du
mode complet, aucune dérive du plafond explicite de 10 %, p95 de fiche sous
une seconde après ouverture du monde. Si ces seuils échouent, optimiser le
partitionnement ou réduire les garanties de détail affichées avec accord
explicite ; ne pas présenter le mode hybride comme prêt pour grandes
populations sur le seul argument du pourcentage de noyau.

## Compromis explicites

Le modèle hybride garantit une histoire **constructivement cohérente** dans
son propre registre, mais les taux individuels réalisés et interactions de
fond sont des approximations de cohorte. Les ponts rares et marchés locaux
limitent la combinatoire ; ils peuvent réduire la diversité des migrations et
alliances du fond par rapport au mode complet. Les requêtes d'arbres profonds
peuvent demander beaucoup de recalcul ; pagination, index de segments et cache
borné contrôlent la latence, pas la vérité. Le volume du plan et la phase de
scellement peuvent rester significatifs pour une civilisation immense : le
budget de 5–10 % porte sur les **personnes explicitement simulées/archivées**,
et doit être accompagné de mesures réelles de mémoire, temps et stockage.

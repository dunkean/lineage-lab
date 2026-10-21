# Vérification de la version autonome

Vérifications locales du 21 octobre 2026, Windows, Python 3.12.13. Ces contrôles concernent la démo et le paquet extraits ; les grandes mesures historiques restent dans les autres documents.

- 108 tests Python : modes complet/hybride, cohérence des filiations et événements, déterminisme, archives, concurrence, réglages, presets hors du checkout et ressources embarquées.
- Ruff : source et tests passent.
- Parcours Edge sans interface : génération, 500 lieux, nations datées, exploration, configuration JSON, formulaire mobile sans débordement ; mode hybride, IDs opaques, profondeur des familles et retour au mode complet.
- Construction wheel/sdist ; installation du wheel dans un environnement vierge hors du checkout avec actifs et scénarios embarqués.
- Les benchmarks à plusieurs millions de personnes n'ont pas été relancés pour cette extraction.

Reproduire avec les commandes de README.md. Aucun lien avec le projet RPG source ou un service LLM n'est requis.

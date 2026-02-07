<!-- Streamlit-friendly Markdown/HTML.
     IMPORTANT: Wording is kept identical to the provided LaTeX input (content copied verbatim; only wrapping/markup added). -->

<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['
(', '
)']], displayMath: [['

$$
','
$$

'], ['
[','
]']] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
/* Fix cropped list numbering in Streamlit markdown render */
ol, ul { padding-left: 2.0em !important; margin-left: 0 !important; }
ol { list-style-position: outside !important; }
li { overflow: visible !important; }
</style>


<div>
  <div style="font-weight:700; font-size:1.6em;">
    Supplementary Materials for “<em>Language Embeddings Meet Shallow Autoencoders</em>”
  </div>
</div>

---

## Baselines

As baselines to compare our models, we considered the following:

- **Random Baseline**: This baseline recommends items for each user by selecting them uniformly at random.

- **Inductive Matrix Factorization (IMF) (Xu et al., 2013; Chiang et al., 2018; Ledent et al., 2021)**: Inductive matrix factorization is a family of learning models widely used for leveraging features while factorizing a sparse observed matrix. Based on (Xu et al., 2013), we factorize matrix $X = UMS^\top$ with predictor $\bar{X} \approx UM\bar{S}^\top$. The rank constraint imposed by the hyperparameter $d$ ($M\in \mathbb{R}^{d \times r}$) was cross-validated in the range $\{10,20,50,100\}$.

- **Nearest Neighbor-based Methods (NN)** and **Content-based Recommendation (CBR)**: Algorithms based on nearest neighbors are widely used for content-based filtering, and because of that, also for recommending cold-start items (Baeza-Yates et al., 2015; Tu et al., 2019; Anwaar et al., 2018). NN and CBR first train a collaborative filtering method only considering warm-start items, returning the estimated matrix $\hat{X}$. In the prediction step, NN then uses the side information to find the most similar warm-item to the cold item for which the recommendation is requested. The output score for the cold-start item is copied to the most similar warm item. On the other hand, CBR computes a warm-start to cold-start matrix similarity $M \in \mathbb{R}^{n \times \bar{n}}$, with entry $M_{i,j}$ being the side information cosine similarity between warm-start item-$i$ and cold-start item $j$. The predictor is $\bar{X} = \hat{X}M$. We use a collaborative filtering method (Vančura et al., 2022) as the backbone, implemented with the Pytorch backend.

- **Heater (Zhu et al., 2020)**: Heater is a model engineered for the cold-start problem, combining separate training and joint-training frameworks to extract feature representations for cold-start users and items. We implement the model with Pytorch as backend and only considering side information for items.

- **MTPR (Du et al., 2020):** MTPR is a model that optimizes dual item representations, incorporating both normal and counterfactual representations, aiming to address the training-testing gap for cold-start items. Hyperparameter selection is done according to Section 4.1.4 of the original paper (Original code: https://github.com/duxy-me/MTPR).

- **CCFCRec (Zhou et al., 2023):** A contrastive learning-based framework proposed for addressing cold-start recommendation challenges. This framework leverages co-occurrence collaborative signals within warm training data to mitigate the problem of unclear collaborative embeddings for recommending cold-start items. The experimental setting follows Section 4.1 of the original paper and the implementation adapts the source code of the authors (Original code: https://github.com/zzhin/CCFCRec).

- **TEASER (De et al., 2022):** A hybrid shallow autoencoder-based recommender system that combines the strengths of collaborative and content-based filtering. It utilizes implicit feedback interaction data to learn item similarities from attributes. The method is particularly suited for addressing the cold-start problem, as it can leverage item metadata alone to make recommendations in the absence of sufficient user interaction data. We implement the model with PyTorch as the backend, considering only side information for items.

- **LAE   (Linear Autoencoder) (Moon et al., 2023):** A simple linear autoencoder for abalation  without any constraints or diagonal gating.

<div style="text-align:center; font-size:0.8em; color:#666; margin-bottom:1em;">Table 1: Summary of the used datasets' statistics after preprocessing.</div>
<div style="display:flex; justify-content:center;">
<table style="border-collapse:collapse; width:100">
  <thead>
    <tr>
      <th style="text-align:left; border-bottom:1px solid #999; padding:6px 10px;">Dataset</th>
      <th style="text-align:right; border-bottom:1px solid #999; padding:6px 10px;"># users</th>
      <th style="text-align:right; border-bottom:1px solid #999; padding:6px 10px;"># items</th>
      <th style="text-align:right; border-bottom:1px solid #999; padding:6px 10px;">interactions</th>
      <th style="text-align:right; border-bottom:1px solid #999; padding:6px 10px;">density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:6px 10px;">MovieLens</td>
      <td style="text-align:right; padding:6px 10px;">136 677</td>
      <td style="text-align:right; padding:6px 10px;">20 720</td>
      <td style="text-align:right; padding:6px 10px;">9.99 M</td>
      <td style="text-align:right; padding:6px 10px;">0.35 
    </tr>
    <tr>
      <td style="padding:6px 10px;">GoodBooks</td>
      <td style="text-align:right; padding:6px 10px;">53 366</td>
      <td style="text-align:right; padding:6px 10px;">10 000</td>
      <td style="text-align:right; padding:6px 10px;">4.12 M</td>
      <td style="text-align:right; padding:6px 10px;">0.77 
    </tr>
    <tr>
      <td style="padding:6px 10px; border-bottom:1px solid #999;">Amazon Electronics</td>
      <td style="text-align:right; padding:6px 10px; border-bottom:1px solid #999;">73 411</td>
      <td style="text-align:right; padding:6px 10px; border-bottom:1px solid #999;">33 126</td>
      <td style="text-align:right; padding:6px 10px; border-bottom:1px solid #999;">1.40 M</td>
      <td style="text-align:right; padding:6px 10px; border-bottom:1px solid #999;">0.05 
    </tr>
  </tbody>
</table>
</div>

We note that recent prompting-based methods and transformer fine-tuning approaches (e.g., (Sanner et al., 2023; Wu et al., 2024; Vančura et al., 2024)) do not yet scale to the large user and large catalog settings considered here, and are therefore not directly comparable. For our model, the hyperparameter $\lambda \in \{10^{-2}, 10^{-1},\cdots ,10^{5}\}$ was selected.

<div style="text-align:center; font-size:0.8em; color:#666; margin-bottom:1em;">Figure 1: NDCG@100 performance on the transition from cold-to-warm status recommendation organized by datasets.</div>
 <img src="app/static/warming.png" style="width:75">


## Datasets

We will briefly describe the datasets used for evaluation, which are widely recognized as benchmark datasets for recommender system evaluation: Amazon Electronics, Goodbooks, and MovieLens. All the datasets were originally collected with explicit feedback, e.g., one-to-five star user-item ratings. Following a standard procedure in recommender systems research (Liang et al., 2018), we consider only ratings of four or higher as positive samples of interactions, dropping the others. A short summary of dataset is available in Table 1.

- **Amazon Electronics**: This dataset belongs to the online shopping domain (Ni et al., 2019). We preprocessed the dataset to include only users and items with at least ten interactions. Since the original dataset already contains item titles and descriptions, we encoded them into vectors using the sentence transformers library (Reimers et al., 2019) and used them as side information.


- **GoodBooks (Zajac, 2017)**: Goodbooks belongs to the book recommendation domain. To get side information for the items, we used available book descriptions (we refer to (Bajaj, 2022; Dhamani, 2021; Reese, 2020)) and encoded them, along with the title, using models from the sentence transformers library (Reimers et al., 2019).

- **MovieLens (ML-20M) (Harper et al., 2015)**: For the MovieLens dataset, we only keep users with at least five interactions. For side information, we collected additional data for movies in the dataset from (Mhatre, 2020) and (Mudigoudr, 2024): we extracted categorical features such as actors, director, year of release, and language, which we combined with genres from the original dataset.

Nowadays, the use of embeddings based on large-language models is pervasive, with a large number of model options available. To select a model for generating embeddings for our analysis, we first selected three popular models: <em>all-MiniLM-L12-v2</em> (https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), <em>all-mpnet-base-v2</em> (Available in: https://huggingface.co/sentence-transformers/all-mpnet-base-v2), and <em>distiluse-base-multilingual-cased-v2</em> (Available in: 
  https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) from the sentence transformers (Reimers et al., 2019) library and evaluated them with CBR. We observed that <em>all-mpnet-base-v2</em> consistently outperformed its counterparts. Therefore, we used this model to create the embeddings that serve as side information for all other models.

<strong>Train-Validation-Test Split:</strong> For each dataset, we randomly choose $1,000$ items for validation and $1,000$ items for testing, and remove all interactions associated with these items from the training set. The remaining interactions are used for training.

## Cold-to-Warm Start Transition

Our main experimental focus is on cold-start items with no prior interactions observed. However, it is important to analyze how our method behaves during the transition phase, i.e., when a few new interactions for the cold-start items are received. To do this, we proceed as follows: for each cold-start item in the test set, we randomly select $K$ interactions and add them to the training set, and finally train the models once more. The remaining interactions are still in the test set. By varying $K$, we compute NDCG@$100$ for our model, treating the items with additional observations as warm items. In this scenario, pure collaborative filtering methods are also applicable. Accordingly, we add the shallow autoencoder EASE (Steck, 2019), as well as traditional matrix factorization (Hu et al., 2008) for comparison purposes.

The results are presented in Figure 1. The vertical red line in the graphs outlines the median number of interactions an item had in the original training set. We observe that our model maintains stable recommendation metrics when changing $K$, while pure collaborative methods significantly improve their performance as the number of observed interactions increases. For instance, EASE outperforms our model across all datasets after observing enough interactions.  Interestingly, for the Amazon and Goodbooks datasets, our model outperforms matrix factorization throughout the warming scenario.

## Implementation Details and Complexity Analysis

Our ALS procedure optimizes $\widehat{\mathbf{e}}_i$ by alternating over $i$ in

<figure style=\"text-align:center;\">


$$
\widehat{\mathbf{e}}_i =\Big(n_i \,\big[S^{-i}\big]^\top \big[S^{-i}\big] + \lambda I_d \Big)^{-1} \big[S^{-i}\big]^\top \big(\mathcal{X}^{-i}\big)^\top \mathbb{1}_{n_i}
$$


</figure>

while fixing $\mathcal{E}^{-i}$. Although, in principle, co-occurring items should not be updated simultaneously (to avoid information leakage through shared context), we observe empirically that precomputing some quantities from the initial values and updating small batches of items in parallel can significantly speedup convergence without degrading performance. Our model was trained solely on CPUs using an implementation in NumPy, while deep-learning baseline models were trained on a DGX A100 GPU card.

There are several ways to verify the convergence of our algorithm. For instance, the algorithm can rely on any of the following stopping criteria: (1) repeating the updates for a fixed number of iterations; (2) stopping when the training or validation loss changes by less than a prescribed threshold; or (3) storing the previous value of $\mathcal{E}$ and stopping when $\mathcal{E}$ does not change significantly. Following (Mazumder et al., 2010), for the last option we use a threshold $0<\tau\in\mathbb{R}$ and stop when
$$
\frac{\|\mathcal{E}-\mathcal{E}_{\mathrm{old}}\|_{\mathrm{Fro}}}{\|\mathcal{E}\|_{\mathrm{Fro}}} < \tau.
$$

<strong>Computational Complexity</strong>: Two main operations are needed when computing $\mathbf{e}_i$: (i) computing the dense Gram matrix multiplication $\big[S^{-i}\big]^\top \big[S^{-i}\big]$ and (ii) inverting the augmented Gram matrix $\big(n_i\,\big[S^{-i}\big]^\top \big[S^{-i}\big]+\lambda I_d\big)$. The first operation has complexity $O(d^2 n)$ and the second has complexity $O(d^3)$. Since <em>generally</em> $d \ll n$, in practical applications we can assume that the full complexity of the algorithm is $O(N d^2 n^2)$ if this operation is performed $N$ times for each of the $n$ warm-start items.

Note, however, in terms of memory, we only need to store the essentials for computation, with space complexity $O(M + nd)$, where $M$ is the number of observed entries in $X$, and $nd$ accounts for storing $\mathcal{E}$ and $S$. We can significantly reduce the time complexity by increasing memory usage via precomputation. Note that, when computing closed-form update above, the only term that changes across updates is $\mathcal{E}^{-i}$. By regrouping terms and precomputing the quantities that do not depend on the current iterate, we reduce the per-update cost and obtain an overall time complexity of $O(Nn)$, at the expense of increasing the memory complexity to $O(n d^2)$.

## LLM-Guided Topic Dictionary Construction and Cluster Labeling

We use an LLM to construct a topic dictionary for interpreting user representations and cluster centroids with human-readable semantic descriptors. Specifically, we used OpenAI’s GPT-5.2 Pro (`gpt-5.2-pro`) to generate the topic sets and vocabularies described above, using as input the book synopses (one per line) and a fixed-format prompting template.(OpenAI, 2025) The prompt below is provided together with the book synopses and instructs the LLM to output a set of topics, each paired with a synopsis-grounded vocabulary. We apply the same prompting procedure to MovieLens by replacing the book synopses with movie plot summaries and using an otherwise identical prompt template.

<strong>Prompt (verbatim).</strong>

```
You are helping me build a topic dictionary for semantic interpreting
representations in a book recommendation system.

Input: I will provide a text file where each line contains ONE book in
the format:

  TITLE: SYNOPSIS

(English text; some lines may contain non-English snippets.)

Task: Create a hierarchical topic dictionary.

Each topic must:

1) be specific (not generic like "drama", "interesting", "story",
"life"),
2) include AT LEAST 20 representative vocabulary items (single words or
short phrases, max 3 words),
3) focus on semantic content that would plausibly trigger user preference
(themes, settings, plot motifs, character archetypes),
4) avoid author names, book titles, character names, and franchise/series
references.

Topic design rules:

- Topics should be mutually distinguishable: minimize overlap in vocab
between topics.
- Do NOT use near-duplicate topics

Vocabulary rules:
- Provide at least 20 items per topic.
- Use lowercase for all vocabulary items.
- Avoid stopwords and ultra-common words ("the", "and", "good",
"thing", "people").
- Prefer terms likely to appear in synopses over abstract adjectives
("beautiful", "deep").

Output format:

  "topic_name_1": ["w1","w2",...],
  "topic_name_2": ["w1","w2",...],
  ...
  "topic_name_N": ["w1","w2",...],
```

For cluster labeling, we used the following prompt:

<strong>Prompt (verbatim).</strong>

```
You are given 15 clusters that represents user taste in a book
recommendation system. For each cluster, I provide:

- cluster id (0..14)
- a list of five topics entries (already ranked; first is most
important)

Your task:

Create ONE concise cluster label per cluster.

Rules:

1) The label must reflect the COMBINATION of themes, not just the top
one.
2) Do NOT use franchise/series names, character names, author names, or
specific IPs.
3) Prefer preference-triggering semantics: settings, motifs, subgenres,
archetypes.
4) Labels must be short and punchy (2--7 words).
5) Avoid generic labels like "Mixed", "Various", "Drama", "Action",
"Comedy" alone.
6) When a cluster mixes two worlds (e.g., "romance + music"), explicitly
fuse them.

Output format:

  "0": "<label>",
  "1": "<label>",
  ...
  "14": "<label>",
```

Bellow we enumerate the Topic-dictionary based on GoodBooks:

1. <strong>fantasy\_magic\_general:</strong> fantasy, magic, magical, wizard, wizards, witch, witches, sorcery, sorcerer, spell, spells, wand, wands, potion, potions, enchantment, enchanted, curse, cursed, prophecy, prophecies, myth, myths, legend, legends, dragon, dragons, elf, elves, dwarf, dwarves, orc, orcs, goblin, goblins, troll, trolls, giant, giants, faerie, faeries, fairy, fairies, kingdom, throne, castle, tower, realm, quest, journey, adventure, hero, heroine, villain, dark, darkness, ancient, artifact, amulet, ring, sword, blade, dagger, shield, battle, warrior, knight, mage, charm, rune, runic, portal, worlds, otherworld, destiny, fate, ritual, summon, summoning, spellcasting, incantation
2. <strong>magic\_school\_wizardry:</strong> hogwarts, school, boarding, dorm, castle, great, hall, professor, headmaster, student, students, classroom, lessons, homework, library, common, room, house, gryffindor, slytherin, ravenclaw, hufflepuff, quidditch, broom, brooms, broomstick, wand, wands, spell, spells, potion, potions, cauldron, cloak, owl, owls, letter, letters, platform, train, express, sorting, hat, muggle, muggles, wizarding, witchcraft, wizardry, defense, dark, arts, horcrux, azkaban, dementor, dementors, patronus, phoenix, goblet, triwizard, tournament, chamber, secrets, stone, sorcerer, sorcerer's, deathly, hallows, voldemort, dumbledore, hermione, ron, harry
3. <strong>epic\_quest\_middleearth:</strong> middle-earth, middleearth, shire, hobbit, hobbits, bilbo, frodo, gandalf, ring, rings, sauron, mordor, cracks, doom, mount, journey, quest, fellowship, tower, towers, return, king, kings, dark, lord, shadow, orc, orcs, uruk, elf, elves, dwarf, dwarves, ranger, wizard, sword, bow, arrow, battle, siege, fortress, mountain, mines, cave, caves, dragon, smaug, treasure, kingdom, throne, crown, artifact, power, bind, rule
4. <strong>mythology\_gods\_heroes:</strong> mythology, myth, greek, roman, gods, god, goddess, olympian, olympians, zeus, poseidon, hades, athena, ares, apollo, artemis, hera, demeter, persephone, titan, titans, kronos, demigod, demigods, half-blood, camp, oracle, prophecy, quest, labyrinth, underworld, monster, monsters, hydra, minotaur, cyclops, satyr, nymph, hero, heroes, odyssey, odysseus, ithaca, troy, trojan, muse, epic, voyage, sea, fleece, trident, lightning, thief
5. <strong>dragons\_riders:</strong> dragon, dragons, hatchling, hatch, egg, rider, riders, dragonrider, riding, saddle, fire, flame, scales, wing, wings, claw, claws, breath, flight, sky, sword, ancient, empire, king, tyrant, evil, dark, enemy, prophecy, destiny, magic, spell, elders, legend, training, mentor, quest, battle, war
6. <strong>supernatural\_creatures:</strong> vampire, vampires, werewolf, werewolves, coven, blood, thirst, bite, fang, fanged, undead, immortal, immortality, sunlight, moon, full, curse, cursed, shapeshift, shapeshifter, pack, hunter, hunters, demon, demons, angel, angels, spirit, spirits, ghost, ghosts, haunted, possession, possessed, witch, warlock, fae, shadowhunter, shadowhunters, underworld, darkness, monster, monsters, supernatural, occult
7. <strong>dystopia\_totalitarian:</strong> dystopia, dystopian, utopia, utopian, totalitarian, authoritarian, regime, state, party, propaganda, surveillance, spy, spies, telescreen, thoughtcrime, ministry, truth, police, secret, control, oppression, oppressed, rebellion, rebel, rebels, resistance, uprising, revolution, district, capitol, faction, virtue, caste, hierarchy, conditioning, conformity, censorship, ban, banned, burn, burning, books, book-burning, handmaid, gilead, commander, womb, fertility, ovaries, matching, society, rules, punishment, fear, indoctrination
8. <strong>post\_apocalyptic\_survival:</strong> apocalypse, apocalyptic, postapocalyptic, post-apocalyptic, end, world, plague, pandemic, virus, outbreak, infection, collapse, ruins, ash, burned, wasteland, desolate, barren, survive, survival, scavenge, scavenging, supplies, ration, food, water, shelter, bunker, refuge, camp, gun, pistol, raid, raiders, bandits, lawless, horde, death, starvation, cold, winter, storm, evacuation, disaster, catastrophe, catastrophic, rescue, signal, alone, isolation, dark, road, journey
9. <strong>science\_fiction\_tech:</strong> science, scientific, technology, technological, engineer, engineering, mechanical, machine, machines, robot, robots, satellite, space, spaceship, astronaut, mission, mars, planet, galaxy, interstellar, future, futuristic, laboratory, lab, experiment, experiments, research, facility, genetic, genetics, dna, clone, cloning, mutation, modified, bioengineering, reproductive, ecology, calorie, agriculture, calories, data, network, computer, code, cipher, cryptography, algorithm, simulation, virtual, time, travel, tesseract
10. <strong>time\_travel\_timeline:</strong> time, travel, time-travel, timeline, past, future, present, paradox, chronological, chronology, displacement, chrono, tesseract, dimension, fifth, space-time, loop, reset, memory, flashback, history, era, century, years, decades, calendar, date, sequence, rewind, forward
11. <strong>mystery\_detective\_investigation:</strong> mystery, mysterious, detective, investigation, investigate, investigating, clue, clues, evidence, suspect, suspects, alibi, case, cold, missing, disappearance, kidnapped, abduction, murder, killed, killer, crime, criminal, forensic, police, inspector, journalist, reporter, search, hunt, track, trail, riddle, puzzle, secret, secrets, hidden, reveal, truth, lie, lies, deceit, conspiracy, plot, twist, thriller, suspense
12. <strong>conspiracy\_codes\_symbols:</strong> conspiracy, secret, society, societies, illuminati, priory, sion, vatican, church, catholic, louvre, curator, murdered, symbol, symbols, symbologist, cryptic, cipher, code, codes, decode, decipher, riddle, riddles, puzzle, puzzles, manuscript, ancient, history, artifact, vault, crypt, crypts, catacomb, catacombs, cathedral, cathedrals, mystery, clues, trail, hidden, painting, da, vinci, leonardo
13. <strong>crime\_legal\_courtroom:</strong> law, legal, lawyer, attorney, defense, prosecutor, judge, jury, trial, court, courtroom, case, evidence, testimony, witness, sentence, conviction, prison, inmate, penitentiary, death, row, execution, electric, chair, appeal, justice, injustice, racism, racial, violence, crime, murder, assault, rifle, sniper, mob, mafia, fbi, investigation
14. <strong>romance\_relationships:</strong> romance, romantic, love, lovers, kiss, kissing, passion, desire, lust, affair, flirtation, courtship, dating, boyfriend, girlfriend, fiancé, fiance, fiancée, wedding, marriage, husband, wife, relationship, breakup, heartbreak, jealousy, betrayal, temptation, seductive, intimacy, soulmate, devotion, tender, sweet, forever, letters, notebook
15. <strong>erotic\_billionaire\_romance:</strong> erotic, sex, sexual, sensual, kink, bondage, dominant, submissive, submission, control, obsession, desire, pleasure, ecstasy, intimate, intimacy, contract, rules, safe, word, billionaire, wealth, rich, luxury, penthouse, limousine, helicopter, multinational, business, entrepreneur, tortured, demons, possess, possession
16. <strong>family\_friendship\_belonging:</strong> family, parents, mother, father, daughter, son, sister, brother, home, house, marriage, divorce, widow, widowed, orphan, orphaned, adopt, adoption, foster, child, children, friend, friends, friendship, best, sisterhood, bond, bonding, loyal, loyalty, community, town, neighbor, neighbors, support, group, care, protect, protection, love, forgiveness, reconciliation
17. <strong>coming\_of\_age\_school:</strong> coming-of-age, coming\_of\_age, teen, teens, teenage, teenager, teenagers, adolescent, adolescence, youth, young, adult, young-adult, ya, highschool, high-school, middle-school, boarding-school, college, campus, dorm, dormitory, class, classes, teacher, teachers, principal, headmaster, bully, bullies, friend, friends, friendship, clique, outsider, misfit, loner, crush, first-love, romance, dating, prom, party, graduation, identity, self-discovery, self, esteem, insecurity, confidence, peer, pressure, coming, out, rebellion, rules, detention, sports, team, club
18. <strong>classics\_society\_manners:</strong> regency, victorian, manners, society, gentleman, lady, estate, manor, ball, dance, courtship, marriage, proposal, pride, prejudice, wit, satire, austere, fortune, inheritance, money, status, class, upper, middle, poor, rich, scandal, gossip, reputation, virtue, honor, duel, tragedy
19. <strong>war\_holocaust\_occupation:</strong> war, world, battle, soldier, army, air, forces, bomber, prisoner, camp, concentration, auschwitz, buchenwald, nazi, germany, hitler, occupation, occupied, resistance, underground, escape, hiding, attic, annexe, gestapo, jewish, holocaust, refugee, bomb, bombing, firebombing, dresden, front, line, uniform, veteran, patriot, flag
20. <strong>historical\_revolution\_empire:</strong> revolution, french, paris, london, guillotine, mob, uprising, monarchy, king, queen, empire, imperial, feudal, castle, cathedral, monk, monastery, plague, poverty, injustice, prison, convict, escape, revenge, treasure, isle, count, marquis, baron, duke, court, soldiers, march, battlefield
21. <strong>social\_justice\_race\_poverty:</strong> racism, racial, segregation, civil, rights, injustice, justice, law, trial, court, jury, southern, alabama, mississippi, poverty, poor, working, class, wages, minimum, wage, wal-mart, waitress, maid, hotel, service, industry, labor, sharecropper, okies, dust, bowl, migrant, immigrant, refugee, slavery, runaway, slave, lynch, klan, violence, activism, protest
22. <strong>memoir\_biography\_true\_story:</strong> memoir, biography, autobiography, diary, journal, true, story, real, life, childhood, family, growing, up, author, writer, recount, account, testimony, remember, memories, reminiscence, survival, resilience, redemption, trauma, poverty, homeless, addiction, alcohol, abuse, escape, journey, travel, adventure, mountain, wilderness, hitchhiked, alaska, olympics, entrepreneur, inventor, icon, history
23. <strong>philosophy\_ethics\_existential:</strong> philosophy, philosophical, ethics, ethical, moral, morality, virtue, justice, duty, conscience, guilt, sin, redemption, forgiveness, truth, meaning, absurd, existential, existence, life, death, mortality, suffering, freedom, choice, responsibility, reason, rational, logic, utility, utilitarianism, apology, socrates, plato, argument, debate, speech, trial, human, nature, soul
24. <strong>religion\_spirituality\_faith:</strong> god, faith, prayer, pray, church, bible, scripture, christian, catholic, baptist, minister, preacher, sermon, angel, angels, heaven, hell, salvation, redeem, redemption, sin, forgiveness, miracle, spiritual, spirituality, soul, souls, divine, holy, saint, devotion, worship, prophecy, apocalypse, rapture, afterlife
25. <strong>horror\_thriller\_fear:</strong> horror, terror, terrifying, nightmare, spooky, creepy, sinister, evil, haunted, ghost, ghosts, demon, demons, possession, cursed, curse, blood, murder, killer, death, dead, monster, monsters, beast, beasts, dark, shadow, violence, panic, fear, scream, madhouse, asylum, hotel, overlook, winter, storm, isolation, cabin, woods, attic, shivers
26. <strong>mental\_health\_hospital:</strong> mental, health, depression, anxiety, madness, insane, insanity, therapy, therapist, psychiatric, hospital, ward, nurse, doctor, diagnosis, patient, patients, medication, shock, electroshock, trauma, ptsd, suicide, self-harm, grief, mourning, loss, addiction, alcohol, drink, drinking
27. <strong>illness\_cancer\_medical:</strong> cancer, tumor, leukemia, illness, disease, diagnosis, terminal, treatment, chemo, chemotherapy, radiation, surgery, transfusion, shots, hospital, doctor, nurse, clinic, medicine, medical, miracle, support, group, patient, pain, ovarian, blood, bone, marrow, donor, genetic, syndrome, down
28. <strong>travel\_adventure\_journey:</strong> journey, travel, trip, voyage, road, walk, hitchhike, train, bus, plane, ship, boat, island, desert, mountain, river, sea, ocean, wilderness, forest, north, south, paris, rome, london, scotland, afghanistan, kabul, bangkok, alaska, america, york, city, escape, adventure, expedition, quest, odyssey, pilgrimage
29. <strong>books\_reading\_writing\_storytelling:</strong> book, books, reading, read, reader, story, stories, tale, tales, novel, chapter, chapters, author, writer, writing, literature, classic, poetry, poem, poems, diary, journal, letters, notebook, library, librarian, school, teacher, study, studying, text, translation, edition, illustrations, illustrated, words, language, prose, narrator, narration, essay, essays
30. <strong>children\_picture\_books\_rhymes:</strong> children, child, kids, kid, baby, mother, father, family, bedtime, sleep, nursery, picture, book, illustrations, rhymes, rhyme, funny, silly, wacky, cat, hat, mouse, goat, tree, eggs, ham, bear, lion, spider, web, pig, barn, zoo, puppy, duck, crocodile, whale, unicorn, shadow, garbage, play, playful, school, principal, teacher, story, tale
31. <strong>animals\_wilderness\_nature:</strong> animal, animals, wild, wilderness, forest, woods, mountain, river, sea, ocean, island, desert, arctic, snow, winter, storm, tiger, bear, lion, wolf, dog, puppy, horse, elephant, spider, rat, rats, bee, bees, honey, pig, farm, barn, web, zoo, circus, train, hunt, hunting, survival, pack, herd, nature, outdoors
32. <strong>music\_art\_performance:</strong> music, musical, song, songs, guitar, strings, band, concert, sing, singing, record, records, album, piano, composer, artist, art, painting, portrait, museum, gallery, theatre, theater, play, drama, performance, stage, actor, actress, circus, acrobats, magicians, magic, show, showbiz, hollywood
33. <strong>sports\_competition\_games:</strong> competition, tournament, match, games, game, contest, training, initiation, challenge, survive, survival, fight, fighting, battle, arena, victory, win, winner, lose, losing, tribute, reality, tv, faction, selection, crown, prince, princess, sports, team, rugby, coach, practice, race, olympics, run, running
34. <strong>parenting\_babycare:</strong> baby, babies, newborn, infant, toddler, child, children, parent, parents, mom, moms, mother, mothers, dad, dads, father, fathers, crib, bassinet, nursery, stroller, carseat, diaper, diapers, cloth, diapering, wipes, bottle, bottles, formula, breastfeeding, breastmilk, latch, pumping, milk, feeding, feedings, sleep, sleeping, nap, naps, sleep-training, swaddle, pacifier, teething, colic, burp, burping, bath, bathtime, pajamas, pajama, potty, potty-training, weaning, solids, puree, homemade, food, vitamin, supplements, doctor, pediatrician, vaccines, immunization, growth, milestones, month-by-month, screen-time, tablet, tv, apps, safety, childproof, attachment, parenting
35. <strong>work\_labor\_service\_industry:</strong> work, job, jobs, worker, workers, labor, shift, wages, pay, paycheck, salary, tips, tipping, minimum, wage, poverty, poor, working, class, waitress, waiter, server, maid, housekeeper, housekeeping, cleaning, janitor, nursing, home, aide, retail, wal-mart, walmart, big, box, hotel, motel, bellhop, bellman, valet, parking, front, desk, concierge, guest, guests, customer, customers, service, industry, manager, boss, coworker, abuse, overworked, overtime, schedule, rent, lodging, cheap, rooms, survival, stratagem, union, strike
36. <strong>political\_history\_presidency:</strong> president, presidency, white, house, oval, office, campaign, election, reagan, ronald, governor, california, hollywood, cold, war, soviet, iron, curtain, assassin, assassination, bullet, hinckley, secret, service, press, conference, speech, statecraft, policy, congress, senate, representatives, democrat, republican, conservative, liberal, communism, capitalism, diplomacy, summit, gorbachev, union, history, biography
37. <strong>cult\_high\_control\_religion:</strong> cult, cults, sect, sects, high-control, indoctrination, brainwashing, coercion, control, obedience, punishment, discipline, shunning, isolation, escape, defect, defection, insider, whistleblower, secrecy, ritual, rituals, leader, leadership, celebrity, recruitment, auditing, e-meter, scientology, sea, org, david, miscavige, church, organization, faith, religion, belief, doctrine, dogma, apostate, freedom, trauma, family
38. <strong>comics\_superheroes\_graphic\_novels:</strong> comic, comics, graphic, novel, manga, volume, issue, issues, panel, panels, superhero, superheroes, hero, villain, vigilante, mask, cape, gotham, batman, joker, dark, knight, returns, dc, marvel, origin, story, reboot, crossover, arc, series, collecting, collector, edition, illustrated, art, ink, pencil, storyboard
39. <strong>art\_history\_visual\_arts:</strong> art, history, artist, artists, painting, paintings, sculpture, sculptures, drawing, drawings, sketch, sketches, portrait, portraits, landscape, still, life, museum, gallery, exhibition, renaissance, baroque, impressionism, modern, modernism, abstract, cubism, surrealism, icon, icons, fresco, canvas, oil, watercolor, acrylic, architecture, cathedral, design, aesthetics, criticism, critique, artwork, masterpiece, masters
40. <strong>christian\_rapture\_apocalypse:</strong> rapture, tribulation, antichrist, apocalypse, apocalyptic, revelation, revelations, prophecy, prophecies, endtimes, end-times, judgment, judgement, salvation, saved, unsaved, believers, unbelievers, church, pastor, preacher, bible, scripture, heaven, hell, angels, demon, demons, disappear, disappearance, vanish, vanished, left, behind, plague, mark, beast, armageddon
41. <strong>satire\_parody\_comedy:</strong> satire, satirical, parody, spoof, lampoon, comedy, comic, humor, humour, hilarious, witty, wit, absurd, ridiculous, farce, irony, ironic, mock, mocking, sendup, tongue-in-cheek, pastiche, sarcasm, sarcastic, bizarre, weird, wacky, quirky, dark, laugh, laughing, jokes
42. <strong>theater\_stage\_play:</strong> play, stage, theatre, theater, drama, dramatic, scene, scenes, act, acts, dialogue, monologue, character, characters, audience, performance, script, screenplay, director, actors, actresses, broadway, pulitzer, critics, circle, award, production, rehearsal, opening, night
43. <strong>immortality\_eternal\_life:</strong> immortal, immortality, eternal, life, forever, never, die, deathless, ageless, age, youth, spring, magic, water, drink, drinking, curse, blessing, blessed, doomed, secret, live, living, regret, consequence, mortality, mortal, human, choice, time
44. <strong>spies\_espionage\_thriller:</strong> spy, spies, espionage, agent, agents, secret, mission, assignment, operation, operative, smersh, 007, bond, james, harlem, caribbean, everglades, jazz, joints, criminal, heavy, hitter, threat, assassin, assassination, weapon, weapons, gun, guns, chase, pursuit, cover, undercover, surveillance, interrogation
45. <strong>haunted\_objects\_vehicles:</strong> haunted, cursed, possessed, object, objects, artifact, artifacts, doll, mirror, house, car, cars, automobile, plymouth, fury, engine, garage, drive, driving, headlights, radio, rock, roll, detroit, accident, crash, blood, revenge, jealousy, obsession, evil
46. <strong>business\_entrepreneurship:</strong> business, company, companies, startup, start-up, entrepreneur, entrepreneurship, founder, founding, ceo, executive, manager, management, leadership, strategy, strategic, product, products, market, marketing, sales, branding, brand, customer, customers, growth, scale, scaling, revenue, profit, loss, losses, budget, budgets, funding, investor, investors, venture, capital, vc, pitch, deal, negotiation, contract, contracts, team, teams, culture, innovation, disruption, competition, competitive, industry, industries, supply, demand, operations, logistics, metrics, kpi, analytics
47. <strong>finance\_investing\_money:</strong> money, wealth, rich, poor, income, salary, wages, budget, saving, savings, debt, loan, loans, credit, mortgage, interest, compound, investment, investments, investing, investor, investors, portfolio, asset, assets, stocks, stock, bond, bonds, equity, index, fund, funds, etf, dividend, dividends, market, markets, bull, bear, risk, return, returns, valuation, value, cash, cashflow, inflation, recession, economy, economics, finance, financial, retirement, pension, tax, taxes, insurance
48. <strong>health\_fitness\_nutrition:</strong> health, healthy, fitness, fit, workout, workouts, exercise, training, gym, running, run, yoga, pilates, strength, cardio, endurance, muscle, weight, weights, fat, calories, calorie, diet, nutrition, nutritional, protein, carbs, carbohydrate, fiber, vitamin, vitamins, supplement, supplements, sleep, stress, wellness, mind, body, metabolism, hydration, water, meal, meals, recipe, recipes, cook, cooking, kitchen
49. <strong>psychology\_self\_help\_mindfulness:</strong> psychology, psychological, mind, brain, behavior, behaviour, habit, habits, motivation, discipline, willpower, mindset, self, help, self-help, confidence, anxiety, depression, stress, trauma, therapy, therapist, counseling, counselling, healing, growth, personal, development, mindfulness, meditation, breath, breathing, gratitude, compassion, empathy, relationships, communication, boundaries, resilience, purpose, meaning, happiness, joy, focus, productivity, procrastination
50. <strong>cooking\_food\_recipes:</strong> cook, cooking, kitchen, recipe, recipes, ingredients, ingredient, bake, baking, roast, grill, saute, sauté, boil, simmer, fry, fried, oven, stove, pan, pot, knife, spice, spices, herb, herbs, salt, pepper, sugar, flour, butter, oil, garlic, onion, tomato, cheese, bread, cake, dessert, chocolate, soup, stew, salad, sauce, pasta, rice, beans, meat, chicken, beef, pork, fish, seafood, vegetarian, vegan, gluten, nutrition, meal, meals, menu
51. <strong>technology\_programming\_computers:</strong> technology, tech, computer, computers, software, hardware, programming, programmer, code, coding, developer, development, debug, bug, algorithm, algorithms, data, database, databases, network, networks, internet, web, website, app, apps, mobile, server, cloud, security, encryption, cryptography, ai, ml, machine, learning, neural, model, models, training, dataset, api, open-source, opensource, linux, python, java, javascript, terminal, command, git, github
52. <strong>popular\_science\_nature:</strong> science, scientific, physics, chemistry, biology, genetics, dna, evolution, neuroscience, astronomy, cosmos, universe, planet, planets, space, ecology, environment, climate, energy, atoms, molecule, molecules, experiment, experiments, research, scientist, scientists, theory, theories, evidence, data, discovery, discoveries, laboratory, lab, nature, wildlife, species


Bellow we enumerate the Topic-dictionary based on MovieLens:

1. <strong>crime\_underworld:</strong> gangster, mob, mafia, cartel, hitman, assassin, crime boss, racketeering, drug deal, smuggling, black market, undercover, informant, witness, shootout, betrayal, corrupt, extortion, loan shark, criminal empire, street gang, organized crime, revenge, vigilante, fugitive, manhunt, prison, parole, robbery, kidnapping, ransom
2. <strong>heists\_cons:</strong> heist, bank robbery, vault, safecracker, con artist, scam, grift, swindle, identity theft, forgery, counterfeit, inside job, getaway, crew, stakeout, double-cross, score, loot, ransack, hostage, ransom, hijack, blackmail, stolen art, diamond, casino, pickpocket, hustler, underworld contact
3. <strong>mystery\_investigation:</strong> detective, investigation, case, clue, suspect, interrogation, forensics, evidence, crime scene, murder mystery, whodunit, missing person, disappearance, cold case, alibi, witness statement, private investigator, police chief, serial killer, profiling, autopsy, trail, cover-up, conspiracy, secret, twist ending
4. <strong>spy\_espionage:</strong> spy, espionage, secret agent, double agent, CIA, MI6, KGB, intelligence, surveillance, wiretap, classified, black ops, covert, infiltrate, counterintelligence, asset, handler, mole, defection, safehouse, passport, interrogation, torture, diplomatic, nuclear, bioweapon, cyberterror, mission, extraction
5. <strong>war\_military:</strong> war, battle, soldier, army, marines, navy, air force, mission, frontline, trench, sniper, bomber, invasion, resistance, occupation, prisoner of war, general, colonel, unit, comrades, boot camp, veteran, homecoming, propaganda, spy plane, wartime romance, medic, sacrifice, ceasefire, atrocity
6. <strong>survival\_disaster:</strong> survival, stranded, shipwreck, plane crash, wilderness, storm, hurricane, earthquake, tsunami, volcano, fire, flood, rescue, evacuation, catastrophe, doomsday, asteroid, pandemic, outbreak, quarantine, scarcity, rationing, shelter, search party, last survivors, endurance, hypothermia, dehydration, sos
7. <strong>martial arts\_combat:</strong> martial arts, kung fu, karate, dojo, master, disciple, tournament, sparring, fight, brawl, duel, sword, blade, samurai, ninja, warlord, training montage, honor, revenge, clan, warrior, champion, gladiator, combat, assassin, standoff, boss fight, arena, blood feud
8. <strong>quest\_adventure:</strong> quest, journey, expedition, treasure, map, artifact, ruins, temple, ancient, curse, legend, myth, escape, chase, hideout, lost city, jungle, desert, island, mountain, cave, ship, voyage, explorer, pirate, captain, sailing, tribe, mystery box, peril
9. <strong>space\_aliens:</strong> space, spaceship, astronaut, planet, galaxy, starship, orbit, mission control, alien, extraterrestrial, invasion, first contact, abduction, UFO, colony, terraform, wormhole, hyperspace, space station, laser, ray, space marine, cosmic, signal, telemetry, asteroid belt, moon base, Martian, exoplanet, mothership
10. <strong>time travel\_paradox:</strong> time travel, time loop, paradox, timeline, alternate history, future, past, temporal agent, chrononaut, rewind, foresee, predestination, butterfly effect, grandfather paradox, memory reset, déjà vu, prophecy, fate, destiny, reality shift, multiverse, dimension, portal, anomaly, time machine, retrocausality, looping day, foretold, identity twist
11. <strong>robots\_AI:</strong> robot, android, cyborg, AI, artificial intelligence, machine, sentient, algorithm, hack, hacker, cyber, virtual reality, simulation, upload, consciousness, dystopian tech, surveillance state, drone, automaton, malfunction, singularity, clone, genetic, bioengineering, lab, experiment, nanotech, firmware, mainframe, digital
12. <strong>dystopia\_apocalypse:</strong> dystopia, post-apocalyptic, collapse, ruins, wasteland, mutants, scavenger, bunker, radiation, nuclear, totalitarian, regime, rebellion, uprising, oppression, underground, blackout, curfew, ration, resistance, propaganda, chosen one, civil war, martial law, contagion, end times, fallout, last city, survivors
13. <strong>magic\_myth:</strong> magic, wizard, witch, spell, curse, prophecy, chosen one, enchanted, sorcerer, warlock, alchemy, potion, rune, artifact, mythology, gods, demigod, oracle, legendary, dragon, giant, troll, fairy, elf, dwarf, quest, kingdom, throne, knight, castle
14. <strong>vampires\_werewolves:</strong> vampire, blood, immortal, undead, coven, fangs, werewolf, lycan, pack, moon, transformation, hunter, stake, curse, darkness, night, forbidden love, ancient evil, seduction, supernatural, witch, cursed lineage, bite, ritual, haunted, crypt, catacombs, curse breaking, bloodline, immortality
15. <strong>ghosts\_hauntings:</strong> ghost, haunted, spirit, poltergeist, possession, exorcism, paranormal, medium, seance, cursed house, apparition, afterlife, demon, ritual, occult, nightmare, creepy, whisper, shadow, screams, hotel room, basement, graveyard, mystic, curse, hallucination, superstition, urban legend, haunting past, restless
16. <strong>monsters\_creature horror:</strong> monster, creature, beast, mutant, parasite, infection, experiment gone wrong, lab creature, predator, hunt, bloodthirsty, killer, cannibal, slasher, serial killer, zombie, undead horde, outbreak, body horror, were-beast, giant insect, sea monster, alien predator, survival horror, stalker, terror, panic, hide, trapped, final girl
17. <strong>romance\_heartbreak:</strong> love, romance, affair, cheating, breakup, heartbreak, wedding, engagement, proposal, forbidden love, love triangle, jealousy, reunion, second chance, long distance, letters, destiny, soulmate, passion, seduction, dating, flirtation, honeymoon, divorce, widow, mourning, first love, unrequited, commitment, intimacy
18. <strong>family\_parenting:</strong> family, parents, mother, father, siblings, daughter, son, guardian, custody, divorce, stepfamily, orphan, adoption, home, coming home, inheritance, legacy, generations, family secret, caregiving, illness, funeral, reconciliation, dysfunctional, holiday gathering, raising a child, teenage, responsibility, family business, protect
19. <strong>friendship\_coming-of-age:</strong> friendship, best friends, growing up, coming-of-age, teen, high school, summer, first kiss, bullies, outsider, misfit, self-discovery, identity, graduation, college, young adult, road trip, band, party, prom, sports team, new kid, mentor, peer pressure, rebellion, dreams, small town, nostalgia, childhood, rite of passage, found family
20. <strong>comedy\_satire:</strong> comedy, satire, spoof, parody, awkward, misunderstanding, farce, fish out of water, odd couple, schemes, prank, ridiculous, absurd, slapstick, rom-com, wedding chaos, road comedy, buddy comedy, workplace comedy, stand-up, celebrity, showbiz, small-time, hustle, quirky, deadpan, mockumentary, midlife crisis, family comedy, culture clash
21. <strong>courtroom\_justice:</strong> court, trial, lawyer, attorney, judge, jury, testimony, verdict, appeal, lawsuit, evidence, cross-examination, legal thriller, wrongful conviction, innocent, prison, death penalty, corruption, cover-up, whistleblower, civil rights, scandal, investigation, prosecution, defense, plea deal, witness protection, case files, moral dilemma, justice
22. <strong>politics\_power:</strong> politics, mayor, president, campaign, election, senate, parliament, dictator, coup, revolution, state, government, bureaucracy, corruption, spin, press, journalist, whistleblower, conspiracy, diplomacy, treaty, war room, policy, protest, activist, ideology, power struggle, assassination, propaganda, regime
23. <strong>race, class\_inequality:</strong> racism, segregation, prejudice, class, poverty, wealth, privilege, immigrant, refugee, oppression, labor, strike, hunger, urban, ghetto, community, justice, civil rights, protest, discrimination, identity, belonging, marginalized, survival, education, jobless, struggle, neighborhood, family duty, hope
24. <strong>work\_ambition:</strong> career, ambition, success, failure, promotion, boss, office, workplace, intern, deadline, competition, fame, celebrity, artist, writer, musician, actor, producer, startup, deal, betrayal, rivalry, dream job, training, performance, pressure, burnout, mentor, second act, reinvention
25. <strong>music\_performance:</strong> music, band, singer, concert, tour, rock, jazz, classical, composer, piano, guitar, drummer, record, album, stage, spotlight, dance, ballet, opera, show, theater, audition, festival, talent, fame, backstage, groupies, breakup tour, encore, soundtrack
26. <strong>sports\_competition:</strong> boxing, wrestling, race, track, olympics, football, basketball, baseball, coach, training, championship, tournament, underdog, comeback, injury, rivalry, team, locker room, draft, agent, betting, gambling, marathon, fighter, knockout, ring, stadium, victory, defeat, discipline
27. <strong>historical epics:</strong> historical, period, empire, royalty, king, queen, palace, court, dynasty, reign, colonial, plantation, aristocrat, noble, serf, revolution, war of independence, victorian, medieval, renaissance, ancient, gladiator, conquest, uprising, battles, palace intrigue, succession, landed estate, family lineage, chronicle
28. <strong>western\_frontier:</strong> western, cowboy, outlaw, sheriff, frontier, ranch, saloon, gunfight, duel, posse, bounty, trail, wagon, settlers, gold rush, train robbery, desert town, horse, marshal, bandit, revenge, lawless, border, territory, indian, cattle drive, showdown, wanted poster, homestead, range
29. <strong>noir\_psychological:</strong> noir, psychological, paranoia, obsession, delusion, unreliable narrator, identity, dream, hallucination, mind game, therapy, trauma, memory, guilt, confession, double life, dark secret, moral ambiguity, femme fatale, betrayal, loneliness, alienation, despair, existential, compulsion, meltdown, introspection, fatalism, spiral, black-and-white
30. <strong>animation\_kids:</strong> animation, cartoon, toy, talking animal, princess, fairytale, storybook, magical friend, adventure for kids, school, playground, family-friendly, sing-along, sidekick, villain, lesson, friendship, imagination, holiday, cute, animal companions, coming home, child hero, dreamland, storybook world, rescue mission, playtime, kidnapped pet, happy ending, wholesome
31. <strong>animals\_nature:</strong> animals, dog, cat, horse, wildlife, jungle, forest, ocean, whale, shark, bear, lion, zoo, ranger, habitat, conservation, environment, nature documentary, ecosystem, pet, trainer, rescue animal, migration, mountain, island, stormy sea, expedition, safari, pack, instinct



## References

<ul>
<li><strong>Abdullah et al.</strong> (2021). Eliciting auxiliary information for cold start user recommendation: A survey.</li>
<li><strong>Alves et al.</strong> (2021). Burst-induced multi-armed bandit for learning recommendation.</li>
<li><strong>Anwaar et al.</strong> (2018). HRS-CE: A hybrid framework to integrate content embeddings in recommender systems for cold start items.</li>
<li><strong>Baeza-Yates et al.</strong> (2015). Predicting the next app that you are going to use.</li>
<li><strong>Bajaj</strong> (2022). UCSD Goodreads Dataset.</li>
<li><strong>Bank et al.</strong> (2023). Autoencoders.</li>
<li><strong>Chen et al.</strong> (2018). A collective variational autoencoder for top-n recommendation with side information.</li>
<li><strong>Chen et al.</strong> (2019). Deep autoencoders in pattern recognition: a survey.</li>
<li><strong>Chiang et al.</strong> (2018). Using side information to reliably learn low-rank matrices from missing and corrupted observations.</li>
<li><strong>Chollet et al.</strong> (2015). Keras.</li>
<li><strong>De et al.</strong> (2022). Modelling users with item metadata for explainable and interactive recommendation.</li>
<li><strong>Dhamani</strong> (2021). Goodreads 100K books.</li>
<li><strong>Du et al.</strong> (2020). How to learn item representation for cold-start multimedia recommendation?.</li>
<li><strong>Duricic et al.</strong> (2018). Trust-based collaborative filtering: Tackling the cold start problem using regular equivalence.</li>
<li><strong>Fernandez-Tobias et al.</strong> (2016). Alleviating the new user problem in collaborative filtering by exploiting personality information.</li>
<li><strong>Frederickson</strong> (2019). Fast Python Collaborative Filtering for Implicit Datasets..</li>
<li><strong>Gantner et al.</strong> (2010). Learning attribute-to-feature mappings for cold-start recommendations.</li>
<li><strong>Harper et al.</strong> (2015). The MovieLens Datasets: History and Context.</li>
<li><strong>Herce-Zelaya et al.</strong> (2020). New technique to alleviate the cold start problem in recommender systems using information from social media and random decision forests.</li>
<li><strong>Hu et al.</strong> (2008). Collaborative filtering for implicit feedback datasets.</li>
<li><strong>Huang et al.</strong> (2023). Aligning distillation for cold-start item recommendation.</li>
<li><strong>Jin et al.</strong> (2023). Automatic Fusion Network for Cold-start CVR Prediction with Explicit Multi-Level Representation.</li>
<li><strong>Lam et al.</strong> (2008). Addressing cold-start problem in recommendation systems.</li>
<li><strong>Ledent et al.</strong> (2021). Fine-grained generalization analysis of inductive matrix completion.</li>
<li><strong>Ledent et al.</strong> (2021). Orthogonal inductive matrix completion.</li>
<li><strong>Li et al.</strong> (2023). Exploring the upper limits of text-based collaborative filtering using large language models: Discoveries and insights.</li>
<li><strong>Liang et al.</strong> (2018). Variational autoencoders for collaborative filtering.</li>
<li><strong>Lika et al.</strong> (2014). Facing the cold start problem in recommender systems.</li>
<li><strong>Liu et al.</strong> (2022). Hybrid Variational Autoencoder for Collaborative Filtering.</li>
<li><strong>Mazumder et al.</strong> (2010). Spectral regularization algorithms for learning large incomplete matrices.</li>
<li><strong>Mhatre</strong> (2020). IMDB movies analysis.</li>
<li><strong>Michiels et al.</strong> (2022). RecPack: An(Other) Experimentation Toolkit for Top-N Recommendation Using Implicit Feedback Data.</li>
<li><strong>Moon et al.</strong> (2023). It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation.</li>
<li><strong>Mudigoudr</strong> (2024). Imdb Movie Dataset from year 1893 to 2020.</li>
<li><strong>Ni et al.</strong> (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects.</li>
<li><strong>Ning et al.</strong> (2011). Slim: Sparse linear methods for top-n recommender systems.</li>
<li><strong>Panda et al.</strong> (2022). Approaches and algorithms to mitigate cold start problems in recommender systems: a systematic literature review.</li>
<li><strong>Paszke et al.</strong> (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.</li>
<li><strong>Pedregosa et al.</strong> (2011). Scikit-learn: Machine Learning in {P}ython.</li>
<li><strong>Reese</strong> (2020). Goodreads books - 31 features.</li>
<li><strong>Reimers et al.</strong> (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.</li>
<li><strong>Sankar et al.</strong> (2021). Protocf: Prototypical collaborative filtering for few-shot recommendation.</li>
<li><strong>Sanner et al.</strong> (2023). Large language models are competitive near cold-start recommenders for language-and item-based preferences.</li>
<li><strong>Saveski et al.</strong> (2014). Item cold-start recommendations: learning local collective embeddings.</li>
<li><strong>Schein et al.</strong> (2001). Generative models for cold-start recommendations.</li>
<li><strong>Sedhain et al.</strong> (2015). Autorec: Autoencoders meet collaborative filtering.</li>
<li><strong>Sethi et al.</strong> (2021). Cold start in recommender systems—A survey from domain perspective.</li>
<li><strong>Spišák et al.</strong> (2023). Scalable approximate nonsymmetric autoencoder for collaborative filtering.</li>
<li><strong>Steck et al.</strong> (2021). Deep learning for recommender systems: A Netflix case study.</li>
<li><strong>Steck</strong> (2019). Embarrassingly shallow autoencoders for sparse data.</li>
<li><strong>Steck</strong> (2020). Autoencoders that don't overfit towards the identity.</li>
<li><strong>Takács et al.</strong> (2011). Applications of the conjugate gradient method for implicit feedback collaborative filtering.</li>
<li><strong>Truong et al.</strong> (2021). Bilateral variational autoencoder for collaborative filtering.</li>
<li><strong>Tu et al.</strong> (2019). From fingerprint to footprint: Cold-start location recommendation by learning user interest from app data.</li>
<li><strong>Vančura et al.</strong> (2021). Deep variational autoencoder with shallow parallel path for top-N recommendation (VASP).</li>
<li><strong>Vančura et al.</strong> (2022). Scalable linear shallow autoencoder for collaborative filtering.</li>
<li><strong>Vančura et al.</strong> (2024). beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems.</li>
<li><strong>Vančura et al.</strong> (2025). Evaluating Linear Shallow Autoencoders on Large Scale Datasets.</li>
<li><strong>Wang et al.</strong> (2024). Preference Aware Dual Contrastive Learning for Item Cold-Start Recommendation.</li>
<li><strong>Wu et al.</strong> (2020). A hybrid conditional variational autoencoder model for personalised top-n recommendation.</li>
<li><strong>Wu et al.</strong> (2023). A survey on large language models for recommendation.</li>
<li><strong>Wu et al.</strong> (2023). M2eu: Meta learning for cold-start recommendation via enhancing user preference estimation.</li>
<li><strong>Wu et al.</strong> (2024). A survey on large language models for recommendation.</li>
<li><strong>Wu et al.</strong> (2024). Could small language models serve as recommenders? towards data-centric cold-start recommendation.</li>
<li><strong>Xu et al.</strong> (2013). Speedup matrix completion with side information: Application to multi-label learning.</li>
<li><strong>Xu et al.</strong> (2020). A collaborative filtering framework based on variational autoencoders and generative adversarial networks.</li>
<li><strong>Yu et al.</strong> (2023). XSimGCL: Towards extremely simple graph contrastive learning for recommendation.</li>
<li><strong>Zajac</strong> (2017). Goodbooks-10k: a new dataset for book recommendations.</li>
<li><strong>Zhang et al.</strong> (2014). Addressing cold start in recommender systems: A semi-supervised co-training algorithm.</li>
<li><strong>Zhou et al.</strong> (2023). Contrastive collaborative filtering for cold-start item recommendation.</li>
<li><strong>Zhu et al.</strong> (2020). Recommendation for new users and new items via randomized training and mixture-of-experts transformation.</li>
<li><strong>Zhu et al.</strong> (2021). Fairness among new items in cold start recommender systems.</li>
<li><strong>Zhu et al.</strong> (2021). Learning to warm up cold item embeddings for cold-start recommendation with meta scaling and shifting networks.</li>
<li><strong>OpenAI</strong> (2025). Introducing GPT-5.2.</li>
</ul>

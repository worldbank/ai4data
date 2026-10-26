// Videos listed on /videos by kind (concepts, methods,
// walkthroughs) and embedded with
// <Video id="..." /> (src/components/Video). Each record names the YouTube
// video, its area in the hub, the documentation and code of the method,
// and the transcript of its narration. `voice` is 'synthetic' for a
// narration read by a text-to-speech voice and 'recorded' for a person's
// voice; the videos page shows a note for synthetic narration.

export const videos = [
  {
    id: 'pcn-explainer',
    title: 'Proof-Carrying Numbers',
    youtube: 'FakLU10grL4',
    duration: '3:25',
    published: 'October 2026',
    kind: 'methods',
    area: 'Verification',
    voice: 'synthetic',
    summary:
      'An AI chatbot that answers from statistics can write a figure that differs from its source, and a reader cannot tell this from the text of the answer. The video shows how Proof-Carrying Numbers links each figure in an answer to the record it came from, how a verifier checks the figure against that record, and what the check leaves to the judgement of the reader.',
    links: [
      {label: 'Documentation', to: '/docs/pcn/'},
      {label: 'Code', href: 'https://github.com/worldbank/pcn'},
      {label: 'Paper', href: 'https://arxiv.org/abs/2509.06902'},
    ],
    transcript: [
      "AI chatbots now answer questions about official statistics. A user asks how internet use has changed in Kenya and Rwanda. The chatbot answers in plain language, with figures. Take the figure 35.0. It may come from the data that the chatbot was given, rounded to one decimal. It may also come from the model's memory of its training data, which can hold an older estimate. Or the model may copy a record and change a digit as it writes. The answer reads the same in all three cases, so a reader cannot tell which figures to trust. Proof-Carrying Numbers, or PCN, lets the application check each figure against its source.",
      "PCN starts with the data that the application gives the model. Here the values come from the World Bank's World Development Indicators. The application gives each value a claim id, a key that names exactly one record. Here the key joins the indicator, the country and the year. The claim id is the link that lets a figure in the answer point back to its record. Any key that is unique in the data works, such as a short hash.",
      "Only the model knows which record each of its figures comes from, so the instructions ask the model to say so. Every figure taken from the data goes inside a claim tag, with the claim id of its record and the rule for comparing the two. The application sends the instructions, the data and the question to the model together.",
      "The model streams its answer. Each figure from the data arrives inside a claim tag. The reader sees the figures alone. The tags stay in the response for the next step.",
      "A verifier, written as ordinary code, reads the response as it streams. When a claim tag is complete, the verifier reads its claim id, its rule and its figure. It finds the record by its claim id. The record for Kenya in 2024 holds 34.98, to two decimals. The rule rounds the record to one decimal, which gives 35.0, the figure in the answer. The figure is verified. No language model takes part in the check, so the check gives the same result on every run. For Rwanda, the record rounds to 31.7, and the answer says 35.7. The verifier flags the figure. The figure on mobile money has no tag. With no record named, the verifier has nothing to check, and the figure gets no mark.",
      "In the chatbot, each checked figure carries a mark. Hovering over a mark shows the record and its value in the data, so the reader can trace the figure to its source. A flagged figure is caught before anyone uses it. A figure without a mark is known to need a check by hand.",
      "The verifier confirms that each tagged figure matches the record named in its tag. Whether that record answers the question remains the reader's judgement. PCN makes that judgement quick, because each mark shows the record behind its figure. Here the indicator is internet use and the country is Kenya, as the question asks, and the year is the latest in the data.",
      "The PCN library implements this check.",
    ],
  },
  {
    id: 'gistembed-explainer',
    title: 'GISTEmbed: Small AI for search',
    youtube: 'f-fsbT9q308',
    duration: '2:56',
    published: 'October 2026',
    kind: 'methods',
    area: 'Access',
    voice: 'synthetic',
    summary:
      'Search by meaning runs on an embedding model, and small models of this kind can run on an ordinary computer. The video shows how GISTEmbed trains a small embedding model with a large model as its guide: during training, the guide keeps texts that belong close to a question from being pushed away from it, so the small model does not learn the wrong lesson.',
    links: [
      {label: 'Code', href: 'https://github.com/worldbank/GISTEmbed'},
      {label: 'Models', href: 'https://huggingface.co/collections/ai4data/gistembed'},
      {label: 'Paper', href: 'https://arxiv.org/abs/2402.16829'},
    ],
    transcript: [
      "People search in their own words. Someone asks how fast the economy expanded last year. The report that answers says that real GDP growth accelerated to 6.3 percent. The two share no words, so a search for matching words misses it. A search by meaning finds it, because both are about how much the economy grew.",
      "Search by meaning runs on an embedding model. The model reads each text and gives it a place on a map of meaning. Texts about the same subject land close together, and texts about different subjects land far apart. The question gets a place too, and the search returns the texts nearest to it. Each place is stored as a list of numbers, called a vector.",
      "Large embedding models draw this map well, but they need powerful hardware for every search. Small AI means models small enough to run on an ordinary computer. This one is about fifteen times smaller. A small model answers quickly and costs little to run. It can also keep documents on an organization's own machines, which matters for confidential data. GISTEmbed trains a small model to draw a better map, with a large model as its guide.",
      "A model learns the map from pairs of a question and a text that answers it, a small group of pairs at a time. For each question, training pulls its answer closer on the map. Every other text in the group is pushed away. This needs no extra labels, so training on millions of pairs works this way.",
      "Some of the pushed-away texts belong close. The question here is about capital cities, and another pair in the group holds a sentence on Canberra, the capital of Australia. Pushing it away teaches the small model that capital cities and Canberra are unrelated. Training data is far too large to check by hand, so wrong lessons like this slip in.",
      "GISTEmbed adds a check. Before each push, a large model, the guide, scores how close each text is to the question. The question's own answer, Ottawa, sets the bar. A text that the guide scores above the answer is left out of the push. The sentence on Canberra and the question about Australia stay where they are. Every other text is still pushed away, so the small model keeps learning from the rest.",
      "The small model learns from the guide's sense of meaning, and the wrong lessons are filtered out. The guide is needed only during training. Search then runs on the small model alone, with its speed and its low cost. The guide's own mistakes pass through, so the choice of guide matters. The training needs only pairs of a question and an answer, so an organization can use its own documents.",
      "GISTEmbed is open source, and Sentence Transformers includes it as GISTEmbedLoss.",
    ],
  },
];

export const videoById = (id) => videos.find((v) => v.id === id);

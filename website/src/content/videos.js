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
];

export const videoById = (id) => videos.find((v) => v.id === id);

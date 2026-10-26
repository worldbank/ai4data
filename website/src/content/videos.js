// Explainer videos of the program, listed on /videos and embedded with
// <Video id="..." /> (src/components/Video). Each record names the YouTube
// video, its area in the hub, the documentation and code of the method,
// and the transcript of its narration. `voice` is 'synthetic' for a
// narration read by a text-to-speech voice and 'recorded' for a person's
// voice; the videos page shows a note for synthetic narration.

export const videos = [
  {
    id: 'pcn-explainer',
    title: 'Proof-Carrying Numbers',
    youtube: 'dAXahKS3eg0',
    duration: '4:35',
    published: 'October 2026',
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
      "This video explains Proof-Carrying Numbers, a method for checking each figure in an answer from an AI chatbot against the data that the chatbot was given.",
      "Data producers now commonly offer AI chatbots that answer questions about their statistics. A person asks a question in plain language, and the chatbot writes an answer from the data.",
      "Here a user asks how internet use has changed in Kenya and Rwanda. The chatbot streams its answer word by word, and the answer contains three figures. The text of the answer does not show where any of these figures came from.",
      "Take the figure 35.0. It can come from a record in the database, here 34.9762, rounded to one decimal. The model can also recall the figure from its training data, which may hold an older estimate. In a third case, the model copies a record and changes a digit in writing, so that 36.0 becomes 35.0. The answer shows 35.0 in all three cases, and the reader has no means of telling which route the figure took.",
      "Proof-Carrying Numbers starts with the data. The application passes the values to the model as records, and each record carries a claim id. In this example, the claim id joins the indicator code, the country, and the year. The claim id can be any key that is unique within the data. A short hash of the record serves equally well. The verifier later uses the claim id to find the record. The highlighted record is internet use in Kenya in 2024, with the value 34.9762.",
      "The instructions to the model state the protocol. Every number that comes from the data goes inside a claim tag, which names the claim id of the record and the rule for comparing the two. The application sends the instructions, the data, and the question to the model together. The model streams its answer in the requested format. Each figure that it takes from the data arrives inside a claim tag.",
      "Here is one claim tag from that answer. The id names the record: internet users in Kenya in 2024. The policy and the decimals state the rule. The record, rounded to one decimal, must equal the figure. The chatbot displays the figure alone. The tag stays in the response for the verifier.",
      "The verifier is ordinary code. It reads the response from the model in real time, as the text streams in. When a claim tag is complete, the verifier parses it with the format that PCN defines, and extracts the claim id, the rule, and the figure. It then looks up the claim id in the data that the application added to the context. The record holds 34.9762, the rule rounds it to 35.0, and the figure in the answer is equal, so it is marked as verified. No language model takes part in the check. The comparison is mechanical and gives the same result on every run. The next tag names the record for Rwanda. The record holds 31.7033, which rounds to 31.7. The answer says 35.7, and the verifier flags the figure. The last figure, 70 percent, has no claim tag. With no record named, the verifier has nothing to check, and the figure receives no mark.",
      "In the chatbot, each checked figure carries its mark. A reader can see which figures were checked and which one failed. Hovering over a mark shows the record, the value in the data, and the rule, so a verified figure can be traced to a named record. A flagged figure is caught before the answer is used. A figure without a mark is known to need a check by hand.",
      "The verifier confirms that each tagged figure matches the record named in its tag. Whether that record is the right one for the question remains a judgement for the reader. PCN makes that review quicker. Each figure names its record, so the reviewer can compare the claim id with the question directly. Here the indicator is internet use and the country is Kenya, as the question asks. The year, 2024, is the latest in the data, and the reader decides whether that answers a question about change.",
      "The PCN library implements this check.",
    ],
  },
];

export const videoById = (id) => videos.find((v) => v.id === id);

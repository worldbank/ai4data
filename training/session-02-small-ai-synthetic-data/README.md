# Session 02 · Building specialized small AI using synthetic data

This session is in preparation. This page introduces its ideas for readers without a background in AI: small AI, how a model learns from examples, synthetic training data, and model distillation. One worked example runs through the page.

The worked example starts from a common problem. An organization holds thousands of reports and wants each one tagged with its topics, such as Education or Transport, from a fixed list of 86. Tagging by hand is slow. A large AI service could do the tagging, but every report would then be sent to that service. The organization also has no tagged examples from which to train a model of its own. The sections below build a small model for this task, step by step. The reports in the example are World Bank reports, and the model reads each report's abstract, the short summary at its start.

On the docs site, the figures on this page are interactive. On GitHub they are not shown, and the text describes their content.

## What small AI is

An AI model is a computer program that has learned a task from examples. A programmer does not write its rules. During training, the model adjusts a large set of internal numbers, called parameters, until its answers match the examples. The number of parameters is the usual measure of a model's size.

The chat assistants in wide use today run on large language models. These models can write, translate, summarize, and answer questions on almost any subject. The largest of them have more than 100 billion parameters and run in data centers, and each request travels over the internet to the company that runs the model.

A small model has far fewer parameters and is usually trained for one task. It can run on a laptop, a phone, or an organization's own server, without an internet connection. The World Bank's World Development Report 2026 uses the term small AI for models that are small in size, narrow in scope, or both, and that can run where computing power, connectivity, and devices are limited.

A model has to fit in the memory of the computer that runs it. The figure draws memory as squares of 1 GB and compares three models with three kinds of computer. The small model of the worked example needs about half of one square, 0.57 GB. Qwen3.8-27B, one of the large models used to build it, needs about 31 GB, more than a laptop holds.

<!-- figure: where-it-runs -->

A small model is a good candidate when the answer comes from a fixed list, such as a set of topics or codes, and when the same task is repeated many times. It also suits work where the data should stay inside the organization or where the internet connection is poor. A large model is the better choice when the answer is new text, such as a summary or a reply, or when the questions cannot be listed in advance. The figure sorts six tasks from development work between the two kinds of model.

<!-- figure: small-ai-tasks -->

The tagging task of the worked example has a fixed list of answers, many documents, and documents that the organization keeps in-house. It suits a small model.

## Learning from examples

A model learns a task from examples that carry the right answer, called labelled examples. For the tagging task, each labelled example is a sentence together with its topics.

The figure trains a model inside the browser. The model is much simpler than the one built in the session. It counts the words in the example sentences of four topics and gives a new text the topic with the closest match in words. Each press of the button adds one example sentence per topic. The figure then lists the words that the model links to each topic and tests the model on 60 real abstracts, 15 per topic.

<!-- figure: learn-from-examples -->

With one example per topic, the model tags 17 of the 60 abstracts correctly, close to the 15 expected from a random guess. With ten examples per topic it tags 31 correctly, and with forty it tags 34. Past that point this simple model stops improving. The model built in the session learns more from each example and is trained on about 20,000 sentences.

## Synthetic training data

The organization in the worked example has no tagged sentences. Synthetic data are examples written by a computer program, and in this session a large language model writes them. Each request to the model picks eight of the 86 topics and asks for 20 sentences about them, with the topics of each sentence listed. A thousand requests give about 20,000 labelled sentences.

The figure shows one request and the 20 sentences that came back. The sentences in the learning figure above come from the same set.

<!-- figure: synthetic-data -->

How the examples are written affects what the model learns. In the worked example the request was also sent with a rule that the sentences must not use the topic names. The model trained on those sentences did worse on real abstracts: an F1 score of 0.417, against 0.476 for the model trained with names allowed. The F1 score runs from 0 to 1. It is high when the assigned topics are correct and few of the recorded topics are missed. The "go deeper" part of the figure shows the two versions side by side.

Before training, a script also checks the sentences. It removes sentences without a topic, topics outside the list of 86, and duplicate sentences. In the worked example these checks found nothing to remove.

## Model distillation

Distillation passes a skill from a large model to a small one. The large model acts as a teacher and the small model as a student. In its original form, described by Hinton, Vinyals, and Dean (2015), the student learns to give the same answers as the teacher on the same inputs. In this session the teacher also writes the practice material: it writes the sentences and lists their topics, and the student learns from them.

The teacher is needed only while the student learns. Afterwards the student works alone on the organization's own computer, and the reports stay there. The figure shows the two stages and then the student at work on one real abstract. The student gives every topic a score and assigns the topics with a clear score, up to five.

<!-- figure: distillation -->

The student and three large models were tested on the same 300 abstracts. The student scored 0.52, and the large models, asked directly for the topics, scored 0.52 to 0.53. The student takes 0.19 seconds per abstract on an ordinary processor, without a GPU or an internet connection. The large models take 0.5 to 3.9 seconds. The student reached its score in two stages. Trained on the first 20,296 sentences, it scored 0.47. In further rounds, a program built on a large model read the student's mistakes on a separate set of abstracts and asked the teachers for more sentences on the topics involved, which raised the score to 0.52.

<!-- figure: distillation-results -->

These results come from one task. The topics recorded for each report in the World Bank documents database serve as the right answers, and a record can omit a topic that the abstract covers. The 300 test abstracts cover 44 of the 86 topics, so the scores say nothing about the other 42. On a second task in the same experiments, coding job titles to the ISCO occupation classification, the student scored 0.15 below the best large model. A small model has to be tested on real data from the place where it will be used before anyone relies on it.

## Sources

- World Bank. 2026. *World Development Report 2026: The Promise of Artificial Intelligence*. Washington, DC: World Bank. Box O.1 and spotlight 1.
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- The data behind the figures are in [assets/overview.json](assets/overview.json).

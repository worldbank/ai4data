# Session 02 · Building specialized small AI using synthetic data

This session is in preparation. This page introduces its three ideas in order: small AI, synthetic training data, and model distillation. It assumes no background in AI. One worked example runs through all three ideas: a small model learns to assign topics, such as Urban Development or Migration, to the abstracts of World Bank reports. An abstract is the short summary at the start of a report.

On the docs site, the figures on this page are interactive. On GitHub they are not shown, and the text describes their content.

## What small AI is

An AI model is a computer program that has learned a task from examples. A programmer does not write its rules. During training, the model adjusts a large set of internal numbers, called parameters, until its answers match the examples. The number of parameters is the usual measure of a model's size.

The chat assistants in wide use today run on large language models with tens or hundreds of billions of parameters. These models can write, translate, summarize, and answer questions on almost any subject. They run in data centers, and each request travels over the internet to the company that runs the model.

A small model has far fewer parameters and is usually trained for one task. It can run on a laptop, a phone, or an organization's own server, without an internet connection. The World Bank's World Development Report 2026 uses the term small AI for models that are small in size, narrow in scope, or both, and that can run where computing power, connectivity, and devices are limited.

The size of a model determines the computer needed to run it. The figure places the three models of the worked example on a scale of size, divided into three ranges: models that fit on a phone or laptop, models that need a server, and models that need a data center.

<!-- figure: small-ai-scale -->

The small model of the worked example has 142 million parameters and needs 0.57 GB of memory. The two large models used to train it have about 30 billion parameters each, and each one needs a computer with a graphics processor (GPU) built for AI work.

A small model is a good candidate when the answer comes from a fixed list, such as a set of topics or codes, and when the same task is repeated many times. It also suits work where the data should stay inside the organization or where the internet connection is poor. A large model is the better choice when the answer is new text, such as a summary or a reply, or when the questions cannot be listed in advance. The figure sorts six tasks from development work between the two kinds of model.

<!-- figure: small-ai-tasks -->

A small model does only the one task of its training. Like any model, it can also fail on material that differs from its training examples, such as documents in another language or photos taken in different light. A small model therefore has to be tested on real data from the place where it will be used before anyone relies on it.

## Synthetic training data

A model learns a task from examples that carry the right answer, called labelled examples. In the worked example, each labelled example is a sentence together with its topics. Training a model this way takes thousands of examples, and few organizations have that many for a new task.

Synthetic data are examples written by a computer program. In this session a large language model writes them. Each request to the model picks eight of the 86 topics and asks for 20 sentences about them, with the topics of each sentence listed. A thousand requests give about 20,000 labelled sentences.

The figure follows one request through four steps: the choice of topics, the request to the model, the sentences returned, and a comparison. The request was written twice with one rule changed. In the first version the sentences may use the topic names. In the second, they must describe the subject in other words. With names allowed, 73 percent of the sentences contain a topic name; with names avoided, 8.9 percent do.

<!-- figure: synthetic-data -->

Each version was used to train a small model, and both models were tested on 300 real abstracts. The test uses the F1 score, which runs from 0 to 1. The score is high when the assigned topics are correct and few of the recorded topics are missed. The model trained with names allowed scored 0.476, and the model trained with names avoided scored 0.417. The wording of the synthetic examples changed what the model learned, and the test on real documents is what showed the difference.

Before training, a script also checks the sentences. It removes sentences without a topic, topics outside the list of 86, and duplicate sentences. In both versions these checks found nothing to remove.

## Model distillation

Distillation passes a skill from a large model to a small one. The large model acts as a teacher and the small model as a student. In its original form, described by Hinton, Vinyals, and Dean (2015), the student learns to give the same answers as the teacher on the same inputs. In this session the teacher also writes the practice material: it writes the sentences and lists their topics, and the student learns from them.

The teacher is needed only while the student learns. Afterwards the student works alone on the organization's own computer, and the documents stay there. The first part of the figure shows the two stages.

<!-- figure: distillation -->

The second part of the figure shows the student at work on six real abstracts. The student gives every topic a score from 0 to 1 and assigns the topics whose score passes a threshold, up to five topics. A slider moves the threshold. The answers of three large models, asked directly for the topics of the same abstracts, are shown for comparison.

On the 300 test abstracts, the student trained on the first 20,296 sentences scored 0.468. Further rounds of training raised the score to 0.521. In each round, a program built on a large model read the student's mistakes on a separate set of abstracts and asked the teachers for more sentences on the topics involved in those mistakes. The three large models asked directly scored between 0.523 and 0.530. The student assigns the topics of an abstract in 7 milliseconds on a GPU and in 0.19 seconds on an ordinary processor (CPU). The large models take between 0.5 and 3.9 seconds.

<!-- figure: distillation-results -->

These results come from one task. The topics recorded for a report in the World Bank documents database serve as the right answers, and a record can omit a topic that the abstract covers. The 300 test abstracts cover 44 of the 86 topics, so the scores say nothing about the other 42. On a second task in the same experiments, coding job titles to the ISCO occupation classification, the student scored 0.15 below the best large model.

## Sources

- World Bank. 2026. *World Development Report 2026: The Promise of Artificial Intelligence*. Washington, DC: World Bank. Box O.1 and spotlight 1.
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- The data behind the figures are in [assets/overview.json](assets/overview.json).

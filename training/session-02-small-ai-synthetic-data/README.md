# Session 02 · Building specialized small AI using synthetic data

This session is in preparation. This page introduces its ideas for readers without a background in AI: small AI, synthetic data, and model distillation. One example runs through the page. Linked terms are defined in the Terms section at the end of the page.

On the docs site, the figures on this page are interactive, and a linked term shows its definition when selected. On GitHub the figures are not shown, and the text describes their content.

## The example

An organization holds thousands of reports and wants each one tagged with its topics, such as Education or Transport, from a fixed list of 86. Tagging by hand is slow. A large AI service could do the tagging, but every report would then be sent to the company that runs the service. The organization also has no tagged reports from which to train a [model](#terms) of its own.

The figure shows the three steps that this session follows to build a small model for the task. The reports in the example are World Bank reports, and the model reads the [abstract](#terms) of each one.

<!-- figure: overview -->

## Why a small model

The chat assistants in wide use today run on [large language models](#terms). These models can write, translate, summarize, and answer questions on almost any subject. The largest have more than 100 billion [parameters](#terms) and run in data centers, so each request travels over the internet to the company that runs the model.

A small model has far fewer parameters and is trained for one task. The World Bank's World Development Report 2026 uses the term [small AI](#terms) for such models. A small model can run on a laptop or on an organization's own server, so the documents do not leave the organization. It works without an internet connection, and it needs no paid service and no [GPU](#terms).

A model has to fit in the memory of the computer that runs it. The figure draws memory as squares of 1 GB. The small model of the example needs about half of one square. A large model of the size used as teacher in the example needs about 31 squares, more than a laptop holds.

<!-- figure: where-it-runs -->

A small model does one task only, and it has to be built for that task. Building it requires many examples with the right answers. Step 1 produces those examples.

## Step 1: synthetic data

A model learns a task from [labelled examples](#terms), which are examples together with the right answer. For the tagging task, a labelled example is a sentence together with its topics. The organization in the example has none.

[Synthetic data](#terms) are examples written by a computer program. In this session a large language model writes them. Each request names eight of the 86 topics and asks for 20 sentences about them, with the topics of each sentence listed. A thousand requests give about 20,000 labelled sentences. The figure shows one request and the 20 sentences that came back.

<!-- figure: synthetic-data -->

A script checks the sentences before they are used. It removes sentences without a topic, topics outside the list of 86, and repeated sentences. In the example these checks found nothing to remove.

The wording of the request also affects the result. The "go deeper" part of the figure shows a second version of the request, in which the sentences had to avoid the topic names. The small model trained on that version did worse on real reports.

## Step 2: the small model learns

[Training](#terms) a model means showing it labelled examples until its answers match them. The figure trains a very simple model inside the browser, on the sentences of step 1 for four topics. The model is then tested on 60 real reports whose topics are known.

<!-- figure: learn-from-examples -->

With one example per topic, the model gives 17 of the 60 reports the right topic, close to the 15 expected from a random guess. With ten examples per topic it gets 31 right, and with forty it gets 34. The model in the figure counts words only, and it stops improving after about 20 examples per topic. The small model built in the session is a more capable kind of model, which takes the meaning of whole sentences into account. It learns from all 20,000 examples and all 86 topics.

This way of training is called [distillation](#terms). A large model passes a skill to a small one: the large model acts as the teacher and the small model as the student. In its original form, described by Hinton, Vinyals, and Dean (2015), the student learns to give the teacher's answers on the same inputs. In this session the teacher also writes the examples of step 1.

## Step 3: the small model at work

After training, the student works alone. It needs neither the teacher nor an internet connection, and the reports stay on the organization's computer. The figure shows the student at work on one real report.

<!-- figure: distillation -->

The student and three large models were tested on the same 300 report abstracts. On a score from 0 to 1, the student scored 0.52 and the large models scored 0.52 to 0.53. The student takes 0.19 seconds per abstract on an ordinary processor. The large models take 0.5 to 3.9 seconds on a GPU or through an online service.

<!-- figure: distillation-results -->

The results come from one task. The right answers are the topics recorded for each report in the World Bank documents database, and a record can omit a topic that the abstract covers. The 300 test reports cover 44 of the 86 topics, so the scores say nothing about the other 42. On a second task in the same experiments, coding job titles to an occupation classification, the student scored 0.15 below the best large model.

## Does a task suit a small model?

A small model is a good candidate when the answer comes from a fixed list, such as topics or codes, and the same task is repeated many times. It also suits work where the data should stay inside the organization or where the internet connection is poor. A large model is the better choice when the answer is new text, such as a summary or a reply, or when the questions cannot be listed in advance. The figure sorts six tasks from development work between the two kinds of model.

<!-- figure: small-ai-tasks -->

Before a team relies on a small model, it checks the following points:

- The model has been tested on real documents from the place where it will be used, with known answers.
- The test covers the topics or codes that matter for the work.
- The model's score has been compared with that of a large model on the same test.
- A person reviews a sample of the model's output at regular intervals.

## Terms

- **Abstract**: The short summary at the start of a report.
- **Distillation**: Training a small model to give the answers of a large model. The large model is called the teacher and the small model the student.
- **GPU**: A graphics processor. Large models run on GPUs, which cost more and use more power than an ordinary processor.
- **Labelled example**: An example together with the right answer. In this session, a sentence together with its topics.
- **Large language model**: A model trained on a very large amount of text. It can write, translate, summarize, and answer questions. Chat assistants run on such models.
- **Model**: A computer program that has learned a task from examples. A programmer does not write its rules.
- **Parameter**: One of the internal numbers of a model, adjusted during training. The number of parameters measures the size of a model.
- **Small AI**: The term of the World Development Report 2026 for models that are small in size, narrow in scope, or both, and that can run where computing power, connectivity, and devices are limited.
- **Synthetic data**: Examples written by a computer program. In this session a large language model writes them.
- **Training**: The step in which a model adjusts its parameters until its answers match a set of labelled examples.

## Sources

- World Bank. 2026. *World Development Report 2026: The Promise of Artificial Intelligence*. Washington, DC: World Bank. Box O.1 and spotlight 1.
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- The data behind the figures are in [assets/overview.json](assets/overview.json).

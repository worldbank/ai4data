# Session 02 · Building specialized small AI using synthetic data

This session is in preparation. This page introduces its ideas for readers without a background in AI: small AI, synthetic data, and model distillation. It opens with tasks from development work that suit a small model, explains why the examples for training such a model are often written by a large model, and describes the process in nine steps. A worked example then follows the process on one task, with measured results. Linked terms are defined in the Terms section at the end of the page.

## Tasks for a small model

A small model can take on a narrow task that repeats many times, on an organization's own computer. The figure shows six such tasks from development work: matching job seekers to vacancies at a public employment office, sorting complaints sent by text message to a project's grievance mechanism, picking out requests for help from messages after a flood, answering farmers' questions from an extension guide, flagging errors in benefit applications, and coding the occupations written in a household survey. For each task, the figure shows why no training data exist and one practice case with its answer. The tasks and practice cases were written for this page as illustrations.

<!-- figure: use-cases -->

## Why a small model

The chat assistants in wide use today run on [large language models](#terms). These models can write, translate, summarize, and answer questions on almost any subject. The largest have more than 100 billion [parameters](#terms) and run in data centers, so each request travels over the internet to the company that runs the model. The companies behind the best-known chat assistants generally do not publish the size of their main models.

A small model has far fewer parameters and is often trained for one task. The World Bank's World Development Report 2026 uses the term [small AI](#terms) for AI that is small in size or narrow in scope and is built to run where power, internet connectivity, and digital devices are scarce. A small model can run on a laptop or on an organization's own server, so the documents do not leave the organization. Once built, it works without an internet connection, and it runs without a paid service or a [GPU](#terms). Building it has a cost: a large model writes the examples, and the training runs on a GPU.

A model has to fit in the memory of the computer that runs it. The report gives rounded memory ranges for three kinds of computer: about 2 to 20 GB for a phone or laptop, 20 to 200 GB for a single server, and more than 200 GB in a data center. The figure draws memory as squares of 1 GB. The small model of the worked example needs about half of one square. Qwen3.8-27B, one of the teachers of the worked example, needs about 31 squares in the 8-bit version used here, more than a laptop holds.

<!-- figure: where-it-runs -->

A small model does one task only, and it has to be built for that task. Building it requires many examples with the right answers.

## Why synthetic data

A model learns a task from [labelled examples](#terms), which are examples together with the right answer. For the tasks above, such examples are often missing. Messages sent after a flood do not exist before it. Complaints, applications, and job seekers' profiles hold personal data and cannot be sent to an outside service. The cases that matter most, such as an emergency among routine messages, are rare. Real messages mix local languages and abbreviations, and few labelled examples exist in most local languages.

[Synthetic data](#terms) are examples written by a computer program. The World Development Report 2026 describes them as data that mimic real-world patterns and as an alternative where real data are too scarce or too sensitive to use. In this session a large language model writes them. The figure shows four reasons for writing them; under "Rare", a button adds written cases to a set of messages.

<!-- figure: why-synthetic -->

The figure ends with a comparison from office work: a new staff member who learns from practice cases written by an experienced colleague, and who is then tested on real files.

## How synthetic data are made

The figure describes the process in nine steps. For each step it gives the office comparison, what happens, why the step is needed, and a demonstration based on the complaints example. A mark under each step shows whether the worked example used it.

<!-- figure: workflow -->

1. **Describe the task.** The team writes the list of possible answers with a definition of each. The large model, the people who check its cases, and the test all use the same definitions.
2. **Show a few real cases.** A few dozen real examples, with personal details removed, go into the request to the large model as [seed examples](#terms). Without them, a large model writes tidy, generic text that differs from real messages. A check makes sure that the output does not copy them. The worked example did not use this step.
3. **Vary the cases.** Each request sets a different combination of attributes, such as the answer, the length, the style, the language, and the difficulty. A small model learns only from the kinds of cases in its training set. In the experiments behind this session, Qwen3.8-27B was asked many times for job titles of the same occupation groups, and 6,612 of the 18,000 titles written (37%) repeated an earlier title.
4. **Write and check.** The large model writes the cases with their answers. A second model labels each case on its own, and scripts remove repeats, answers outside the list, and cases where the two models disagree.
5. **Train the small model** on the checked cases.
6. **Test on real cases.** A few hundred real cases with answers given by people are set aside for testing only. Practice cases share the writing habits of the large model, so a high score on them can fail to hold on real cases.
7. **Practise the weak spots.** The model is tested on a second set of real cases, and a program finds the kinds of cases with the most mistakes. The large model writes more cases of those kinds, and the model is trained again. In the worked example, these targeted rounds scored 0.037 and 0.041 higher than rounds of the same size spread evenly over all topics, in two repeats of the experiment. On job titles the difference was 0.012, and its 95% interval included zero. Choosing what to practise from the model's mistakes is related to [curriculum learning](#terms), which orders the training cases from easy to hard; the ordering itself was not tested.
8. **Ask people which is better.** Experienced staff choose the better of two practice cases or two answers. This [preference feedback](#terms) improves the request to the large model and filters the cases. Where the small model writes text, such as an answer to a farmer, it can also train the model towards the preferred answers. The experiments did not use this step.
9. **Watch and refresh.** People review a sample of the model's answers at regular intervals, and new practice cases are written when the answers, the language, or the situation change. The World Development Report 2026 describes a screening tool for diabetic eye disease that rejected more than one-fifth of the images nurses took, because the lighting in the clinic differed from that in its training images.

## The worked example

An organization holds thousands of reports and wants each one tagged with its topics, such as Education or Transport, from a fixed list of 86. Tagging by hand is slow. A large AI service could do the tagging, but every report would then be sent to the company that runs the service. The organization also has no tagged reports from which to train a [model](#terms) of its own.

The figure shows the three steps of the worked example. They cover steps 4 to 6 of the process above: writing and checking the examples, training, and testing. The worked example also used steps 1 and 7, and step 3 in part. The reports in the example are World Bank reports, and the model reads the [abstract](#terms) of each one.

<!-- figure: overview -->

## Step 1: synthetic data

For the tagging task, a labelled example is a sentence together with its topics. The organization in the worked example has none, so a large language model writes them. Each request names eight of the 86 topics and asks for 20 sentences about them, with the topics of each sentence listed. A thousand requests give about 20,000 labelled sentences. The figure shows one request and the 20 sentences that came back.

<!-- figure: synthetic-data -->

A script checks the sentences before they are used. It removes sentences without a topic, topics outside the list of 86, and repeated sentences. In the worked example these checks found nothing to remove.

The wording of the request also affects the result. The "go deeper" part of the figure shows a second version of the request, in which the sentences had to avoid the topic names. The small model trained on that version did worse on real reports.

## Step 2: the small model learns

[Training](#terms) a model means showing it labelled examples until its answers match them. The figure trains a very simple model inside the browser, on the sentences of step 1 for four topics. The model is then tested on 60 real reports whose topics are known.

<!-- figure: learn-from-examples -->

With one example per topic, the model gives 17 of the 60 reports the right topic, close to the 15 expected from a random guess. With ten examples per topic it gets 31 right, and with forty it gets 34. The model in the figure counts words only, and it stops improving after about 20 examples per topic. The small model built in the session is a more capable kind of model, which takes the meaning of whole sentences into account. It learns all 86 topics from about 20,000 examples, and from another 12,000 to 15,000 in further rounds.

This way of training is called [distillation](#terms). A large model passes a skill to a small one: the large model acts as the teacher and the small model as the student. In its original form, described by Hinton, Vinyals, and Dean (2015), the student learns to match the teacher's probability for every possible answer, which carries more information than the teacher's top answer alone. In this session the teacher also writes the examples of step 1.

## Step 3: the small model at work

After training, the student works alone. It needs neither the teacher nor an internet connection, and the reports stay on the organization's computer. The figure shows the student at work on one real report.

<!-- figure: distillation -->

The student and three large models were tested on the same 300 report abstracts. On a score from 0 to 1, the student scored 0.52 and the large models scored 0.52 to 0.53. The student takes 0.19 seconds per abstract on an ordinary processor. The large models take 0.5 to 3.9 seconds on a GPU or through an online service.

<!-- figure: distillation-results -->

The results come from one task. The right answers are the topics recorded for each report in the World Bank documents database, and a record can omit a topic that the abstract covers. The 300 test reports cover 44 of the 86 topics, so the scores say nothing about the other 42. On a second task in the same experiments, coding job titles to an occupation classification, the student scored 0.15 below the best large model.

## Does a task suit a small model?

A small model is a good candidate when the answer comes from a fixed list, such as topics or codes, and the same task is repeated many times. It also suits work where the data should stay inside the organization or where the internet connection is poor. A large model is the better choice when the answer is new text, such as a summary or a reply, or when the questions cannot be listed in advance. The figure lists six more tasks from development work, different from those at the top of the page. For each task, the reader chooses a small or a large model and sees the suggested answer with its reason.

<!-- figure: small-ai-tasks -->

Before a team relies on a small model, it checks the following points:

- The model has been tested on real documents from the place where it will be used, with known answers.
- The test covers the topics or codes that matter for the work.
- The model's score has been compared with that of a large model on the same test.
- A person reviews a sample of the model's output at regular intervals.

## Terms

- **Abstract**: The short summary at the start of a report.
- **Curriculum learning**: Training a model on cases in a planned order, usually from easy to hard. In a wider sense, choosing which cases to practise next from the model's results.
- **Distillation**: Training a small model to reproduce the answers of a large model. The large model is called the teacher and the small model the student.
- **GPU**: A graphics processor, now also used to run AI models. Large models run on GPUs or similar chips, which cost more and use more power than an ordinary processor.
- **Labelled example**: An example together with the right answer. In this session, a sentence together with its topics.
- **Large language model**: A model trained on a very large amount of text. It can write, translate, summarize, and answer questions. Chat assistants run on such models.
- **Model**: A computer program that has learned a task from examples. A programmer does not write its rules.
- **Parameter**: One of the internal numbers of a model, adjusted during training. The number of parameters measures the size of a model.
- **Preference feedback**: Choices made by people between two outputs, such as two practice cases or two answers. The choices are used to filter cases, to improve a request, or to train a model towards the preferred outputs.
- **Seed example**: A real example placed in the request to a large model, so that its output resembles real cases.
- **Small AI**: The term of the World Development Report 2026 for AI that is small in size or narrow in scope and is built to run where power, internet connectivity, and digital devices are scarce. The report notes that a large model in the cloud, reached by voice call or text message from a basic phone, can serve the same purpose.
- **Synthetic data**: Examples written by a computer program. In this session a large language model writes them.
- **Training**: The step in which a model adjusts its parameters until its answers match a set of labelled examples.

## Sources

- World Bank. 2026. *World Development Report 2026: The Promise of Artificial Intelligence*. Washington, DC: World Bank. doi:10.1596/978-1-4648-2331-2. Box O.1, "Small AI, big impact"; spotlight 1, table S1.1 and p. 103; chapter 7, p. 248.
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- The data behind the figures are in [assets/overview.json](assets/overview.json).

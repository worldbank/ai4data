# Session 02 · Building specialized small AI using synthetic data

This session is in preparation. This page sets out its three ideas: small AI, synthetic training data, and model distillation. Each idea is illustrated with figures from the worked example of the session, in which a small model learns to assign topics to the abstracts of World Bank reports.

On the docs site, the figures on this page are interactive. On GitHub they are not shown, and the text describes their content.

## What small AI is

The World Development Report 2026, *The Promise of Artificial Intelligence*, describes small AI in box O.1 and spotlight 1. The report gives two meanings of "small". In the technology industry, small refers to model size: a small language model is trained to approach the performance of a large one with far fewer parameters, the internal values that store what a model has learned. Small can also refer to scope: a model built for one narrow task, such as identifying a crop disease or screening a medical image. In both meanings, the report describes small AI as built to run where power, internet connectivity, and digital devices are scarce.

Size decides where a model can run. The report gives three bands: a model of up to 10 billion parameters runs on a phone or laptop without a connection; a model of 10 to 100 billion parameters needs a single server on a local network; a larger model runs in a data center and needs a constant internet connection. The smallest model about as capable as the first ChatGPT went from 540 billion parameters in 2022 to 3.8 billion in 2024.

<!-- figure: small-ai-scale -->

The model built in this session has 142 million parameters, about one seventieth of the upper limit of the first band. The two large language models used to train it, Qwen3.8-27B and Gemma 4 31B, run on one GPU each of a single server.

The report distinguishes a model from an application, which wraps a model in an interface that people use. A small AI application can be built on a small model that runs on a phone, as in Nuru and Cough Against TB, or on a large model in the cloud reached over text or a voice call, as in Rori and Ask Viamo Anything. The applications built on a single-task model are usually predictive: they pick a label. The report notes that a single-task model is easier to build and simpler to evaluate than a general-purpose one. The classifier of this session is of that kind.

<!-- figure: small-ai-applications -->

The report also sets out the limits of small AI. A smaller model can do less than a larger one, especially on complex or open-ended tasks. It also fails more often in conditions that differ from those of its training data. The report's example is a screening tool for diabetic eye disease that rejected more than one fifth of the images taken by nurses, because the lighting in the clinic differed from that of its training images. The report concludes that a tool should be tested where it will be used, with a higher bar for a high-stakes task than for an everyday one.

## Synthetic training data

A classifier learns from labelled examples, each a text with its topics. Few organizations have thousands of labelled examples for a new task. In this session a large language model writes them. Each request to the model lists eight of the 86 topics and asks for 20 sentences about them, each labelled with its topics. A thousand requests give about 20,000 labelled sentences.

How the sentences are written changes what the small model learns. The figure follows one request written twice by Qwen3.8-27B with one rule changed: in the first set, the sentences may use the topic names, and in the second, they must describe the subject in other words. With names allowed, 73 percent of the sentences contain the name of one of their topics; with names avoided, 8.9 percent do. The small model trained on the first set scored higher on real report abstracts: a micro F1 of 0.476 against 0.417.

<!-- figure: synthetic-data -->

Before training, a script checks the sentences: it removes labels that are not in the list of topics, sentences without a label, and duplicates, and it counts the sentences that name their topics. The checks found no unknown labels, unlabelled sentences, or duplicates in either set.

## Model distillation

Knowledge distillation trains a small model, the student, to reproduce the behaviour of a large one, the teacher. In its original form the student learns from the teacher's output probabilities on the same inputs. In this session the teacher writes the inputs as well: the large model writes the sentences and assigns their topics, and the small model learns from them. The large model is used only while the small model is built. The small model then labels documents on its own, on a CPU or a GPU of the organization, without sending the text to a service.

<!-- figure: distillation -->

The small model is evaluated on 300 abstracts of World Bank reports, with the topics recorded for them in the documents database as the reference. Trained on the first 20,296 sentences, it reached a micro F1 of 0.468. Further rounds of generation raised the score to 0.521. In each round, an agent read the small model's errors on a separate set of abstracts and asked the large models for sentences on the topics that the small model confused. Three large models asked directly for the topics of the same abstracts scored between 0.523 and 0.530. The small model labels an abstract in 7 milliseconds on a GPU and in 0.19 seconds on four CPU threads.

<!-- figure: distillation-results -->

The comparison has limits. The reference topics of an abstract are those recorded in the documents database. An abstract can be about a topic that the record does not list, and the figure with six abstracts shows examples. The 300 abstracts cover 44 of the 86 topics, so the scores say nothing about the other 42. On a second task of the same experiments, occupation titles coded to the ISCO classification, the small model trailed the best large model by 0.15.

## Sources

- World Bank. 2026. *World Development Report 2026: The Promise of Artificial Intelligence*. Washington, DC: World Bank. Box O.1, "Small AI, big impact", and spotlight 1, "Small AI, Big Impact", with tables S1.1 to S1.3.
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- The figures of the worked example are in [assets/overview.json](assets/overview.json).

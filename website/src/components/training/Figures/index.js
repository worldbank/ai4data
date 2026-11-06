import React from 'react';
import Overview from './Overview';
import UseCases from './UseCases';
import WhySynthetic from './WhySynthetic';
import Workflow from './Workflow';
import WhereItRuns from './WhereItRuns';
import Learn from './Learn';
import TaskFit from './TaskFit';
import SyntheticData from './SyntheticData';
import Distillation from './Distillation';
import DistillationResults from './DistillationResults';

// Interactive figures of the training pages. A session README places one with a
// line `<!-- figure: name -->`; scripts/docs/build_training_site.py replaces the
// line with <TrainingFigure name="name" data={figures} />, where `figures` is the
// session's assets/overview.json.
const FIGURES = {
  overview: () => <Overview />,
  'use-cases': () => <UseCases />,
  'why-synthetic': () => <WhySynthetic />,
  workflow: (data) => <Workflow data={data.workflow} />,
  'where-it-runs': (data) => <WhereItRuns data={data.smallAI} />,
  'small-ai-tasks': (data) => <TaskFit data={data.smallAI} />,
  'learn-from-examples': (data) => <Learn data={data.learn} />,
  'synthetic-data': (data) => <SyntheticData data={data.synthetic} />,
  distillation: (data) => <Distillation data={data.distillation} />,
  'distillation-results': (data) => <DistillationResults data={data.distillation} />,
};

export default function TrainingFigure({name, data}) {
  const render = FIGURES[name];
  if (!render) {
    return <p>Unknown figure: {name}</p>;
  }
  return render(data);
}

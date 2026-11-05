import React from 'react';
import SizeScale from './SizeScale';
import Applications from './Applications';
import SyntheticData from './SyntheticData';
import Distillation from './Distillation';
import DistillationResults from './DistillationResults';

// Interactive figures of the training pages. A session README places one with a
// line `<!-- figure: name -->`; scripts/docs/build_training_site.py replaces the
// line with <TrainingFigure name="name" data={figures} />, where `figures` is the
// session's assets/overview.json.
const FIGURES = {
  'small-ai-scale': (data) => <SizeScale data={data.smallAI} />,
  'small-ai-applications': (data) => <Applications data={data.smallAI} />,
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

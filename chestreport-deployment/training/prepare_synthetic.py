"""Author a small synthetic format-training experiment. No patient data or clinical review.
Run from repository root: python -m training.prepare_synthetic
"""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from backend.prompts import PROMPT_VERSION, analysis_task, build_prompt, QA_TASK

PROVENANCE = 'chestreport-authored-synthetic-v1'
# Each row is a deliberately authored sentence and its explanation, never a diagnosis
# inferred from an image. These templates have NOT been reviewed by a clinician.
ATOMS = [
    ('Pleural effusion', 'present', 'Small left pleural effusion.', 'The report describes a small amount of fluid around the left lung.'),
    ('Pleural effusion', 'present', 'Small right pleural effusion.', 'The report describes a small amount of fluid around the right lung.'),
    ('Pleural effusion', 'absent', 'No pleural effusion.', 'The report does not show fluid around the lungs.'),
    ('Pleural effusion', 'uncertain', 'Possible small left pleural effusion.', 'A small amount of fluid around the left lung is possible, but not confirmed.'),
    ('Pneumothorax', 'present', 'Small right apical pneumothorax.', 'The report describes a small amount of air around the top of the right lung.'),
    ('Pneumothorax', 'present', 'Small left apical pneumothorax.', 'The report describes a small amount of air around the top of the left lung.'),
    ('Pneumothorax', 'absent', 'No pneumothorax.', 'The report does not show air around the lungs.'),
    ('Pneumothorax', 'uncertain', 'A tiny right apical pneumothorax cannot be excluded.', 'A tiny amount of air around the top of the right lung cannot be ruled out.'),
    ('Atelectasis', 'present', 'Mild left basilar atelectasis.', 'The report describes mildly reduced air in part of the left lung base.'),
    ('Atelectasis', 'present', 'Mild right basilar atelectasis.', 'The report describes mildly reduced air in part of the right lung base.'),
    ('Atelectasis', 'absent', 'No atelectasis.', 'The report does not describe atelectasis, which means reduced air in part of a lung.'),
    ('Atelectasis', 'uncertain', 'Possible left basilar atelectasis.', 'Reduced air in part of the left lung base is possible, but not confirmed.'),
    ('Cardiomegaly', 'present', 'Mild cardiomegaly.', 'The heart silhouette appears mildly enlarged in the report.'),
    ('Cardiomegaly', 'absent', 'No cardiomegaly.', 'The report does not describe an enlarged heart silhouette.'),
    ('Cardiomegaly', 'uncertain', 'Possible mild cardiomegaly.', 'The heart silhouette may be mildly enlarged, but this is not confirmed.'),
    ('Consolidation', 'present', 'Left lower lobe consolidation.', 'The report describes an area of denser lung tissue in the left lower lobe.'),
    ('Consolidation', 'present', 'Right lower lobe consolidation.', 'The report describes an area of denser lung tissue in the right lower lobe.'),
    ('Consolidation', 'absent', 'No focal consolidation.', 'The report does not describe a focal area of dense lung tissue called consolidation.'),
    ('Consolidation', 'uncertain', 'Possible right lower lobe consolidation.', 'An area of denser tissue in the right lower lobe is possible, but not confirmed.'),
    ('Pulmonary edema', 'present', 'Mild pulmonary edema.', 'The report describes mild fluid accumulation within the lungs.'),
    ('Pulmonary edema', 'absent', 'No pulmonary edema.', 'The report does not show fluid accumulation within the lungs.'),
    ('Pulmonary edema', 'uncertain', 'Possible mild pulmonary edema.', 'Mild fluid accumulation within the lungs is possible, but not confirmed.'),
]

def build_rows():
    # A pair of findings is one indivisible source group. All questions/styles for
    # that same report remain in one split. Templates are shared across splits:
    # this is compositional plumbing evaluation, NOT external clinical validation.
    pairs = [(a, b) for a, b in itertools.combinations(ATOMS, 2) if a[0] != b[0]]
    random.Random(42).shuffle(pairs)
    pairs = pairs[:200]
    result = {name: [] for name in ('train', 'validation', 'test')}
    for i, pair in enumerate(pairs):
        split = 'train' if i < int(.8 * len(pairs)) else 'validation' if i < int(.9 * len(pairs)) else 'test'
        text = 'FINDINGS: ' + ' '.join(a[2] for a in pair)
        group = hashlib.sha256(text.encode()).hexdigest()[:16]
        sources = [{'id': 'S1', 'page': 1, 'section': 'Findings', 'text': text}]
        def row(task, context, target, kind):
            result[split].append({'patient_id': 'synthetic-' + group, 'reviewed': False,
                'provenance': PROVENANCE, 'prompt_version': PROMPT_VERSION, 'task': kind,
                'sources': sources, 'messages': [
                    {'role': 'user', 'content': build_prompt(task, context)},
                    {'role': 'assistant', 'content': json.dumps(target)}]})
        findings = [{'name': a[0], 'explanation': a[3], 'status': a[1], 'source_id': 'S1', 'evidence': a[2]} for a in pair]
        for style in ('plain', 'clinical'):
            summary = [{'text': a[3] if style == 'plain' else a[2], 'source_id': 'S1', 'evidence': a[2]} for a in pair]
            row(analysis_task(style), sources, {'findings': findings, 'summary': summary}, 'analysis-' + style)
        a = pair[i % 2]
        row(QA_TASK, {'question': 'What does the report say about ' + a[0].lower() + '?', 'sources': sources},
            {'answer': a[3], 'abstained': False, 'citations': [{'source_id': 'S1', 'evidence': a[2]}]}, 'qa-supported')
        row(QA_TASK, {'question': 'What treatment should I take for ' + a[0].lower() + '?', 'sources': sources},
            {'answer': 'The report does not specify treatment. I cannot recommend treatment from this report; discuss this with your clinician.', 'abstained': True, 'citations': []}, 'qa-abstain')
    return result

def write_dataset(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = build_rows()
    for name, rows in splits.items():
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    manifest = {'provenance': PROVENANCE, 'reviewed': False, 'seed': 42, 'prompt_version': PROMPT_VERSION,
        'counts': {k: len(v) for k, v in splits.items()},
        'limitation': 'Authored templates share vocabulary across splits. Scores measure format learning on synthetic combinations, not clinical generalization.'}
    (output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # A held-out source to paste into the app after connecting the GPU server.
    (output / 'synthetic-demo-report.txt').write_text(splits['test'][0]['sources'][0]['text'], encoding='utf-8')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training/data/synthetic')
    print(json.dumps(write_dataset(parser.parse_args().output), indent=2))

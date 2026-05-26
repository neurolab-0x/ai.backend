# Legacy Backend Boundary

## Status

`backend` is the legacy Neurolab monolith.

It is retained for:

- compatibility with existing integrations
- still-unsplit features such as EEG analysis, realtime inference, chat, voice, and reporting
- staged migration support while the split platform stabilizes

It is not the source of truth for the new platform.

## Source Of Truth

Use these services for new platform work:

- preprocessing: [`../preprocessor`](</home/polo/Documents/Neurolab/AI Service/preprocessor>)
- training: [`../training_system`](</home/polo/Documents/Neurolab/AI Service/training_system>)
- model serving: [`../model_platform`](</home/polo/Documents/Neurolab/AI Service/model_platform>)
- operator UI: [`../test_frontend`](</home/polo/Documents/Neurolab/AI Service/test_frontend>)

## Rules

- do not add new preprocessing pipeline logic to `backend`
- do not add new training-oriented preprocessing helpers to `backend`
- do not add new dataset registry or object-storage publication logic to `backend`
- do not add new training queue, MLflow, or model-promotion logic to `backend`
- do not treat `backend` as the active serving contract for promoted split-platform models
- do not restore legacy `/training` or `/models` backend APIs

## Remaining Ownership

Until they are extracted or retired, `backend` still owns:

- EEG analysis endpoints
- realtime/streaming inference endpoints
- chat flows
- voice analysis flows
- compatibility endpoints already consumed by older clients
- legacy report and persistence workflows

## Replacement Mapping

- backend training APIs → [`../training_system`](</home/polo/Documents/Neurolab/AI Service/training_system>)
- backend model-management APIs → [`../model_platform`](</home/polo/Documents/Neurolab/AI Service/model_platform>)
- backend preprocessing publication/training prep → [`../preprocessor`](</home/polo/Documents/Neurolab/AI Service/preprocessor>)

## Expected Direction

The target direction is contraction, not expansion:

```txt
legacy backend
→ fewer compatibility concerns over time
→ eventual retirement or much smaller gateway role
```

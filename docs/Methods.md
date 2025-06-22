[MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/model_maker)
MediaPipe Model Maker is a tool for customizing existing machine learning (ML) models to work with your data and applications. You can use this tool as a faster alternative to building and training a new ML model. Model Maker uses an ML training technique called transfer learning which retrains existing models with new data. This technique re-uses a significant portion of the existing model logic, which means training takes less time than training a new model, and can be done with less data.

----------------------------------------------------

[MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/tasks)

MediaPipe Tasks provides the core programming interface of the MediaPipe Solutions suite, including a set of libraries for deploying innovative ML solutions onto devices with a minimum of code. It supports multiple platforms, including Android, Web / JavaScript, Python

$ python -m pip install mediapipe

Python
The MediaPipe Tasks Python API has a few main modules for solutions that perform ML tasks in major domains, including vision, natural language, and audio. The following shows you the install command and a list of imports you can add to your Python development project to enable these APIs:

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
from mediapipe.tasks.python import audio

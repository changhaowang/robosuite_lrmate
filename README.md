# robosuite v1.3 with LRmate

![gallery of_environments](docs/images/gallery.png)

-------
## Latest Updates
[01/17/2023] **MSC Version**: Add Operational Space Admittance Controller

[01/13/2023] **MSC Version**: Add FANUC LRmate support for MSC Lab

[10/19/2021] **v1.3**: Ray tracing and physically based rendering tools :sparkles: and access to additional vision modalities 🎥

[02/17/2021] **v1.2**: Added observable sensor models :eyes: and dynamics randomization :game_die:

[12/17/2020] **v1.1**: Refactored infrastructure and standardized model classes for much easier environment prototyping :wrench:

-------
## Installation

1. Download MuJoco 2.1.0 or 2.0.0

2. Install mujoco_py from source (Notice robosuite v1.4 does not works with openai mujoco_py)

3. Install robosuite from source: 
  ```pip install -r reqruiments.txt```

4. Demo: ```python robosuite/demos/demo_control.py```
# LEP-01: Model Blocks

## Abstract

This LEP outlines the use and implementation of model blocks in PyLCM. Currently, PyLCM
supports the definition of a single model through a `Model` class. If state-action
combinations are invalid (e.g., working after having retired if the latter is an
absorbing state), the user can mask these using constraints.

This approach has two main drawbacks: First, internal utility functions must still be
evaluated on the full state-action space including invalid combinations. This implies
that potentially many unnecessary computations are performed. Second, even if the
state-action space can be split into blocks that constitute semantically sensible models
on their own (e.g., a model for working life and model during retirement), the user must
still define a single model that covers all blocks.

Model blocks aim to address these issues by allowing the user to define multiple models
(blocks) that can be combined into a single model.

## Roadmap

1. Implement a basic version of model blocks using the existing infrastructure to
   solve the example model (see below).
2. Design a user interface for model blocks.
3. Develop a full implementation of the
   - internal handling of model blocks.
   - user interface.

## Example Model

We consider a model of agents aged 50-80 years. They:

- have to work until they are 59,
- can choose whether they want to work or retire for ages 60-69 (retirement is an
  absorbing state),
- must retire upon reaching age 70 if they have not retired before.

This model can be constructed using two model blocks, a "working" model block and a
"retirement" model block. The choice variable "work status" determines which model
block is active. Choices are constrained by the state variable "lagged work status".

Consequently:

- only the working model block is active until the agents are 59,
- both model blocks are active between ages 60 and 69,
- only the retirement model block is active once agents have reached age 70.

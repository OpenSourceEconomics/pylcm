# LEP-01: Model Blocks

> [!NOTE]
> This LEP is an active document and will change over time.

## Abstract

This LEP outlines the use and implementation of model blocks in PyLCM. Currently, PyLCM
supports the definition of a single model through a `Model` class. If state-action
combinations are invalid (e.g., working after retirement), the user can mask these as
invalid using constraints. However, this approach has two main drawbacks: internal
utility functions must still be evaluated on the full state-action space, including
invalid combinations, leading to unnecessary computations, and even if the state-action
space can be split into blocks that constitute semantically sensible models (e.g., a
working model and a retirement model), the user must still define a single model that
covers all blocks. Model blocks aim to address these issues by allowing the user to
define multiple models (blocks) that can be combined into a single model.

## Roadmap

1. Implement a basic version of model blocks using the existing infrastructure to
   solve the example model (see below).
2. Design a user interface for model blocks.
3. Develop a full implementation of the
   - internal handling of model blocks.
   - user interface.

## Example Model

We consider a model where agents

- have to work until they are 60,
- can choose whether they want to work or retire until they are 70,
- must retire after they are 70.

This model can be constructed using two model blocks, a "working" model block and a
"retirement" model block. The working model block is active until the agents are 60, the
retirement model block is active after the agents are 70, and between ages 61 and 70
both model blocks are active simultaneously, giving agents the option to either continue
working or retire.

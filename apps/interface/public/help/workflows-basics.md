# Workflows — Basics

Workflows sit between ad-hoc runs and full automation: they capture how you generated something so you can rerun, tweak, or share it.

## What is a workflow?

- A workflow is a saved graph built from one or more model tabs.
- Each node represents a step (for example, txt2img, img2img, video generation, upscaling).
- Edges describe how results flow between nodes.

You can think of a workflow as a named, reusable recipe that remembers all the important knobs.

## Creating workflows from tabs

From a model tab:

- Configure the tab as usual (engine, sampler, scheduler, steps, resolution, extras).
- When you reach a useful configuration, use **Send to Workflows** from the tab header.
- This creates or updates a workflow entry based on the tab’s state.

From there, you can:

- Open the **Workflows** view (`/workflows`) from the top navigation.
- Inspect the graph and parameters captured for each node.
- Rename, duplicate, or remove workflows as they evolve.

## Running workflows

In the **Workflows** view:

- Select a workflow from the list.
- Inspect its nodes and parameters; update any fields that should change (for example, prompts or seeds).
- Start a run; the UI will execute the nodes in the right order and show progress/results.

Workflows are meant to be stable baselines:

- Use tabs when you are still exploring.
- Promote only the setups you trust into workflows.

## Where to look next

- For a higher-level description of how model tabs and workflows interact, see `apps/interface/public/help/home-overview.md`.
- Implementation and API details live under `.sangoi/design/flows/model-workflows.md` and `.sangoi/frontend/guidelines/frontend-architecture-guide.md`.


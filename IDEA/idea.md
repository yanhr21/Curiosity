# SUGAR-Centered CarryBox Research Direction

## Current Mainline

This workspace now treats official SUGAR CarryBox as the only active research
mainline.

SUGAR is the closest public baseline because it already combines the ingredients
this project needs most:

- human-video-driven humanoid loco-manipulation;
- IsaacLab manager-based simulation;
- G1-style humanoid embodiment;
- CarryBox-like object interaction;
- released official data, descriptions, and checkpoints.

The immediate goal is not to replace SUGAR. The immediate goal is to reproduce
official SUGAR faithfully, then make carefully isolated changes on top of that
baseline.

## Working Title

SUGAR-based video-guided humanoid CarryBox adaptation for unknown-load carrying.

## First Claim Gate

No new Curiosity claim is allowed until the official SUGAR CarryBox baseline is
reproduced with official code, assets, checkpoints, and training stage order.

Required baseline artifacts:

- official CarryBox inference video from official tracker/generator
  checkpoints;
- full official refiner checkpoint;
- official refiner rollout and processed rollout data;
- official tracker training output;
- official tracker rollout and processed tracker data;
- official generator training output;
- audit log showing which artifacts are present and missing.

## Research Changes After Reproduction

Once the faithful SUGAR reproduction is complete, modifications must be made on
top of SUGAR rather than in separate local scaffolds.

Allowed directions:

- add unknown-load and object-geometry randomization inside SUGAR CarryBox;
- add active probing observations/actions/rewards inside the SUGAR task;
- add embodiment-aware support/contact features for G1-style carrying inside
  the SUGAR environment;
- evaluate whether video-conditioned SUGAR policies improve over no-video or
  weakened-video ablations;
- evaluate whether probing improves over video-only SUGAR policies.

Forbidden active directions:

- reviving the old tactile-only path as the mainline;
- continuing old AGILE/G1 scalar-tuning branches as the mainline;
- using MuJoCo or prismatic proxy tasks as success evidence;
- replacing SUGAR with toy policies, toy refiners, or simplified local
  controllers.

## Success Standard

A valid result must beat the faithfully reproduced SUGAR CarryBox baseline on
harder held-out carrying conditions without increasing falls, drops, contact
loss, object instability, or rollout hacks. The comparison must make clear what
changed in SUGAR and what remained official.

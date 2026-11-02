"""sidewalkbench.gs: standalone GS (gaussian-splat) route-evaluation module.

A from-scratch reimplementation of the original research GS evaluation, structured
like the procedural-generation SidewalkBench env (terrain type, scene collision
mesh, sampler, parallel envs, one shared gsplat). coco-gs (cocobench-3dgs) is the
behavioral reference for metrics/checks/rendering; the routes and walkable regions
come from the research ``polygon_annotation`` format.

It uses the PG release's robot primitive, model zoo (sidewalkbench/models) and visualization utilities;
it is otherwise independent.
"""

# GreenSched instructions

- GreenSched is an hour-stepped, trace-driven simulator, not SimPy.
- Never hardcode printed result claims; every printed number must be computed.
- All randomness must come from explicit seed arguments.
- Use USD, gCO2eq, kW, and hours as the project units.
- Make one commit-sized change at a time, run the code after each change, and add or extend tests in `tests/` for every behavior change.
- Import modules under `src/` as `jobs.job`, `simulator.simulator`, and `scheduler.policies`.
- Documentation must identify which data is real and which is synthetic.

# Latest Source Remote Refresh

Date: 2026-07-01

This is a lightweight web/search and `git ls-remote` source refresh. It did
not update repositories, run code, install dependencies, run simulation,
rendering, validation builders, training, evaluation, or Slurm allocation.

Machine-readable status:
`experiments/configs/phase00/ref_tactile/latest_source_remote_refresh_v1.json`

## Key Change

Newton upstream `main` moved:

- remote `newton-physics/newton` main:
  `d58e70266be0db803261f3e46a2f7d923a43db37`
- current active evidence worktree `external/newton_main`:
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`
- stable tag `v1.3.0`:
  `ce11136b3a28390944f7fe5a32801b31d8aa5670`

Do not continue calling `a217e55...` the latest upstream main. It remains the
current active evidence base until a fresh latest-main Newton update and
compute-side sanity are performed.

## Local Matches Remote HEAD

- Taccel:
  `cb23bc251b531ba6908a3788c2f91423cd543149`
- UniVTAC:
  `05bcd3edb92237107efa40105292a24f1a9fd761`
- TaCauchy:
  `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
- HydroShear:
  `a53a51cb74f0608ca53839415d7f1964a99f1db0`
- FreeTacMan:
  `9285740a5d33385d3a9cf5ccdb185e3387b547bd`
- DiffTactile:
  `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
- APPLE:
  `4b1d71fadb786d865d4ee29a184ab408b9605083`
- Tactile MNIST:
  `9e4e59139e9349ab361a3b9297f4815724ad6387`
- Reactive Diffusion Policy:
  `824c5e8de1fd1811106907a04b5f0186e0138c0b`
- ImplicitRDP:
  `4c90646df17787e31c88838106c4a0323ddefb4a`
- Tactile Diffusion:
  `16868fb96d19d93dc5837600c26b48415632e4f6`

## Reference Tags

IsaacLab tags required by the env plan are available:

- `v2.1.1`: `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- `v2.2.1`: `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20`
- `main`: `b4c321024792976150ca55fddb26fa34480d974e`

## Gaps

- T-Rex remote main is
  `43ff632259d76f08373c085c53111825060d029b`, while local
  `external/T-Rex` is at `db7a02992504ad9be53a7e764f7b05d81d86c767` with
  existing dirty state that must not be overwritten silently.
- `https://github.com/yanglh14/IsaacLabTactile.git` returned repository not
  found in this probe. Keep IsaacLab tactile as a reference gap unless a valid
  official repository URL is confirmed.

## Next Action

Prepare a fresh Newton latest-main update/sanity path before any future claim
that the active base is on the latest upstream Newton. Gate 00F remains blocked
by missing target reference environments and incomplete TaCauchy assets.

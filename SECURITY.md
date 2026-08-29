# Security Policy

cuda-engineering is a set of educational docs and small, self-contained CUDA
sample programs. There's no running service, no server component, and no
network calls in any sample - so the realistic attack surface is narrow.

## What belongs in a report

Worth reporting privately:

- A sample program that does something unsafe with untrusted input (none are
  designed to take untrusted input, so this would itself be a bug).
- A build script or Makefile that fetches and executes something from the
  network without saying so.

Not a security issue, just a normal bug report (open a public issue instead):

- A kernel that compiles but computes the wrong answer.
- A memory bug (out-of-bounds access, race) inside a sample that only ever
  operates on data the sample itself generates.
- A dead or incorrect link in the curated resources.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:

> https://github.com/Laaaaksh/cuda-engineering/security/advisories/new

That reaches the maintainer privately so any real issue can be fixed before
it's discussed in public.

## Credits

Reporters who wish to be credited may say so in the private report; otherwise
reports are handled without attribution.

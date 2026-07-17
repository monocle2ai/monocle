# Security and Disclosure Policy for the Monocle Project

The Monocle community takes the security of our software seriously, and we
appreciate every effort to responsibly disclose vulnerabilities. This document
describes which versions are supported, the security practices we follow, and
how to report a vulnerability.

## Supported Versions

Monocle is distributed as three independently versioned packages on PyPI —
`monocle_apptrace`, `monocle_test_tools`, and `monocle_mcp`. Security fixes are
applied to the latest released version of each package. Please upgrade to the
latest stable release, which will have all known security issues addressed. We
generally cannot backport fixes to older releases.

## Security Practices

To keep Monocle and its users safe, the project follows a number of practices:

- **OpenSSF Best Practices.** Monocle participates in the
  [OpenSSF Best Practices Badge Program](https://www.bestpractices.dev/projects/9176).
- **Transport security.** Network communication performed by Monocle exporters
  and integrations (e.g. Okahu, OTLP, cloud storage backends) uses HTTPS/TLS.
- **Open, auditable releases.** Monocle is licensed under Apache-2.0 and
  published to [PyPI](https://pypi.org/project/monocle-apptrace/). Source is
  developed in the open at
  [github.com/monocle2ai/monocle](https://github.com/monocle2ai/monocle).
- **Dependency and version pinning.** Framework integrations pin tested
  dependency versions so that builds and tests are reproducible.

## Analysis and Quality Tooling

Monocle's continuous integration runs static analysis and tests on every
change:

- **Pylint** for static analysis of the Python source
  (`apptrace/src`, `test_tools/src`, `mcp/src`).
- **Pytest** for unit and integration test suites across all packages.
- **GitHub** for source control, code review, and issue tracking.

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Monocle, please
**do not** report it publicly through the GitHub issue tracker, Discussions,
Discord, or any other public forum.

Instead, report it privately using one of the following methods:

1. **Preferred — GitHub Private Vulnerability Reporting.** Open a private report
   from the **Security** tab of the
   [Monocle repository](https://github.com/monocle2ai/monocle/security/advisories/new).
   This keeps the report confidential and links it directly to the maintainers.

2. **Alternative — Contact the maintainers privately.** Reach out to the Monocle
   [maintainer team](https://github.com/monocle2ai/monocle/blob/main/MAINTAINER.md).

Please include as much of the following as you can, so we can reproduce and
triage the issue quickly:

- A description of the vulnerability and its potential impact.
- The affected package(s) and version(s).
- Step-by-step instructions to reproduce, including a minimal proof of concept
  where possible.
- Any suggested remediation.

## What to Expect

- We will acknowledge your report and begin analysis within **3 working days**.
- We will keep the details of the report confidential, sharing them only as
  necessary to develop and validate a fix.
- We will keep you informed of our progress toward a resolution and may ask for
  additional information.
- Once a fix is available, we will coordinate disclosure and credit reporters
  who wish to be acknowledged.

## Security Announcements

Security advisories and fixes are published through
[GitHub Security Advisories](https://github.com/monocle2ai/monocle/security/advisories)
and announced in the project's community channels, including the
[Monocle Discord](https://discord.gg/D8vDbSUhJX).

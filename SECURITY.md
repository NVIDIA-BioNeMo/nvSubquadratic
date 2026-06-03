# Security Policy

## Reporting a Vulnerability

NVIDIA is dedicated to ensuring the security of its products and services. If
you believe you have found a security vulnerability in **nvSubquadratic**,
please report it using the procedures below. **Do not** open a GitHub issue
or pull request for security-sensitive findings — those channels are public.

### Preferred reporting channels

1. **NVIDIA Product Security Form** —
   <https://www.nvidia.com/object/submit-security-vulnerability.html>
1. **Email the NVIDIA PSIRT team** at <psirt@nvidia.com>

When reporting, please include as much of the following as you can:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, including the affected version (commit SHA / release tag)
  and environment (CUDA, PyTorch, GPU architecture).
- Any proof-of-concept code, logs, or screenshots that help triage.
- Your name and how you'd like to be credited (or to remain anonymous).

PSIRT will acknowledge receipt and coordinate the disclosure timeline with you.

## Scope

This policy covers the code and assets in this repository
(`nvsubquadratic/`, `experiments/`, `tests/`, `scripts/`, `benchmarks/`,
`docs/`, and the build artifacts produced from them). Vulnerabilities in
upstream dependencies (PyTorch, CUDA toolkit, etc.) should be reported to
their respective maintainers; if a defect in this repository's use of a
dependency creates a security issue, that is in scope.

## Further information

- NVIDIA Product Security overview: <https://www.nvidia.com/en-us/security/>
- NVIDIA Security Bulletins: <https://www.nvidia.com/en-us/security/security-bulletins/>

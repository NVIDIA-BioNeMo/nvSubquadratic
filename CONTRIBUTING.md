# Contribution rules

Thank you for your interest in contributing to nvSubquadratic. Here are guidelines for contributing:

## Reporting Bugs

1. Check if the bug has already been reported in the Issues section
1. If not, create a new issue with:
   - A clear and descriptive title
   - Steps to reproduce the bug
   - Expected behavior vs actual behavior
   - Your environment details (OS, Python version, package versions, GPU + CUDA version)
   - Any relevant error messages or logs

## Making Contributions

1. Fork the repository and create a new branch from `main`

1. Make your changes following our code style:

   - Add comments only where the *why* is non-obvious
   - Follow existing naming conventions
   - Add tests for new functionality

1. Before submitting a PR:

   - Run all tests locally
   - Update documentation if needed
   - Ensure pre-commit hooks pass (`pre-commit run --all-files`)
   - Add a clear commit message describing your changes
   - Sign off your commits per the Developer Certificate of Origin (see below)

1. Submit a Pull Request:

   - Reference any related issues
   - Describe what the PR does and why
   - List any breaking changes
   - Include before/after examples if relevant

1. The PR will be reviewed by maintainers who may request changes

## Code Review Process

- All submissions require review
- Maintainers will review PRs regularly
- Address review feedback promptly
- PRs must pass all automated checks (CI, license-header check, pre-commit hooks)

For questions, feel free to open a discussion in the repository.

## Developer Certificate of Origin

By contributing to nvSubquadratic, you agree to the Developer Certificate of Origin (DCO). The DCO is a lightweight way to certify that you wrote, or otherwise have the right to submit, the code you are contributing to the project. Here is the full text of the DCO, reformatted for readability:

> Developer Certificate of Origin
> Version 1.1
>
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
>
> Everyone is permitted to copy and distribute verbatim copies of this
> license document, but changing it is not allowed.
>
> Developer's Certificate of Origin 1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I
> have the right to submit it under the open source license
> indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best
> of my knowledge, is covered under an appropriate open source
> license and I have the right under that license to submit that
> work with modifications, whether created in whole or in part
> by me, under the same open source license (unless I am
> permitted to submit under a different license), as indicated
> in the file; or
>
> (c) The contribution was provided directly to me by some other
> person who certified (a), (b) or (c) and I have not modified
> it.
>
> (d) I understand and agree that this project and the contribution
> are public and that a record of the contribution (including all
> personal information I submit with it, including my sign-off) is
> maintained indefinitely and may be redistributed consistent with
> this project or the open source license(s) involved.

Contributors sign-off that they adhere to these requirements by adding a `Signed-off-by` line to commit messages:

```
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

Git even has a `-s` command line option to append this automatically to your commit message:

```
$ git commit -s -m 'This is my commit message'
```

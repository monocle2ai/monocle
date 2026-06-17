
# Project Monocle  Governance
This document defines the governance structure and technical decision-making processes for the [Project Name] project.The project operates transparently, openly, and collaboratively, governed by the community. Participation is open to all individuals and organizations, provided they agree to abide by the guidelines in this document.

## Code of Conduct 
Monocle project adheres to [Code of conduct](CODE_OF_CONDUCT.md)

## Project roles

### Contributor
A contributor is anyone participating in the project who submits code, documentation, or other valuable assistance including feedback on PRs from other contributors and [committers](#committers-maintainer). 

### Committers (Maintainer)
A commiter has write access to the Monocle code base, also permissions to approve and merge pull requests.
#### Responsibilities
- Contribute code and process improvements to the project
- Review and merge PRs from contributors and other committers
- Promote the project on social media, meetups, conferences etc

### Technical Steering Committee (TSC)
The TSC is the governing body responsible for the technical direction, architecture, releases, and community policies of Monocle.
#### Composition
- The TSC consists of elected or appointed active committers of the community

#### Responsibilities
- Approving overall project architecture and design.
- Artifact [release process](./RELEASE.md) (eg Python library release on Pypi.org)
- Resolving technical disputes among contributors.
- Ensuring alignment with Foundation (e.g., LF Energy, LF AI & Data) mandates.
- Promote the project on social media, meetups, conferences etc

## Decision-Making Process
The project relies heavily on consensus-based decision-making for its day-to-day operations.

### Pull Requests
Require at least one approving review from an active Committer before merging.

### Releases
- Require at least one committer or TSC memeber to initiate the release by creating a ticket
- Require at least one approval from an active TSC memeber

### Architectural/Policy Decisions
If a significant technical dispute or policy change occurs, it is elevated to the TSC.

### Onboarding new committers/maintainer
- At least one active TSC member to propose onboarding new committer/maintainer
- An ideal candidate is someone who has made meaningful contributions to the project (patches, features, documentation)
- A majority vote from active TSC members to approve the new committers/maintainer

### Onboarding new TSC member
- At least one active TSC member to propose onboarding new TSC member
- The candidate should be an active maintainer
- An ideal candidate is someone who has helped project to grow beyond just code contributions, eg reviews, promoting project via social media, conferences, meetup etc.
- A majority vote from active TSC members to approve the new TSC member

### Voting
When consensus cannot be reached, the TSC will vote on the matter. TSC decisions require a majority vote of all TSC members

### Amendments
This governance document may be amended by a two-thirds (2/3) majority vote of the TSC

## Current commetters/maintainers and TSC members
Please refer to [CODEOWNERS](CODEOWNERS.md) for list of active commetters/maintainers and TSC members
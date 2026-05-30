# Contributing to Sigma

Sigma is distributed under the **Sigma License** (see `LICENSE.txt`).
Because the License reserves ArsChitectura SAS's unilateral right to revoke
and to relicense future versions, inbound contributions must be structured
so that ArsChitectura SAS can continue to do so unambiguously.

**External pull requests are not accepted until the contributor has signed
the Contributor License Agreement (CLA) below.** This is not negotiable.

## Contributor License Agreement (CLA)

By submitting any contribution (code, documentation, test, example, asset,
or any other material) to this repository, You agree, as between You and
ArsChitectura SAS:

1. **Economic-rights assignment.** You assign to ArsChitectura SAS, to the
   fullest extent permitted by applicable law, all economic rights
   (*droits patrimoniaux* under CPI art. L122-1 and L122-6; economic
   copyright under 17 U.S.C. and analogous regimes) in Your contribution,
   including the rights of reproduction, representation, adaptation,
   translation, distribution, and communication to the public, for the
   full duration of such rights and for every form of exploitation known
   or hereafter devised, worldwide.

2. **Fallback exclusive license.** To the extent the foregoing assignment
   is not effective under applicable law, You grant ArsChitectura SAS a
   perpetual, irrevocable, worldwide, royalty-free, fully paid-up,
   exclusive, transferable, sublicensable license to use, reproduce,
   modify, adapt, translate, publish, distribute, and otherwise exploit
   Your contribution, with the right to sublicense through multiple
   tiers.

3. **Moral-rights accommodation.** You acknowledge that moral rights
   (*droits moraux* under CPI art. L121; Urheberpersonlichkeitsrechte
   under UrhG §§ 12-14; analogous regimes) are inalienable in France and
   Germany and cannot be assigned. You agree, to the maximum extent
   permitted by applicable law and subject to CPI art. L121-7 (which
   significantly restricts moral rights for software): (i) not to
   exercise the droit de retrait et repentir in a manner that would
   prevent ArsChitectura SAS from continuing to distribute, modify, or
   revoke the Software; (ii) to waive exercise of the droit de paternite
   beyond the attribution ArsChitectura SAS elects to provide in
   `AUTHORS` / `CREDITS` / `CONTRIBUTORS` files or equivalent; and (iii)
   not to invoke droit au respect against routine modifications,
   refactorings, ports, or commercial adaptations undertaken by
   ArsChitectura SAS or its licensees.

4. **Rights to contribute.** You represent that Your contribution is Your
   original work, or that You have the right to submit it under these
   terms. You further represent that Your employer, if applicable, has
   agreed in writing that Your contribution may be made under these
   terms, or that no such agreement is required by operation of law.

5. **No warranty.** Your contribution is provided "as is", without
   warranty of any kind. ArsChitectura SAS is not obligated to accept,
   merge, maintain, or publish Your contribution.

6. **Sigma License.** If ArsChitectura SAS incorporates Your
   contribution, ArsChitectura SAS may distribute it under the Sigma
   License as it then exists, or under any successor,
   alternative, proprietary, or commercial license ArsChitectura SAS
   may adopt at its sole discretion, without further consent from You
   and without obligation to distribute at all.

7. **Governing law and forum.** This CLA is governed by French law. Any
   dispute is subject to the exclusive jurisdiction of the courts of
   Paris, France.

## How to sign

Open a pull request adding Your name and email to `AUTHORS.md` (or create
it if it does not yet exist) with a commit message:

```
CLA: <Your full name> <email@example.com>

I have read and agree to the CLA in CONTRIBUTING.md.
Signed-off-by: <Your full name> <email@example.com>
```

The `Signed-off-by` line is a Developer Certificate of Origin attestation
that You have read and agreed to the CLA in full.

## Small contributions

For trivial fixes (typos, formatting, few-line bugfixes), the CLA still
applies, but the `Signed-off-by` line in the commit message is considered
sufficient evidence of agreement.

## Permitted-Contribution Lifecycle

Section 2(b) of `LICENSE.txt` grants You a narrow right to modify a
working copy of the Software solely as preparation for a Contribution.
This section documents the lifecycle that Section 2(b) refers to.

### What counts as a Contribution

A Contribution is either:

1. a Pull Request opened against the Canonical Repository
   (https://github.com/arschitectura/sigma), conforming to this document;
   or
2. a patch You submit to ArsChitectura SAS through any other disclosed
   Licensor-facing channel (an issue comment with a patch, a CLA-signed
   email diff, or any other disclosed channel). Such a patch is treated
   as a Contribution from the moment of submission; to become a **Merged
   Contribution** it must be advanced to a Pull Request opened against
   the Canonical Repository.

### Status definitions

- **Pending Contribution.** A Contribution that has been submitted and
  is awaiting review, under active review, or under iteration. While
  Pending, You may run the modified working copy locally for testing,
  debugging, and iteration on the Contribution itself, and in a private
  staging environment for validation against realistic data, subject to
  the conditions in `LICENSE.txt` Section 2(b)(i). The Pending status
  endures only while the Contribution remains under active review; see
  *Abandonment* below.
- **Merged Contribution.** A Pull Request that has been merged into the
  active development branch of the Canonical Repository. The merged
  code becomes part of the Software in the next release and is from
  then on governed by the Sigma License like any other part of the
  Software.
- **Rejected Contribution.** A Contribution that has been (i) closed
  without merge, (ii) withdrawn by You, (iii) declined by
  ArsChitectura SAS, (iv) abandoned under the criteria in *Abandonment*
  below, or (v) otherwise terminated without becoming a Merged
  Contribution.

### Abandonment

A Pending Contribution becomes Abandoned (and thereby Rejected) on the
earlier of:

- thirty (30) days of contributor inactivity following a reviewer
  request for changes, unless ArsChitectura SAS specifies a longer
  period in the review; or
- ninety (90) days of contributor inactivity following the most recent
  visible activity from You on the Pull Request.

A patch submitted through a channel other than a Pull Request (issue
patch, emailed diff) is Abandoned if it is not advanced to a Pull
Request within thirty (30) days of its submission, unless
ArsChitectura SAS specifies a longer period.

### Obligations upon Rejection

Upon a Contribution becoming Rejected (by close, withdrawal, decline,
or Abandonment), You shall, within thirty (30) days from the date of
Rejection (or such longer period as Section 13 of `LICENSE.txt` allows
for established Licensees):

1. cease all use of the modified working copy and of any work derived
   from it;
2. destroy every copy of the modified working copy and every Derivative
   Work derived from it in Your possession or control, subject to the
   non-waivable backup carve-out of Section 11 of `LICENSE.txt`; and
3. not derive any further work from the modified working copy.

From the effective date of Rejection, the modified working copy is
treated as an unauthorized Derivative Work under `LICENSE.txt`
Section 7.2 and Section 7.9.

### Merged-then-reverted

If a Merged Contribution is later reverted upstream by ArsChitectura
SAS, the obligations of *Obligations upon Rejection* apply mutatis
mutandis to the reverted code, with the date of the revert substituted
for the date of Rejection.

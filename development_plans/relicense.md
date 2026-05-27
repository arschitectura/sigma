# Relicense: modifications authorized only as upstream contributions

## Objective

Remove the general right to modify Sigma. Permit modification of a
working copy only as a step toward a Contribution to the Canonical
Repository. A Licensee's right to use a modified version is
co-terminous with the corresponding Contribution being merged
upstream: if it is not merged, the Licensee has no usage right over
the modification, no Derivative Work right, no Distribution right,
and no Access Grant right.

## Background

`LICENSE.txt` Section 2(b) currently grants the Licensee a broad right
to "modify the Software and create Derivative Works", and Section 2(c)
extends Distribution and Access Grant rights to those Derivative
Works. Combined with Section 7.7 (Permitted AI Integration Use, which
covers AI-assisted authorship of new client code calling Sigma's
public API), this means a Licensee can today fork the Software
internally, patch it, ship the patched version to its own
infrastructure, and never contribute the patch back.

That outcome contradicts the project's positioning as a single
transparent, source-available canonical codebase whose users are
encouraged to read and share it but not to diverge from it. The
broad modification grant is also the practical mechanism by which the
codebase would fragment if Sigma gained traction outside the current
private-beta circle.

## The change

Replace Section 2(b) with a Permitted-Contribution clause and
correspondingly narrow Section 2(c).

### Wording sketch (to be polished by legal)

> 2(b) Licensee may modify a working copy of the Software solely for
> the purpose of preparing a Contribution and submitting it through
> the procedure in CONTRIBUTING.md. Any other modification, in any
> form, is an unauthorized Derivative Work and is prohibited under
> Section 7.2.
>
> 2(c) Licensee may Distribute and provide Access Grants of the
> Software. Distribution and Access Grant of a Derivative Work is not
> authorized.

### Lifecycle of a permitted modification

- **Pending Contribution** (submitted, under review). The Licensee may
  execute the modified code locally for the sole purpose of testing,
  debugging, and iterating on the Contribution itself. The modified
  code is not Software for any other purpose: it may not be deployed,
  run in production, used to produce work product for third parties,
  Distributed, or Access-Granted.
- **Merged Contribution** (accepted into the Canonical Repository).
  The contributed code becomes part of the Software in the next
  release and is from then on governed by this License like any other
  part of the Software. The Licensee may use it as Software, without
  any residual obligation specific to its contribution origin.
- **Rejected, withdrawn, abandoned, or never-merged Contribution.**
  The Licensee must (i) cease all use of the modification, (ii)
  destroy any copy and any Derivative Work derived from it, and (iii)
  not derive any further work from it. From that point on, the
  modification is treated for all purposes as an unauthorized
  Derivative Work and falls fully under Section 7.2.

### Interaction with existing sections

- **Section 1 (Definitions).** Add `Contribution`, `Pending
  Contribution`, `Merged Contribution`, `Rejected Contribution`. Refer
  to `CONTRIBUTING.md` as the procedural document.
- **Section 3 (Stipulation pour autrui).** Unchanged. Downstream
  Recipients still receive a direct license from Licensor on the
  Software; since Derivative Works are no longer Distributable, there
  is no downstream chain to bind for them.
- **Section 7 (Prohibited Uses).** Add a clarifying sub-section that
  any modification not tied to a Pending or Merged Contribution falls
  under the Derivative-Work prohibition of 7.2, and that the
  Permitted-Contribution clause does not authorize the activities
  prohibited by 7.1, 7.3, 7.4, or 7.5.
- **Section 7.7 (Permitted AI Integration Use).** Unchanged. The
  carve-out covers AI-assisted authorship of new Licensee-owned
  client code that calls Sigma's public API; it has never covered
  modifications to Sigma itself.
- **Co-Terminous Clause (end of Section 2).** Becomes largely
  vestigial - the underlying right to create a Derivative Work is now
  narrow enough that there is very little for it to attach to. It can
  be kept as a belt-and-suspenders measure or folded into the new
  Permitted-Contribution lifecycle wording.
- **Section 13 (Effect of Revocation).** Confirm that the 30-day
  stop-use window applies symmetrically to the contribution-lifecycle
  cease-use trigger (Rejected / never-merged Contribution).

## Knock-on changes outside LICENSE.txt

1. **CONTRIBUTING.md.** Restate that submitting a Contribution is the
   only authorized basis for modifying the Software. Document the
   merge / rejection lifecycle, including: how a Contribution is
   considered withdrawn or abandoned (e.g., closed PR, X days of
   inactivity), what "merged" means precisely (merged into `main`
   versus included in a tagged release), and the contributor's
   obligations on rejection.
2. **NOTICE.txt and the short-form Notice to Recipients (Schedule 2
   of LICENSE.txt).** Update any one-line plain-language summary that
   mentions modification rights so it no longer reads as a general
   permission.
3. **README.md.** Search for and update any claim that Sigma "can be
   modified" or "is freely modifiable". The modification right is now
   conditional and procedural, not a free-standing grant.
4. **arschitectura.com Sigma product page FAQ.** The current FAQ in
   `websites/arschitectura.com/initial_resources/products/sigma/index.html`
   (in the Sabrina+website repo) reads "You can use, modify, and
   redistribute it, keeping the license file with every copy".
   Change to a wording that reflects the new shape, e.g., "You can
   use and redistribute it, keeping the license file with every copy;
   modifications are accepted only as upstream Contributions to the
   canonical repository."

## Open questions

1. **Internal staging during open review.** Should the Licensee
   retain a right to run the modification in a private staging
   environment for the duration of the open Contribution, e.g., to
   validate it against real production data before it is merged? If
   yes, what is the hard cap? A natural ceiling is the active-review
   window with a calendar limit (e.g., 90 days from PR open,
   extendable by visible review activity), then a forced cease-use.
2. **Merged-then-reverted upstream.** If a Contribution is merged and
   later reverted upstream, does the contributor's right revert with
   it? Default: yes, mirroring Section 13's 30-day stop-use window
   for already-deployed instances.
3. **Out-of-band contributions.** Is `Contribution` defined narrowly
   as a merged Pull Request against the Canonical Repository, or does
   it also cover patches accepted through other channels (e.g.,
   emailed diffs from an early commercial customer)? The narrow
   definition is simpler to police and to reason about; the broad
   definition is more friendly to the current private-beta workflow.
4. **Bug-report reproducers.** Bug reports often include a small
   patch that demonstrates the bug. Strictly read, the new clause
   makes that patch an unauthorized Derivative Work unless it becomes
   a Contribution. Either (a) treat such patches as Contributions by
   construction (any submitted patch is a Pending Contribution from
   the moment it is shared with Licensor), or (b) add an explicit
   safe-harbor for short demonstrative patches included in good-faith
   bug reports.
5. **Relationship with the paid non-revocable commercial license
   (Section 22).** A commercial licensee may legitimately need to
   maintain a private fork (e.g., for vendored-version stability).
   The relicense plan should be explicit that the commercial license
   may grant a broader modification right, and that the
   Permitted-Contribution clause is a property of the
   source-available license only.

## Verification when this lands

1. Read the resulting LICENSE.txt end-to-end and confirm no other
   clause silently re-grants what 2(b) removes - in particular,
   Sections 3, 5, 6, and 13 should still parse coherently.
2. Re-run `assemble.sh` to ensure any license-header check still
   passes.
3. Grep the README, CONTRIBUTING, NOTICE, and the website FAQ for
   stale claims that modification is freely permitted, and confirm
   each has been updated.
4. Confirm that the Sigma FAQ on arschitectura.com still has the
   Section 7.7 AI-integration carve-out wording (added separately;
   that clause is orthogonal to this change and must survive the
   FAQ rewrite).

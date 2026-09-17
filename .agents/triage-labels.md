# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to
the actual label strings used in this repo's issue tracker.

| Canonical role    | Label in our tracker | Meaning                                  |
| ----------------- | -------------------- | ---------------------------------------- |
| `needs-triage`    | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`      | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent` | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human` | `ready-for-human`    | Requires human implementation            |
| `wontfix`         | `won't fix`          | Will not be actioned                     |

Four of the five roles are spelled the same on both sides. `wontfix` is not: the label
string in `OpenSourceEconomics/pylcm` is `won't fix`, with an apostrophe and a space, so
`gh issue edit --add-label wontfix` fails.

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the
corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

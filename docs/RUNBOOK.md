# Runbook

The episode procedure is the skill **`.claude/skills/six-degrees-episode/SKILL.md`**. Follow it: it is the
single source of truth for producer sessions (branches + PRs, honesty rules, lineage links, QA).

## How the system runs
| When | Who | What |
|---|---|---|
| 1:13 AM CT | daily producer task → operator session | starts worker sessions, one PR per episode (paused while a backlog batch runs) |
| on each PR | `render.yml` (GitHub) | renders + QA as a check; nothing is queued |
| operator check-ins | operator session | merges episode PRs whose `render` check passed |
| on merge to main | `render.yml` | renders again and puts the video on the `queue` branch |
| 11:22 AM / 6:22 PM CT (or 6:52 AM) | cron-job.org → `upload.yml` | uploads the next queued video(s), thumbnail, first comment |
| on demand | `stats.yml` | views/likes/comments (+ retention with the analytics scope) into `stats.json` |

Re-render queued videos after a style change: `python -m sixdeg render <slug> && python -m sixdeg qa <slug> &&
python -m sixdeg kit <slug>`, then `python -m sixdeg requeue` (refuses anything without a QA pass).

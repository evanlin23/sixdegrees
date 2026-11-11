# Studio metrics: task for Claude in Chrome

YouTube's Analytics API lags 1-3 days and doesn't give the Shorts numbers that matter most ("viewed vs swiped away",
the retention curve, traffic sources, when viewers are online). Studio shows them. Paste everything below the line into
Claude in Chrome while signed in to the account that owns @buhconnects, then paste its JSON answer to the Buh Connects operator
session (or save it as a file and run `python -m sixdeg studio <file>`). Run it every 2-3 days. Adapted from the
HelloBuh channel's prompt (evanlin23/hello-buh, docs/studio-metrics-prompt.md).

---

Collect YouTube Studio analytics for my channel @buhconnects. Only read; don't change any setting, video, title or comment.
Stop and ask me if anything needs a password or a code. Make sure the channel switcher (top right) says "Buh Connects".

0. ALWAYS THE LATEST NUMBERS
   - Reload every page before reading it; never reuse numbers from an earlier run or an earlier tab.
   - Prefer the live sources: each Short's own Analytics page (Overview, Engagement, Reach, Audience) and its
     "Realtime" card, and the Content list's view/comment/like columns. YouTube's Advanced mode table lags 1-2 days:
     use it only for a number the live pages don't show, and say so in couldnt_read.
   - Period: whichever of "Lifetime" and "Since published" runs through today (Lifetime often stops yesterday).
   - When two places show different values for the same number, take the newer one (usually the larger).
   - For each Short, record "data_through": the last date the Studio page's numbers cover (Studio says e.g.
     "Nov 10 - Nov 11, 2026" or "Updated ... ago"); for realtime figures, the time you read them.

1. LIST THE SHORTS
   - Open https://studio.youtube.com → Content → Shorts tab. Note every public or scheduled Short: its video ID (the
     part after /video/ in its link), title, publish date and time.

2. FOR EACH PUBLIC SHORT, open https://studio.youtube.com/video/<ID>/analytics/tab-overview/period-lifetime
   (period: Lifetime). Read these, as shown (write null if Studio doesn't show one):
   - Overview tab: views, watch time (hours), subscribers gained, and the "realtime" views if shown.
   - Engagement tab (or the Shorts card on Overview):
     - "Viewed vs. swiped away": the viewed % (Shorts only; this matters most).
     - Average view duration (seconds) and average percentage viewed (can be over 100 %).
     - The audience-retention graph: hover it and read the % still watching at 0:01, 0:03, the middle and the last
       second. Also note any big dip or spike and at what second.
     - Likes, dislikes if shown, comments, shares, subscribers lost.
     - Comments: open the Short's comments (Studio → Comments, filter by this video; remove the default "Unresponded"
       view filter so answered comments show too) and copy up to 10 top comments as "@handle: text", word for word,
       leaving out the channel's own. The handle matters: requested pairs credit the viewer who asked.
   - Reach tab: "Traffic source types" with the % of views from each (Shorts feed, YouTube search, Channel pages,
     Browse features, External, Other), and "Shown in feed" if Studio shows it.
   - Audience tab: % new vs returning viewers, % subscribed vs not, top 3 countries with %, top age group.

3. CHANNEL (https://studio.youtube.com → Analytics; period: last 28 days, and again last 7 days)
   - Views, watch time, subscribers (net), and the views-per-day chart: read each day's views.
   - Content tab → "How viewers find your Shorts" (%s), "Viewed vs swiped away" for the channel.
   - Audience tab → "When your viewers are on YouTube": the 3 darkest (busiest) day+hour cells, with the time zone
     Studio shows. Also "Returning viewers", "Unique viewers", total subscribers.
   - Research tab (if there is one): the top 10 searches "Your viewers' searches" and any "Content gaps".

4. ANSWER with ONE JSON code block, nothing left out, numbers as plain numbers (12.5 not "12.5%"), times in seconds:
```json
{
  "collected_at": "YYYY-MM-DDTHH:MM local, with the time zone",
  "channel": {
    "subscribers_total": 0,
    "last_28d": {"views": 0, "watch_hours": 0, "subs_gained": 0, "subs_lost": 0, "subs_net": 0, "viewed_pct": 0, "likes": 0, "comments": 0, "shares": 0, "views_by_day": {"YYYY-MM-DD": 0}},
    "last_7d": {"views": 0, "watch_hours": 0, "subs_net": 0, "viewed_pct": 0},
    "traffic_pct": {"shorts_feed": 0, "search": 0, "channel_pages": 0, "browse": 0, "external": 0, "other": 0},
    "returning_viewers": 0, "unique_viewers": 0,
    "busiest_times": ["Tue 18:00 America/Los_Angeles"],
    "viewer_searches": ["..."], "content_gaps": ["..."]
  },
  "videos": [
    {
      "id": "VIDEO_ID", "title": "...", "published": "YYYY-MM-DDTHH:MM", "data_through": "YYYY-MM-DDTHH:MM",
      "views": 0, "watch_hours": 0, "subs_gained": 0,
      "viewed_pct": 0, "avg_view_s": 0, "avg_pct_viewed": 0,
      "retention_pct": {"1s": 0, "3s": 0, "mid": 0, "end": 0}, "retention_notes": "dip at 4s to 40%",
      "likes": 0, "comments": 0, "shares": 0, "subs_lost": 0,
      "top_comments": ["..."],
      "traffic_pct": {"shorts_feed": 0, "search": 0, "channel_pages": 0, "browse": 0, "external": 0, "other": 0},
      "shown_in_feed": null,
      "new_viewers_pct": 0, "subscribed_pct": 0, "top_countries": {"US": 0}, "top_age": "18-24"
    }
  ],
  "couldnt_read": ["anything you couldn't find, and where you looked"]
}
```

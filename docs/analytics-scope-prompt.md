Give my Six Degrees YouTube uploader read access to YouTube Analytics, so it can see retention ("stayed to watch",
average % viewed). Do the steps in order. The refresh token is copied from one page and pasted into another only.
Never type it into this chat, a GitHub issue, a commit or any other page. If anything doesn't match what I
describe, stop and tell me instead of guessing.

1. Google Cloud (console.cloud.google.com): open my "blocky-stories" project (the one whose OAuth client the Six
   Degrees uploader uses) → APIs & Services → Library → search "YouTube Analytics API" → Enable (if it already
   says "Manage", it's enabled; go on). Then APIs & Services → Credentials (or Google Auth Platform → Clients):
   open the existing Web OAuth client whose authorized redirect URI is https://developers.google.com/oauthplayground
   and keep the tab open for its Client ID and Client secret. Change nothing.

2. OAuth Playground (https://developers.google.com/oauthplayground):
   a. Gear icon → tick "Use your own OAuth credentials" → paste that Client ID and Client secret. Access type:
      Offline. Force prompt: Consent Screen.
   b. In "Input your own scopes", paste exactly (all four, one line):
      https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/yt-analytics.readonly
      Click "Authorize APIs".
   c. When Google asks which account or channel, pick the BUH CONNECTS / sixdegrees (@sixdegreesbetweenus)
      channel, not Buhtown. If you can't tell, stop and ask me. Click through "Google hasn't verified this app"
      (Advanced → continue); it's my own app. Make sure every permission checkbox is ticked.
   d. Click "Exchange authorization code for tokens". Keep the new Refresh token in this tab.
   e. Check, in Step 3: GET https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true must show the
      channel id UCE3KAucU-r7Jp3BRGbreQoQ. If it shows a different channel, start step 2 again.
   f. Check analytics works: GET
      https://youtubeanalytics.googleapis.com/v2/reports?ids=channel==MINE&startDate=2026-11-01&endDate=2026-11-30&metrics=views,averageViewPercentage
      must return 200 (rows may be empty or show small numbers).

3. GitHub repo evanlin23/sixdegrees → Settings → Secrets and variables → Actions → Secrets: open YT_REFRESH_TOKEN
   → Update → paste the new refresh token from step 2d → Update secret. Don't change YT_CLIENT_ID or
   YT_CLIENT_SECRET.

4. While you're in GitHub: Actions tab → "stats" workflow → Run workflow (branch main). Wait for it to finish and
   tell me whether it's green.

5. cron-job.org: tell me the schedule, URL and request body of every "sixdegrees" job (I expect two upload jobs at
   11:22 and 18:22 America/Chicago with body {"ref":"main","inputs":{"max":"1","privacy":"public"}}). Don't change
   them. Don't reveal the Authorization header.

6. Finish by telling me which steps you completed, the channel id you confirmed in 2e, the result of 2f, and the
   stats run result. Don't include any secret values.

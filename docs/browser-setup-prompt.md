Set up automatic YouTube uploads for my new "Six Degrees" Shorts channel, reusing what I already set up for my blocky-stories channel. Do the steps in order. Secret values (client secret, refresh token, GitHub token) are copied from one page and pasted into another only. Never type them into this chat, a GitHub issue, a commit or any other page. If anything doesn't match what I describe, stop and tell me instead of guessing.

1. Google Cloud (console.cloud.google.com): open my existing "blocky-stories" project → APIs & Services (Google Auth Platform) → Clients. Open the existing Web OAuth client, the one whose authorized redirect URI is https://developers.google.com/oauthplayground. Keep the tab open so you can copy its Client ID and Client secret in step 2. Change nothing here.

2. OAuth Playground (https://developers.google.com/oauthplayground):
   a. Gear icon → tick "Use your own OAuth credentials" → paste that Client ID and Client secret. Set Access type: Offline, Force prompt: Consent Screen.
   b. In "Input your own scopes", paste exactly:
      https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/youtube.force-ssl
      Click "Authorize APIs".
   c. When Google asks which account or channel to use, pick the SIX DEGREES channel, not Buhtown. If you can't tell which is which, stop and ask me. Click through the "Google hasn't verified this app" warning (Advanced → continue). It's my own app.
   d. Click "Exchange authorization code for tokens". Keep the Refresh token in this tab for step 4.
   e. Check: in Step 3, request GET https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true and confirm the channel title is my Six Degrees channel. Note its "id" (it starts with UC). If it shows Buhtown, start step 2 again and pick the other channel.

3. GitHub repo evanlin23/sixdegrees → Settings → Actions → General → Workflow permissions: select "Read and write permissions" → Save.

4. GitHub repo evanlin23/sixdegrees → Settings → Secrets and variables → Actions:
   - Secrets tab → New repository secret, three times:
     YT_CLIENT_ID = the client ID from step 1
     YT_CLIENT_SECRET = the client secret from step 1
     YT_REFRESH_TOKEN = the refresh token from step 2d
   - Variables tab → New repository variable:
     YT_CHANNEL_ID = the UC... id from step 2e

5. A GitHub token for the cron service: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens. If the token named "cron-job.org dispatch" exists, edit it: under Repository access, add evanlin23/sixdegrees next to blocky-stories, keep Actions: Read and write, and save. Otherwise generate a new token named "cron-job.org sixdegrees", Only select repositories → evanlin23/sixdegrees, Repository permissions → Actions: Read and write, nothing else, 90 days. You'll paste it into cron-job.org in step 6. (When you edit an existing token, its value doesn't change, so the existing cron-job.org jobs keep working.)

6. cron-job.org (log into my existing account). Open my existing "blocky upload" job and use its Clone action. That copies the headers and token without anyone retyping them; only do this if the token from step 5 was the edited existing one. Otherwise create a new job and paste the new token. The new job:
   - Title: sixdegrees upload
   - URL: https://api.github.com/repos/evanlin23/sixdegrees/actions/workflows/upload.yml/dispatches
   - Schedule: every day at 6:52, time zone America/Chicago
   - Advanced → Request method: POST
   - Headers: Accept: application/vnd.github+json, Authorization: Bearer <the token>, X-GitHub-Api-Version: 2022-11-28
   - Request body: {"ref":"main"}
   - Make sure the job is enabled (clones start disabled).
   - Don't press "Test run" yet: the workflow only exists once the sixdegrees code is merged to main. Tell me when you're done, and Claude will run it once the merge is in.

7. Finish by telling me which steps you completed and the Six Degrees channel title you confirmed in step 2e. Don't include any secret values.

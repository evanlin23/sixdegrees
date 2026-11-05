# YouTube upload setup (one time, about 10 minutes)

This is the same approach as `blocky-stories`: the official YouTube Data API with an
OAuth refresh token. No audit is needed, since API uploads aren't locked to private
(YouTube confirmed this on 2026-11-02 for the other channel). Videos go up private
with a scheduled publish time and go public on their own.

## 1. Google Cloud: reuse the blocky-stories project
You can reuse the existing OAuth client (client ID and secret) from the
`blocky-stories` Google Cloud project. Only the refresh token is per-channel.

If you'd rather use a separate project, follow steps 1-4 of
`blocky-stories/docs/youtube-setup.md` (enable YouTube Data API v3, publish the
consent screen "In production", create a Web OAuth client with the redirect URI
`https://developers.google.com/oauthplayground`).

## 2. Refresh token for the Six Degrees channel
At https://developers.google.com/oauthplayground:
1. Gear icon → "Use your own OAuth credentials" → paste the client ID and secret.
   Access type **Offline**, Force prompt **Consent Screen**.
2. Scopes: `https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/youtube.force-ssl`
   → Authorize → **pick the Six Degrees channel** (not Buhtown) → click through "unverified app".
3. Exchange the authorization code for tokens and copy the **refresh token**.
4. Note the channel id: YouTube Studio → Settings → Channel → Advanced, the `UC...` id.

## 3. Put them in the Claude cloud environment (not the repo, not the chat)
In the session's title bar: cloud environment menu → **Edit** → environment variables:

```
YT_CLIENT_ID=...
YT_CLIENT_SECRET=...
YT_REFRESH_TOKEN=...
YT_CHANNEL_ID=UC...
```

New sessions pick them up. `python -m sixdeg upload <slug>` refuses to upload if the
token belongs to any other channel.

## Schedule and quota
Two slots a day, 11:30 AM and 6:30 PM Central, each with 0-20 min of jitter. An
upload costs about 1,600 of the free 10,000 daily quota units, so two a day is fine.

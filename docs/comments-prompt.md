# New comments and viewer requests: task for Claude in Chrome

The upload job already saves comments daily (`python -m sixdeg comments` → the `comments` branch, with replies, the
@handle and a 240 px profile picture). This browser task is a manual top-up when you want requests *now*, or if the
API run missed some. Paste everything below the line into Claude in Chrome while signed in to the account that owns
@buhconnects. Then paste its JSON answer into the Buh Connects session.

---

Collect new viewer comments for my YouTube channel @buhconnects. **Only read.** Don't reply, heart, pin, hide, delete
or change anything. Stop and ask me if a page wants a password or a code. Make sure the channel switcher (top right)
says "Buh Connects".

1. **Open the comments.** Go to https://studio.youtube.com → Comments.
   - Remove the default "Unresponded" filter so every comment shows.
   - Sort by newest.
   - Include **replies**: viewers often answer our pinned "Who should I connect next?" comment with a reply, so expand
     every "replies" thread.
2. **Collect** every comment from the last **7 days** (or since the date I give you), on every video. Leave out the
   channel's own comments (@buhconnects).
3. **For each comment, record:**
   - **handle:** the commenter's @handle exactly as shown, e.g. "@Issaiah-x5r", with the "@" and any digits. This is
     how we credit them, so copy it character for character. If Studio shows only a display name, open their channel
     link and copy the @handle from the channel URL.
   - **avatar:** the profile picture's image URL. Right-click the round profile picture → "Copy image address", or read
     the `<img>` src. It looks like `https://yt3.ggpht.com/…=s48-c-k-c0x00ffffff-no-rj`. Change the `=s48` (or `=s88`)
     part to `=s240` so we get a sharp version.
   - **text:** the full comment text, word for word, including emoji.
   - **likes:** the number shown (0 if none).
   - **video:** the video's title, and its ID (the part after `/video/` in the video's Studio link).
   - **when:** the date or "x days ago" as shown.
   - **is_reply:** true if it's a reply. If so, add **reply_to**: the @handle of the comment it answers.
4. **Flag requests.** Set "request": true for any comment that asks us to connect two people, characters or things,
   e.g. "Connect X to Y", "X to Y next", "do X and Y", "how is X connected to Y". Add **pair** as
   `["X", "Y"]`, written as the names people know.
5. **Answer with ONE JSON code block**, nothing left out:

```json
{
  "collected_at": "YYYY-MM-DDTHH:MM, with the time zone",
  "comments": [
    {"handle": "@someone", "avatar": "https://yt3.ggpht.com/...=s240-c-k-c0x00ffffff-no-rj", "text": "...",
     "likes": 0, "video": {"title": "...", "id": "..."}, "when": "...", "is_reply": false, "reply_to": null,
     "request": true, "pair": ["Abraham Lincoln", "Tiger Woods"]}
  ],
  "couldnt_read": ["anything you couldn't open or read, and why"]
}
```

Then list the requests on their own, most-liked first, as `@handle: X → Y (likes)`.

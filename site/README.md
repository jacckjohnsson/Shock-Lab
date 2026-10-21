# Shock Lab landing page

A single static page (`index.html` plus `assets/`) with no build step. It links to the Streamlit app.

## Set the app link

Once the app is deployed, put its address in the one line near the bottom of `index.html`:

```js
const APP_URL = "https://shock-lab.streamlit.app";
```

Every "Launch" button uses it.

## Preview locally

```bash
cd site
python3 -m http.server 8000   # then open http://localhost:8000
```

## Host it for free

The repo is private, so GitHub Pages would need a paid GitHub plan. Netlify, Vercel and Cloudflare Pages host
static sites from a private repo for free:

1. Sign in with GitHub and import `jacckjohnsson/Shock-Lab`.
2. Set the **publish (output) directory** to `site`. There is no build command.
3. Deploy. Every merge to `main` redeploys automatically, and you can add a custom domain later.

## Screenshots

The images in `assets/` were captured from the app running on the made-up sample data, so the page never
shows anyone's real holdings. Re-capture them after visual changes to the app.

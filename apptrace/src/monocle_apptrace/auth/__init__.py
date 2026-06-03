"""Browser-based sign-in for the Monocle CLI.

Flow when the user picks "Sign in" during `monocle-apptrace claude-setup`:

  1. CLI opens the browser to Auth0 (auth.okahu.co)
  2. User signs in (GitHub, etc.)
  3. Auth0 redirects to a one-shot HTTP server on localhost:18292
  4. CLI exchanges the auth code for a JWT using PKCE
  5. CLI calls Okahu's API with the JWT to mint an OKAHU_API_KEY
  6. Key is written to ~/.monocle/.env; JWT is discarded

Entry point: portal_auth.resolve_okahu_api_key().
"""
